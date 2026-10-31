# 第 4 周：进阶专题（一）—— 量化与投机解码（扩展版）

> **本周目标**：两个高频专题达到"原理 + 场景 + 权衡 + 失效模式"四段式水平。
> **衔接关系**：Week 1 的 roofline/显存手算 → 本周的量化收益推导；Week 2-3 的调度与 KV 管理 → 本周 mini 引擎收尾；本周量化/投机专题 → Week 6-7 项目 A（vllm-ascend 量化 GEMM）与项目 C（消融实验）的素材。
> **全程以 vLLM V1（分离 EngineCore 进程架构）为准，V0 代码不要读。**

---

## 本周总览

| Day | 主题 | 关键产出 |
|---|---|---|
| 22 | 量化基础串讲 | 《昇腾量化算子 vs GPU 量化 GEMM 对照》一页 |
| 23 | KV cache 量化 | FP8 KV 实验记录（吞吐/显存/精度三列对比） |
| 24 | 量化动手 + 总结 | 专题 A4《量化：原理/场景/权衡/失效模式》 |
| 25 | 投机解码原理 | 收益公式推导笔记 + 三路线对比表 |
| 26 | 投机解码实验 | 接受率-γ 调参曲线，验证负收益失效模式 |
| 27 | mini 引擎收尾（项目 B） | chunked prefill + preemption + benchmark 报告 |
| 28 | 复盘日 | 专题 A4《投机解码》+ 口头自测 |

---

## Day 22：量化基础串讲

### 1. 统一数学框架：一切量化都是"缩放 + 取整 + 截断"

```
量化:   x_q = clamp(round(x / scale) + zero_point, qmin, qmax)
反量化: x̂  = (x_q - zero_point) × scale
```

- **对称量化**（zero_point=0）：适合权重（近似零均值）；INT8 范围 [-127,127]。
- **非对称量化**（zero_point≠0）：适合激活（ReLU 后非负分布）。
- 量化误差 = 量化噪声（round 带来，~均匀分布）+ 截断误差（outlier 超范围被 clip）。
  - 量化噪声是"良性的"：per-channel 下信噪比 ≈ 6.02·bit dB；
  - 截断误差是"恶性的"：少数 outlier 会迫使 scale 变大，拖垮所有正常值的精度——这是激活量化的核心难点（见第 4 节）。

### 2. 粒度（granularity）光谱

| 粒度 | scale 数量 | 精度 | kernel 实现代价 |
|---|---|---|---|
| per-tensor | 1 | 最差（一个 outlier 毁全部） | 最简单，FP8 tensor core 原生支持 |
| per-channel（weight 按输出通道） | out_features | 好 | 权重侧标配，几乎零代价 |
| per-token（activation 按行） | batch×rows | 好 | 激活侧标配，动态计算 |
| per-block / group-wise（如 group=128） | 非常多 | 最好（W4 的救命稻草） | 反量化逻辑复杂，需融合进 GEMM（Marlin/Machete/AWQ kernel） |
| per-token-per-channel（2D） | rows×cols | 最好 | 只在 dequant 路径可行 |

**记忆锚点**：粒度越细，outlier 被隔离得越好；但 scale 元数据越多，kernel 越难写。GPU 上 per-channel/per-token 是"免费"的，group-wise 是 W4A16 专属的昂贵选项。

### 3. 三大部署形态：W8A8 / W4A16 / FP8

| 形态 | 权重 | 激活 | 收益来源 | 主要场景 |
|---|---|---|---|---|
| **W8A8 (INT8)** | INT8 per-channel | INT8 per-token | 算力：INT8 tensor core ≈ 2× BF16 峰值 | prefill 重、吞吐优先 |
| **W4A16** | INT4 group-wise | BF16 | 带宽：decode 权重读取量 ÷4 | decode 重、显存紧张 |
| **FP8 (E4M3)** | FP8 per-channel | FP8 动态 per-token | 算力 + 带宽双 2×（Hopper 起） | 当前生产默认推荐 |

- **W4A16 的本质**：不动激活（避开 outlier 难题），只压权重。decode 是 memory-bound，权重占访存大头 → W4 理论 TPOT 减为 1/4，实际 1.5~2.2×（受激活/KV/反量化开销限制）。GEMM 路径是"边反量化边算"（weight-only fused dequant GEMV/GEMM）。
- **FP8 的本质**：E4M3（4 指数 3 尾数，范围 ±448，相对精度 ~2⁻³）对 LLM 权重/激活分布几乎无量化难点，不需要 SmoothQuant 这类预处理——**FP8 赢在"工程简单性"**，这是它取代 INT8 W8A8 的根本原因。
- **E4M3 vs E5M2**：E5M2 范围大（±57344）但尾数只有 2 位，精度差；LLM 推理权重/激活/KV cache 主流用 E4M3。

### 4. 激活 outlier 与三大 PTQ 算法

LLM 激活存在**系统性离群通道**（固定 channel、幅度可达正常值 100×，涌现于 >2.7B 模型）：per-tensor INT8 直接被它们打崩。三条解法路线：

| 算法 | 形态 | 核心思想 | 一句话记忆 |
|---|---|---|---|
| **SmoothQuant** | W8A8 | 等效变换 `Y=(X·s⁻¹)·(s·W)`，把激活的量化难度按幅度比例迁移到权重侧（激活难量化、权重好量化） | "搬家" |
| **AWQ** | W4A16 | 观察 ~1% 的"重要通道"（激活幅度大），per-channel scale 保护它们；不依赖反传，仅等效缩放 | "保重点" |
| **GPTQ** | W2~W8 | 逐列量化 + 基于该层 Hessian 的误差补偿（把当前列误差修到剩余列），最小化逐层输出重建误差 | "边量边修" |

**vLLM 实操口径**：官方首选 llm-compressor（compressed-tensors 格式），见 Day 24。

### 5. Roofline 视角的收益推导（面试必考，必须能手推）

**Decode（memory-bound）**：
- 单 token 理论时延 ≈ `参数量 × 权重字节 / HBM 带宽`
- Llama-3-70B @ H100（3.35 TB/s）：BF16 → 70×2/3350 ≈ **41.8 ms/token**；FP8 → **20.9**；INT4 → **10.4**（理论上限，实际打折）
- 算术强度：decode GEMV 的 AI ≈ `2/dtype_bytes`（BF16 约 1 FLOP/Byte），离 H100 ridge point（BF16 ≈ 295、FP8 ≈ 591）差两个数量级 → **decode 时量化收益全部来自带宽减半/减四分之一，与算力无关**

**Prefill（compute-bound）**：
- AI 高（≈ 2×seq_len/dtype_bytes），吃 tensor core → **量化收益来自算力翻倍（INT8/FP8）**；W4A16 对 prefill 无算力收益，只有显存/容量收益
- 这就是"**量化对 TTFT 和 TPOT 的影响是两条独立曲线**"的第一性原因

**Batch 的影响**：batch=B 时权重只读一次，AI ≈ `2B/dtype_bytes`；B 增大 decode 逐渐从 memory-bound 滑向 compute-bound → **量化的 TPOT 收益随 batch 增大而衰减**（这也是"高并发下 W4A16 可能不如 FP8"的原因）。

### 6. 昇腾经验对照表（你的差异化素材，务必整理成一页）

| 维度 | 昇腾（你的经验） | GPU（vLLM 生态） |
|---|---|---|
| 量化算子形态 | WeightQuantBatchMatmulV2：量化 GEMM + 反量化**算子级融合**，一条算子出结果 | 融合下沉到 kernel/epilogue：Marlin（W4A16）、CUTLASS FP8 GEMM、CUTLASSSmoothQuant |
| 多精度路径 | INT4/INT8/FP8/FP16/BF16 指令序列择优（Cube 指令选型） | CUTLASS 模板按 dtype 实例化；torch._scaled_mm（FP8） |
| 带宽/算力建模 | L2/HBM 带宽与 Cube 算力的 bound 分界模型 + balanceRate≥0.9 剪枝搜优 baseM/baseN | ncu 看 SM busy / DRAM busy；roofline 手算 |
| 数据搬运优化 | ASW 蛇形滑窗提升 L2 命中；L1 全载模板（A/权重驻留 L1，重复搬运 O(n·A)→O(A)） | 共享内存 swizzle / L2 persistence；权重常驻对 decode GEMV 天然友好 |
| 流水线 | Fixpipe / 无 Queue 手工流水（SetFlag/WaitFlag 事件驱动 L0A/L0B/L0C 乒乓） | cp.async / TMA 多级流水，software pipelining |
| 量化粒度 | 算子内 per-tensor/channel 反量化路径 | per-channel weight + per-token activation + group-wise W4 |

**面试话术主线**：我在昇腾上做过"量化 GEMM 的访存/计算 bound 建模 + tiling 搜优 + L1 驻留优化"，这套方法论映射到 GPU 就是 roofline 分析 + kernel 选型（Marlin vs CUTLASS）+ batch 维度的收益衰减建模——**平台变了，方法论不变**。

### 7. 面试问答（Day 22 自测）

- Q：为什么 W4A16 加速比到不了 4×？
  A：① 激活与 KV cache 仍是 BF16，非权重访存占比随 batch 上升；② 反量化/depack 开销（group-wise scale 读取）；③ 大 batch 时滑入 compute-bound，瓶颈从带宽转向算力；④ 部分 layer（lm_head、norm）不量化。
- Q：为什么 FP8 比 INT8 W8A8 更受欢迎？
  A：E4M3 指数位保留了动态范围，对 outlier 天然鲁棒，免去 SmoothQuant 校准环节；Hopper 起 FP8 tensor core 原生支持 per-tensor scale，工程链路最短。
- Q：量化对 TTFT 和 TPOT 影响分别是什么？
  A：TPOT——W4/FP8 权重带宽减半以上，decode 直接受益，batch 越小收益越大；TTFT——只有 W8A8/FP8（算力翻倍）受益，W4A16 基本不变甚至略劣（dequant 开销）。

### 8. 今日产出

- [ ] 一页《昇腾量化算子 vs GPU 量化 GEMM 对照》（第 6 节表格扩写）
- [ ] 手推：Llama-3-70B 三种精度下的 decode 时延下界、batch=64 时的 AI 与 ridge point 距离

---

## Day 23：KV Cache 量化

### 1. 为什么 KV cache 是量化的"第二战场"

公式（必须条件反射）：**每 token KV 显存 = `2 × layers × kv_heads × head_dim × dtype_bytes`**（2 = K 和 V；GQA 用 kv_heads 不是 q_heads）

| 模型 | layers | kv_heads | head_dim | 每 token 元素 | BF16 | FP8 | INT4 |
|---|---|---|---|---|---|---|---|
| Llama-3-70B | 80 | 8 | 128 | 163,840 | 320 KB | 160 KB | 80 KB |
| Qwen3-8B | 36 | 8 | 128 | 73,728 | 144 KB | 72 KB | 36 KB |

- Llama-3-70B 单条 100K 上下文 = 32 GB（BF16）——**KV 显存是长上下文/高并发的第一瓶颈**。
- KV FP8 化 = 同显存下并发×2 或上下文×2；对应 `/metrics` 里 `gpu_cache_usage` 下降、`num_preemptions` 归零、goodput 上升。
- 关键联动（串起 Week 2/3 知识）：KV 量化 → 每 block 字节数不变（block 数由 token 数决定）但**物理显存占用减半** → KVCacheManager 的可用 block 数×2 → scheduler 能容纳的 running batch 更大 → 吞吐上升。这就是"量化收益要顺着调度链路传导"的分析框架。

### 2. 精度代价：K 比 V 敏感

- **K 的误差被 QK^T 点积放大**，再经 softmax 成为 attention 权重偏差 → 直接改变"看哪里"；V 的误差只是线性混合进输出。
- 经验排序（精度敏感度）：**K > V**；且 K 也有固定通道 outlier（与激活 outlier 同源）。
- INT8 KV：per-block/per-head scale 下大多任务损失可忽略；INT4 KV：需要更细粒度 + outlier 处理，代码/数学类任务慎用。
- FP8 (E4M3) KV：精度损失几乎测不出（ppl 变化 <0.1%），**生产默认推荐**。

### 3. 什么负载适合开

| 负载特征 | 是否适合 KV 量化 | 原因 |
|---|---|---|
| 长上下文（RAG、长文档总结、Agent 多轮） | 强烈推荐 | KV 是绝对瓶颈，收益 = 并发/上下文翻倍 |
| 高并发短输出（chat 泛滥负载） | 推荐 | preemption 减少、batch 上限提高 |
| 短上下文 + 精度敏感（代码补全、数学） | 谨慎 | 收益小，INT4 有精度风险；FP8 可用 |
| prefix caching 重度依赖 + 多租户 | 可用但注意 | 缓存的是量化后 KV，跨精度不兼容；命中率统计不变 |

### 4. 实验：Qwen3 FP8 + KV FP8

```bash
# 基线：BF16 权重 + BF16 KV
vllm serve Qwen/Qwen3-8B \
  --max-model-len 32768 \
  --max-num-seqs 256

# 权重 FP8（官方 checkpoint）+ KV FP8
vllm serve Qwen/Qwen3-8B-FP8 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --kv-cache-dtype fp8
```

压测（沿用 Day 6 的负载，控制变量）：

```bash
vllm bench serve --model Qwen/Qwen3-8B-FP8 \
  --dataset-name sharegpt --dataset-path ShareGPT_Vicuna_unfiltered_5k.json \
  --request-rate 8 --num-prompts 500
```

**采集三列对比**（BF16 / FP8 权重 / FP8 权重+FP8 KV）：
1. **显存与容量**：启动日志中 "GPU KV cache size: N tokens"（应×2）；`vllm:gpu_cache_usage` 曲线
2. **性能**：TTFT p99、TPOT p99、总吞吐；重点观察高 request-rate 段的拐点右移（goodput 提升）
3. **精度**：`lm_eval --model vllm --model_args pretrained=...,dtype=float16 --tasks gsm8k,hellaswag` 对比分差

**预期现象**（写进实验记录的"机制解释"）：
- 短上下文低并发段：TPOT 几乎不变（KV 不是瓶颈）
- 挤压段（KV 接近满）：无 KV 量化时 preemption 计数上升、TPOT p99 尖刺；开 KV FP8 后尖刺消失——**收益主要体现在尾延迟与 goodput，而不是平均吞吐**

### 5. 面试问答

- Q：KV cache 量化为什么对 TTFT 也可能有收益？
  A：prefill 也要写 KV；更关键的是并发容量翻倍后队列等待（排队论中的 ρ 下降）缩短，TTFT p99 显著改善。
- Q：KV FP8 与 prefix caching 交互的坑？
  A：缓存的 block 是量化后的字节，draft/quant 精度必须一致才能复用；块 hash 不含 dtype 时要靠部署约束保证同实例同精度。

### 6. 今日产出

- [ ] 实验记录：三列对比表 + "现象 → 源码机制 → 指标表现"三段对照
- [ ] 手算：Qwen3-8B 在 80GB 卡上，BF16 KV vs FP8 KV 的最大并发（给定平均上下文 16K）

---

## Day 24：量化动手 + 专题总结

### 1. 用 llm-compressor 产出自己的量化模型（三条 recipe）

```bash
pip install llmcompressor
```

**(a) FP8 动态量化**（最短路径，校准只需少量数据）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(512))

recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)
oneshot(model=model, dataset=ds, recipe=recipe,
        output_dir="./Qwen3-8B-FP8-own", max_seq_length=2048)
model.save_pretrained("./Qwen3-8B-FP8-own")
tokenizer.save_pretrained("./Qwen3-8B-FP8-own")
```

**(b) W4A16 GPTQ**：`recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])`（`from llmcompressor.modifiers.gptq import GPTQModifier`）

**(c) W8A8 SmoothQuant**：两段式 `recipe = [SmoothQuantModifier(smoothing_strength=0.8), QuantizationModifier(targets="Linear", scheme="W8A8_DYNAMIC")]`

**调参直觉**：calibration 数据要贴近业务分布（代码业务就用代码语料）；512 条通常够；`ignore` 掉 lm_head 与极端敏感层。

### 2. 验证闭环（精度 + 性能）

```bash
# 精度
lm_eval --model vllm --model_args pretrained=./Qwen3-8B-FP8-own --tasks gsm8k,arc_challenge --batch_size auto

# 性能（对比 BF16 基线）
vllm bench serve --model ./Qwen3-8B-FP8-own \
  --dataset-name sharegpt --request-rate 4 --num-prompts 300
```

**验收标准示例**（写进总结，避免"感觉还行"）：GSM8K 掉点 <1.5%；小 batch TPOT 提升 ≥1.5×；TTFT 不劣化超 5%。

### 3. A4 专题总结模板《量化：原理/场景/权衡/失效模式》

- **原理**（1/4 页）：统一公式 + 粒度光谱 + outlier 难点 + 三算法一句话
- **场景**（1/4 页）：决策树——decode 重/显存紧 → W4A16；追求最短工程链路 → FP8；prefill 重且老硬件 → W8A8+SmoothQuant；KV 瓶颈 → KV FP8
- **权衡**（1/4 页）：两条独立曲线（TTFT 走算力、TPOT 走带宽）+ batch 增大收益衰减 + 精度-吞吐 Pareto 图（自己实验的数据点画上去）
- **失效模式**（1/4 页）：
  1. 校准分布偏移 → 上线掉点（重新校准）
  2. 大 batch 下 W4 收益消失甚至为负（depack 开销）→ 切 FP8
  3. KV INT4 在代码/数学任务精度崩 → 降回 FP8
  4. MoE 模型专家权重稀疏激活 → W4 收益打折（专家被读的频率不均）
  5. 量化与 CUDA Graph/piecewise compilation 的兼容矩阵要过一遍

### 4. 今日产出

- [ ] 自制 FP8 / W4 模型各一个，跑通精度+性能双验证
- [ ] A4 总结《量化》定稿（这是 Week 8 白板四件套的素材）

---

## Day 25：投机解码原理

### 1. 第一性原理：为什么"用计算换访存"是赚的

- Decode 单步只生成 1 token，却要读**全部权重 + 该 seq 的全部 KV** → 算力利用率 <1%（AI ≈ 1 FLOP/Byte vs ridge ~300）。
- 投机解码：draft 先便宜地猜 γ 个 token，target **一次 forward 并行验证 γ+1 个位置**——验证时 batch=γ+1 的 GEMM 权重只读一次，AI 提升 γ+1 倍，把闲置算力换成访存节省。
- **本质**：decode 的浪费在"每 token 串行读一遍权重"；投机解码把"逐 token 串行"改成"块内并行验证"，摊薄了每次访存的 token 产出。
- 类比你熟悉的流水线视角：draft 是"预取"，target 验证是"批量确认"。

### 2. 收益公式（面试要能现场推）

设每 token 独立接受率 α，草稿长度 γ，draft 单步成本为 target 的 c 倍。一次迭代的：

- **期望产出 token 数**：`E = (1 - α^(γ+1)) / (1 - α)`
- **迭代成本**：`T = 1 + γ·c`（target 验证 ≈1 步：memory-bound 下读同样权重；draft 走 γ 步）
- **加速比**：`S ≈ E / T`

代入体感数字：
- α=0.8, γ=4, c=0.05 → E≈3.36, T=1.2 → **S≈2.8×**
- α=0.8, γ=8, c=0.05 → E≈3.92, T=1.4 → S≈2.8×（γ 翻倍收益饱和——E 有上界 1/(1-α)）
- α=0.4, γ=4, c=0.05 → E≈1.63, T=1.2 → S≈1.36×；若 c=0.15 → S≈1.05×，**接近零收益**

**三个关键结论**：① α 是一切，α<0.5 基本不值得开；② γ 存在最优值（再大只加成本不加产出，还浪费 KV block 与验证算力）；③ batch 很大时 target 本身已 compute-bound，验证的"并行免费"消失 → 投机收益衰减。

### 3. 三条（+1）路线对比

| 路线 | draft 来源 | 优点 | 缺点 | 代表 |
|---|---|---|---|---|
| **小模型 draft** | 独立小 LM（68M→70B） | target 不用重训 | 需维护第二个模型；domain 不匹配 → α 低 | 早期 classic speculative |
| **MTP** | target 自带的多 token 预测头（训练时让模型学会同时预测未来 token） | 同分布 α 高；无第二模型 | 需要训练期改造 | DeepSeek-V3（1 个 MTP 头，报告接受率 ~85% 量级） |
| **EAGLE-3** | 轻量 draft head，在**特征层**（hidden state）自回归 + 多层特征融合（低/中/高层拼接）、直接回归最终特征、加深加宽 draft 结构 | 训练成本低（几 GPU 时）；接受长度 ~5 token/步 | 推理侧需要树形注意力与 KV 管理 | EAGLE-3（vLLM `method: eagle3`） |
| **n-gram（免费午餐）** | 从 prompt/上下文里 n-gram 查表当草稿 | **零额外模型零额外计算** | 只对重复性负载有效（代码、RAG 引用、摘要） | vLLM `method: ngram` |

**MTP vs EAGLE 的深层差异**：MTP 从 token 概率出发学"未来"，EAGLE 从特征出发学"未来"——特征层自回归误差更小（draft 看到的是 target 已充分计算的表达），这是 EAGLE 系列接受率高的根本原因。

### 4. 验证机制怎么落地（联系 Week 3 的 attention 后端）

1. **并行 scoring**：target 一次 forward 算出 γ+1 个位置的 next-token 分布（注意这是 prefill-like 的 batch 维并行，不是 γ 次串行 decode）
2. **rejection sampling**：逐位置比较 draft token 与 target 分布——draft token 以 `min(1, p_target/p_draft)` 的概率被接受（保证输出分布与 target 独立采样**完全一致**，理论无损）
3. **树形草稿**（EAGLE 系）：draft 不是链而是树 → 需要 **tree attention mask**（同一 batch 里让不同分支只看自己的祖先）；vLLM 里对应 draft token 的 KV 也按 paged block 管理，验证后**回滚未接受分支的 KV**——这里与 KVCacheManager/free block 逻辑强耦合，是源码走读点
4. **CUDA Graph 适配**：验证步的 batch 形状随 γ 变化 → capture 时按 γ+1 的 bucket 处理

### 5. 面试问答

- Q：投机解码改变输出分布吗？
  A：不改变。rejection sampler 的接受规则保证联合分布与 target 自回归采样逐 token 一致（有精确证明），这是"无损加速"的来源；但注意 temperature=0 与采样路径在实现上要分别验证。
- Q：EAGLE 为什么比小模型 draft 接受率高？
  A：draft 复用 target 的 hidden state（信息更充分），在特征层做自回归误差更小；且树形草稿保留了多候选取舍空间。
- Q：γ 怎么选？
  A：由 α 和 c 决定，α 高（>0.8）选 γ=3~5；α 中等选 γ=1~2；也可以像 vLLM 的动态投机/自适应 γ 方向那样在线根据接受率调（项目 C 消融素材）。

### 6. 今日产出

- [ ] 收益公式推导笔记（含上面三组数字代入）
- [ ] 四路线对比表 + "验证-回滚 KV"机制的一段白板讲解稿

---

## Day 26：投机解码实验

### 1. 实验矩阵（核心变量：负载 × γ）

- **负载 A（高接受率）**：代码补全（HumanEval 风格 prompt / 代码仓库片段）、RAG 引用式问答——草稿与原文重复度高
- **负载 B（低接受率）**：开放对话、高 temperature 创作生成
- **γ 扫描**：1 / 2 / 3 / 4 / 5 / 8

### 2. 命令（vLLM V1）

```bash
# EAGLE-3（以 Llama 3.1 8B + 官方 EAGLE3 头为例）
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 4
  }'

# DeepSeek MTP
vllm serve deepseek-ai/DeepSeek-V3 \
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 3}'

# n-gram（零成本，先拿它练手建立体感）
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4}'
```

压测用 `vllm bench serve`，固定 request-rate 与 seed，只改 `num_speculative_tokens`。

### 3. 指标采集

- **接受率**：`/metrics` 中 `vllm:spec_decode_*` 系列（accepted / drafted tokens 计数），acceptance = accepted/drafted；版本无指标时从 engine 日志自行打点
- **性能**：TPOT p50/p99、单序列 tokens/s、总吞吐；注意记录 **batch 水位**（投机收益随并发衰减，要分低/中/高并发三段）
- **验收口径**：对每组 (负载, γ) 记录 (接受率, TPOT 加速比, 吞吐变化) 三元组

### 4. 预期现象与失效模式（本周最重要的实验结论）

| 现象 | 机制解释 |
|---|---|
| 负载 A：α≈0.7-0.9，TPOT 加速 2-3× | 高重复 → 草稿命中率高 |
| 负载 B：α≈0.3-0.5，γ=4 时吞吐**低于**不开 | 验证浪费 + draft 开销 + 未接受草稿的 KV 分配/回滚开销 |
| γ 从 4→8：加速比几乎不变甚至下降 | E 有上界 1/(1-α)；T 线性涨 |
| 高并发段：加速比明显衰减 | target 已 compute-bound，验证不再"免费" |
| n-gram 在代码负载可白拿 1.5-2× | 草稿零成本（c≈0），哪怕 α 一般也赚 |

### 5. 今日产出

- [ ] 接受率-γ 曲线（两种负载各一条）+ 加速比-并发曲线
- [ ] 用 Day 25 公式拟合实验数据，标出预测 vs 实测偏差及原因（独立接受假设不成立等）

---

## Day 27：mini 引擎收尾（项目 B）

本周给 Week 3 起步的 mini 引擎补上最后两块调度能力，并完成对比实验。

### 1. Chunked prefill 实现要点

- 每个 step 有 token budget（如 512）：waiting 队列头部的长 prompt 切成 ≤budget 的块，本 step 只算第一块；剩余部分记录 `prompt_tail`，下个 step 继续（请求保持 running）
- decode 与 prefill chunk 共享同一 step：budget 内先排 prefill chunk 再排 decode
- **注意**：被 chunk 的请求其 KV 已写入的 block 要保留（部分前缀已算过）；与 prefix caching 的 block hash 交互——完整前缀命中才能复用，部分块不进 hash 表
- 思考题自答：为什么 chunked prefill 能降 TPOT 抖动？（prefill 大块不再独占 step，decode 每步都有机会被调度；代价是 TTFT 略升、总 FLOPs 不变）

### 2. Preemption 实现要点（recompute 模式）

- 触发：`allocate_slots` 失败（free block 不足且无 prefix 可驱逐）
- victim 选择：默认**从 running 队列尾部**（最新进入的）开始抢
- recompute 模式：释放该 seq 全部 KV block 回 free pool，请求状态改 PREEMPTED，**重新放回 waiting 队列头部**，下轮从 prompt 重新 prefill（对照 Week 2 Day 12 的源码结论）
- 记录 `num_preemptions` 计数，benchmark 中与 TPOT p99 尖刺对齐
- swap 模式（进阶可选）：KV 搬到 CPU，恢复时搬回——对比两者适用场景（显存换时延）

### 3. Benchmark：static batching vs continuous batching

```python
# 负载生成：泊松到达，ISL/OSL 取对数正态（如 ISL~N(400,1.2), OSL~N(200,1.1)）
# 三组引擎：①static（凑满 batch 或 50ms 超时起跑，全批跑完才接新请求）
#          ②continuous（每 iteration 检查完成/新到达）
#          ③continuous + chunked prefill + preemption（你的完整版）
# 指标：总吞吐（tokens/s）、TTFT p50/p99、ITL p50/p99、preemption 次数
```

**预期结论**（写进项目 README）：
- OSL 方差越大，static 的气泡占比越高（最长的请求拖着整个 batch），continuous 优势可达数倍
- 高到达率下，③的 ITL p99 明显优于无 chunked prefill 的 ②（prefill 洪峰不再阻塞 decode）

### 4. 项目 README 结构（面试作品集素材）

1. 架构图：scheduler / kv manager / block pool / executor 模块关系
2. 已实现特性清单：continuous batching、block table + 引用计数、chunked prefill、preemption（recompute）
3. 性能对比表 + 三张曲线图（吞吐-到达率、TTFT 分布、ITL p99）
4. 已知局限与 vLLM 的差距（async scheduling、CUDA Graph、prefix caching……）——**主动写局限是面试加分项**

### 5. 今日产出

- [ ] mini 引擎功能收尾 + benchmark 数据表
- [ ] 项目 B 的 README 定稿

---

## Day 28（复盘日）

### 1. 产出专题 A4 总结《投机解码》

四段式结构（与《量化》对齐）：
- **原理**：memory-bound 换算力 + E/T/S 三公式 + rejection sampling 无损性
- **场景**：决策树——代码/RAG/Agent → 开（优先 n-gram 起步）；有 MTP 头的模型（DeepSeek 系）→ 开；开放创作 + 无现成 draft → 不开；高并发大 batch → 谨慎
- **权衡**：α-γ-c 三角；KV 管理与树形注意力的实现复杂度；与 CUDA Graph bucket 的耦合
- **失效模式**：α 低负收益 / γ 过大饱和 / 大 batch 收益衰减 / 草稿模型与 target tokenizer 不匹配 / 未接受草稿的 KV 回滚开销

### 2. 自测（口头录音，每题 3 分钟）

1. **投机解码什么时候是负收益？**（至少说出 4 种：α 低、γ 过大、c 大、target 已 compute-bound、KV 回滚开销占比高）
2. 给定 α=0.75、γ=3、c=0.05，口算加速比（E=(1-0.75⁴)/0.25≈2.73，T=1.15，S≈2.4）
3. 量化与投机解码可以叠加吗？收益怎么建模？（draft 也量化时 c 下降；两者都吃 decode 的冗余，收益非相乘而是取重叠后的较小者）
4. 从昇腾算子视角：投机解码对 GEMM 形态的影响是什么？（M 维从 1 变 γ+1，GEMV 变小 GEMM，正好落入你 L1 全载模板的 M≤256 甜点区——跨平台叙事素材）
5. 遍历本周所有实验数据，每条曲线能说出"现象 → 机制 → 指标"三段

### 3. 本周产出物核对

- [ ] 《昇腾量化算子 vs GPU 量化 GEMM 对照》一页（Day 22）
- [ ] KV FP8 实验三列对比记录（Day 23）
- [ ] 自制 FP8/W4 模型 + 双验证（Day 24）
- [ ] A4《量化：原理/场景/权衡/失效模式》（Day 24）
- [ ] 投机解码公式推导 + 四路线对比（Day 25）
- [ ] 接受率-γ 曲线与失效模式验证（Day 26）
- [ ] 项目 B README + benchmark 数据（Day 27）
- [ ] A4《投机解码》（Day 28）

---

## 附：本周面试高频问题速查（含答题要点）

| 问题 | 答题要点 |
|---|---|
| 量化对 TTFT / TPOT 的影响？ | 两条独立曲线：TTFT 吃算力（W8A8/FP8 翻倍），TPOT 吃带宽（W4/FP8 减半+）；batch 增大 TPOT 收益衰减 |
| 为什么 W4A16 到不了 4×？ | 非权重访存 + dequant 开销 + compute-bound 滑坡 + 未量化层 |
| FP8 为什么赢了 INT8？ | E4M3 动态范围对 outlier 鲁棒，免校准，工程链路短 |
| KV cache 量化收益看什么指标？ | GPU KV tokens 容量×2 → preemption↓ → goodput/尾延迟改善，而非平均吞吐 |
| 投机解码什么时候负收益？ | α 低 / γ 过大 / c 大 / 大 batch compute-bound / KV 回滚开销 |
| EAGLE-3 为什么强？ | 特征层自回归 + 多层特征融合 + 树形草稿；训练还便宜 |
| MTP 和 EAGLE 选谁？ | 模型自带 MTP 头用 MTP（零额外部署）；改造存量模型用 EAGLE；不确定负载先试 n-gram |
| chunked prefill 与投机解码有交互吗？ | 两者都在平滑 decode 与"批量计算"的关系；vLLM 中 chunked prefill 的 budget 也要覆盖验证步的 γ+1 token，预算分配要一起调 |

## 附：参考资料清单

- PagedAttention（SOSP'23）——Week 1 已读，本周回看 KV 量化相关讨论
- SmoothQuant (ICML'23)、GPTQ (ICLR'23)、AWQ (MLSys'24)——各读方法节 + 一张图
- SpecDec / Medusa / EAGLE-1/2/3 论文（重点 EAGLE-3 的特征融合与树形草稿）
- DeepSeek-V3 技术报告 MTP 章节（训练目标与部署接受率）
- vLLM 官方文档：Quantization（FP8/llm-compressor/compressed-tensors）、KV cache quantization、Speculative Decoding 三页
- llm-compressor 官方 examples（FP8 one-shot / GPTQ / SmoothQuant recipe）

## 附：与前后的衔接

- **回头看**：Day 22 的收益推导依赖 Week 1 的 roofline 手算；Day 26 的 KV 回滚机制依赖 Week 3 的 KVCacheManager 源码
- **向后看**：本周两份 A4 是 Week 8 白板素材；《量化对照表》是 Week 6-7 项目 A（vllm-ascend 量化 GEMM tiling 优化）的选题依据；接受率-收益曲线直接成为 Week 7 项目 C 消融实验 ③ 的设计蓝本
