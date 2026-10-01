# 第 1 周：推理基础与性能建模（第一性原理）· 展开手册

> **本周目标**：把 LLM 推理变成能手算的数学题，建立 GPU 版 Roofline 思维。
> **使用方法**：每天先读对应 Day 的「核心内容」，完成「动手任务」，用「自测题」检验，最后按「产出物」归档。
> **约定**：全程以 vLLM V1 架构为准（V0 已移除，不要读旧代码）。
> **背景衔接**：你已有昇腾 NPU 的访存/计算 bound 建模（`CalRebalanceBlock`）与量化算子经验，本周的任务是**把同一套第一性原理搬到 GPU 平台**。

## 本周知识地图

```
LLM 推理性能第一性原理
├── 两个阶段：prefill（计算密集）/ decode（访存密集）        ← Day 1
├── 两个公式：KV 显存公式 / decode 时延下界                  ← Day 2
├── 一个模型：Roofline（算力 vs 带宽的分界）                 ← Day 3
├── 一个系统：PagedAttention（KV 显存管理）                  ← Day 4
├── 一套指标：TTFT / TPOT / ITL / goodput                   ← Day 5
└── 一次实测：环境搭建 + 压测曲线                             ← Day 6
```

## 本周产出物检查单

| 产出物 | 对应 Day | 状态 |
|---|---|---|
| 笔记《prefill/decode 计算与访存量推导》 | Day 1 | [ ] |
| 3 道手算题完整推导过程 | Day 2 | [ ] |
| 笔记《从昇腾 bound 建模到 GPU Roofline 的映射表》 | Day 3 | [ ] |
| PagedAttention 论文精读笔记（含 3 个最巧设计点） | Day 4 | [ ] |
| 指标定义卡片（面试抽背版） | Day 5 | [ ] |
| 第一张性能曲线图（并发 vs TTFT/TPOT p99） | Day 6 | [ ] |
| 《LLM 推理性能的第一性原理》一篇 + 自测 | Day 7 | [ ] |

---

# Day 1：Transformer 推理机制

- [ ] 精读 prefill vs decode 的本质区别
- [ ] 搞懂 KV cache 的生成与复用，手画一次 decode 迭代的张量流

## 1.1 自回归解码的两个阶段

LLM 推理 = 自回归地逐 token 生成，一次 serving 内部分成两个性质完全相反的阶段：

| 维度 | Prefill（预填充） | Decode（解码） |
|---|---|---|
| 输入 shape | prompt 一次进来，序列长度 s（几百~几万） | 每步 1 个 token，batch 维 B |
| 矩阵形态 | **GEMM**（M=s 或 s×B） | **GEMV / 窄 M 的 GEMM**（M=B，通常 1~256） |
| 瓶颈 | **计算密集（compute-bound）** | **访存密集（memory-bound）** |
| 每读 1 字节权重算多少 FLOP | ~s FLOP/B（s 越大越高） | ~1 FLOP/B（BF16） |
| KV cache 行为 | 生成（写）整段 prompt 的 K/V | 每步追加 1 个 token 位置，读全部历史 |
| 时延敏感度 | 决定 **TTFT** | 决定 **TPOT/ITL** |
| 对应你的昇腾经验 | 大 M 大 N 的 GEMM，tiling 拼 Cube 利用率 | M≤256 窄 M 场景（你的 L1 全载模板正是为此设计） |

**一句话本质**：prefill 把权重从 HBM 搬一次就能算 s 个 token 的活，摊销后算力是瓶颈；decode 每算 1 个 token 都要把**全部权重**从 HBM 搬一遍，带宽是瓶颈。

## 1.2 KV cache：生成与复用

Attention 的第 i 个 query 只能看到 ≤i 的 key/value。若每步都重算全部历史的 K/V，代价是 O(s²) 次 GEMM。KV cache 把每一层每个 token 的 K、V 向量存下来，decode 时只算新 token 的 q/k/v，历史部分直接读缓存。

- 第 i 步生成 token 时：`K[0..i], V[0..i]` 从 cache 读出，`k_i, v_i` 写入 cache 第 i 个槽位
- Attention 输出：`softmax(q_i · K^T / √d) · V`，读全部 i+1 个位置的 KV
- **代价**：显存。见 Day 2 公式，这是整个推理系统（PagedAttention、prefix caching、KV 量化、P/D 分离的传输）围绕旋转的核心对象

## 1.3 一次 decode 迭代的张量流（手画参考）

以 batch=1、第 i 步为例，逐层走一遍（照着画在纸上，标出每步的 shape 和数据量来源）：

```
token_id(i-1)                         # 标量
  │  Embedding 查表                    # 读 embedding 表的一行，写 B×H
  ▼
h ∈ R^{1×H}                           # H = hidden_size
  │
  ├─► 对每一层 ℓ = 0..L-1：
  │     │
  │     ├─ QKV 投影（GEMV）            # 读 3×(H×H_qkv) 权重 ← ★权重访存大头
  │     ├─ q_i 算出；k_i,v_i 追加写入该层 KV cache 槽位
  │     ├─ Attention（gather kernel）  # 读该层全部 KV[0..i] ← ★KV 访存大头
  │     │    scores = q_i·K^T/√d → softmax → ·V
  │     ├─ O 投影（GEMV）              # 读 H×H 权重
  │     └─ residual + RMSNorm + MLP（gate/up GEMV → SwiGLU → down GEMV）
  │                                        # 读 3 个 H×FFN 权重（FFN≈8H/3×3）
  ▼
final RMSNorm → lm_head（GEMV 到 |V| 词表） # 读 H×|V| 权重
  ▼
logits → sample → token_id(i)          # 下一步的输入
```

**画完图要能回答**：这一步总共从 HBM 读了多少字节？（答：≈全部权重 + 全部 KV cache + 零头激活——这就是 Day 2 的时延下界来源。）

## 1.4 计算量与访存量推导（产出物核心）

**记号**：N=参数量，L=层数，H=hidden，d=head_dim，H_q=Q 头数，H_kv=KV 头数（GQA），s=序列长度，B=batch，P=字节数/参数（BF16=2，FP8=1，INT4=0.5）。

**Prefill（每个请求）**
- FLOPs ≈ `2·N·s`（GEMM 部分）+ attention 部分 `4·L·H_q·d·s²/2 ≈ 2·L·H·s²`（s 大时不可忽略）
- 访存 ≈ 权重一遍 `N·P` + KV 写出 `s·KV_per_token`
- 判定：AI（算术强度）≈ `2s·N·P/(N·P)` 量级 ∝ s，s 超过几百 token 后稳稳落在 Roofline 屋顶右侧 → **compute bound**

**Decode（每步、batch=B）**
- FLOPs = `2·N·B`（每 token 约 2N）
- 访存 ≈ 权重一遍 `N·P`（与 B 无关，可摊销！）+ KV `B·ctx·KV_per_token`
- 判定：权重部分 AI ≈ `2B·N·P/(N·P)·(1/P)`……直接算：每字节权重对应 2 个 FLOP（乘+加），即 **BF16 下 AI=1 FLOP/B，FP8 下 AI=2 FLOP/B**，而 H100 的屋顶分界在 ~295（BF16）→ 差 **150~300 倍**，铁定的 memory bound

**与昇腾经验对接**：decode 的 GEMV 就是你做过的 M≤256 窄 M 场景——A 矩阵（激活）小、权重矩阵大且每步必读，你的"权重 L1 全载"模板解决的是同一个问题（把 O(n·A) 的重复搬运降为 O(A)）；GPU 侧没有 L1 全载这种大 SRAM 驻留能力（SMEM 只有 228KB/SM），所以走的是 **batch 摊销 + 权重量化 + kernel 融合** 的路线。

## 1.5 动手任务

1. 白纸手画 1.3 的张量流，标出每处访存的字节数来源（权重/KV/激活）
2. 用 1.4 的公式，代入 Qwen3-8B（L=36，H_kv=8，d=128）算：decode batch=1 时权重读多少字节？

## 1.6 自测题

1. 为什么 prefill 是 compute bound 而 decode 是 memory bound？（用 AI 讲，不要背结论）
2. batch 从 1 升到 32，decode 每步的 FLOPs 变多少？权重访存变多少？——由此推出什么？
3. GQA 中 H_kv < H_q，影响的是计算量还是访存量？影响哪一部分？
4. lm_head 的 GEMV 访存（H×128256×2B ≈ 8.6GB for 70B？验证一下这个数）占总权重多少？

**产出物**：笔记《prefill/decode 计算与访存量推导》（把 1.4 的推导用自己的话重写一遍 + 1.6 的答案）

---

# Day 2：显存与时延的手算公式（面试必考）

- [ ] KV cache 每 token 显存公式
- [ ] Decode 单 token 理论时延下界
- [ ] 练习：Llama-3-70B FP8 在 H100 上的 decode 时延下界

## 2.1 KV cache 每 token 显存

```
KV_per_token = 2 × L × H_kv × d × P
              ↑  ↑    ↑     ↑   ↑
            K和V  层数 KV头数 头维 每参数字节
```

- **2**：K 和 V 两份
- **H_kv 不是 H_q**：GQA 模型 KV 头数远小于 Q 头数（如 70B：64 Q 头 vs 8 KV 头），这是 KV cache 显存可控的根本原因
- 常见值速查（BF16）：

| 模型 | L | H_kv | d | KV/token (BF16) | KV/token (FP8) |
|---|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 128 KiB | 64 KiB |
| Llama-3-70B | 80 | 8 | 128 | 320 KiB | 160 KiB |
| Qwen3-8B | 36 | 8 | 128 | 144 KiB | 72 KiB |
| Qwen2.5-72B | 80 | 8 | 128 | 320 KiB | 160 KiB |

（计算示例：Llama-3-70B BF16 = 2×80×8×128×2 B = 327,680 B = 320 KiB）

**单序列 KV 总量** = `ctx_len × KV_per_token`。例：70B FP8 在 4K 上下文 = 4096×160 KiB = **640 MiB/序列**。

## 2.2 显存全景：一张卡上有什么

```
总显存 = 权重 (N×P)  +  KV cache 池  +  激活/Workspace  +  CUDA Graph/框架开销(~2-4 GB)
          ↑放不下模型一切免谈    ↑决定并发上限            ↑decode 时很小, prefill 长 prompt 时可观
```

**并发上限估算**（面试高频）：

```
可用 KV 池 ≈ 显存 − N×P − 4 GB
最大并发 ≈ 可用 KV 池 / (平均 ctx × KV_per_token)
```

## 2.3 Decode 单 token 理论时延下界

decode 每步**至少**要把所有权重和该序列的全部 KV 从 HBM 读一遍（激活读写在 M=1 时是零头）：

```
TPOT_min ≈ ( N×P  +  B × ctx × KV_per_token ) / HBM带宽
           ↑权重一遍(与B无关)  ↑每条序列各自的KV
```

- batch=1 时第二项远小于第一项 → **TPOT 下界 ≈ 模型参数字节数 / HBM 带宽**
- 这就是"decode 是带宽换 token"的定量表述；实际值因未完美流水、kernel 开销、采样等，约为下界的 1.2~1.8 倍

## 2.4 例题一（计划原题）：Llama-3-70B FP8 在 H100（3.35 TB/s）上的 decode 时延下界

**已知**：N=70.6 B 参数，FP8（1 B/参数）；L=80，H_kv=8，d=128；H100 SXM 带宽 3.35 TB/s，显存 80 GB。

**推导**：

1. 权重字节 = 70.6e9 × 1 B = **70.6 GB**
2. 权重读取时间 = 70.6 GB ÷ 3.35 TB/s ≈ **21.1 ms**
3. KV 读取（设 ctx=4096）= 4096 × 160 KiB = 640 MiB → 640e6/3.35e12 ≈ **0.2 ms**
4. TPOT 下界 ≈ 21.1 + 0.2 ≈ **21.3 ms/token**，即 batch=1 时 **≤ 47 tok/s**

**附加结论（面试加分）**：
- 70.6 GB 权重 > 80 GB 卡的可用显存（剩 ~9 GB 根本放不下多少 KV）→ **单卡 H100 跑不动 70B FP8**，至少 2×H100 TP=2
- 反证计算侧只占多少：每 token FLOPs=2N=141.2 GFLOP，H100 FP8 算力 1979 TFLOPS → 若 compute bound 只需 71 µs，与 21 ms 差 **297 倍**（正好 = ridge point 591 ÷ AI 2，与 Day 3 Roofline 互验）

## 2.5 例题二：2×H100 TP=2 跑 Llama-3-70B FP8，并发上限？

1. 每卡权重 = 70.6/2 = 35.3 GB → 每卡可用 ≈ 80 − 35.3 − 4 = 40.7 GB → 全系统 KV 池 ≈ **81.4 GB**
2. 每序列（4K ctx，FP8 KV）= 0.625 GB
3. 并发 ≈ 81.4 / 0.625 ≈ **130 序列**（32K ctx 则只剩 ~16）

## 2.6 例题三：batch 摊销（TPOT vs 吞吐的剪刀差）

Qwen3-8B BF16（权重 16.4 GB），A100 80GB（2.04 TB/s），ctx=8K（KV/seq=1.125 GiB）：

| batch | 访存量 | TPOT | 吞吐 (tok/s) |
|---|---|---|---|
| 1 | 16.4+1.1=17.5 GB | 8.6 ms | 116 |
| 8 | 16.4+9=25.4 GB | 12.5 ms | 642 |
| 32 | 16.4+36=52.4 GB | 25.7 ms | **1244** |

**结论**：TPOT 只涨 3 倍，吞吐涨 10.7 倍——权重读取被 batch 摊销，这就是 continuous batching 提高吞吐的第一性原理；同时 TPOT 上涨就是用户体感的代价，SLO 决定 batch 上限（→ Day 5 goodput）。

**进阶思考**（可与 Day 3 互验）：B 很大时每序列 KV 读取 `ctx·KV/BW` 与每序列计算时间 `2N/峰值算力` 谁大？H100 FP8 + ctx=4096 时：200 µs vs 71 µs → 永远访存 bound；ctx≈1500 时两者打平；**只有短上下文 + 超大 batch 才可能翻转为 compute bound**。

## 2.7 练习题（做完合上答案）

1. **Qwen3-8B BF16 在 A100 80GB，32K 上下文，最大并发？**（答：~13）
2. **如果 Llama-3-70B 没有 GQA（64 个 KV 头），2×H100 能并发几个 4K 请求？**（答：KV/token×8=2.5 MiB → 每序列 10 GB → 仅 ~8 个；GQA 是大并发 serving 的前提）
3. **4090（24GB，1.0 TB/s）跑 Qwen3-8B BF16：TPOT 下界？KV 池剩多少？**（答：16.4/1.0≈16.4 ms→~61 tok/s；KV 池≈3.6 GB→32K ctx 仅 1 条）

## 2.8 面试话术模板

> "任何一个 serving 容量问题我分三步：①模型字节数=参数量×精度，看几张卡放得下；②KV 池=显存−权重−预留，除以 单序列 KV=ctx×2L·H_kv·d·P 得并发；③TPOT 下界=(权重+ΣKV)/带宽，实际乘 1.3~1.8 的系数估真实值。"

**产出物**：3 道手算题完整推导过程（含单位换算过程，面试白板要写的就是这个过程）

---

# Day 3：Roofline 模型（GPU 版）

- [ ] 对照昇腾的访存/计算 bound 分界模型，理解 GPU 的 arithmetic intensity 判定
- [ ] 用 ncu 跑一个小 kernel 看 SM busy / DRAM busy

## 3.1 Roofline 基本形式

```
可达算力 = min( 峰值算力 , AI × 带宽 )
                  ↑屋顶         ↑斜坡

AI (arithmetic intensity) = FLOPs / 访存字节数
ridge point（屋檐拐点）= 峰值算力 / 峰值带宽   [单位: FLOP/Byte]
```

- AI < ridge → memory bound，性能由斜坡决定（`AI×BW`）
- AI > ridge → compute bound，性能贴屋顶
- **ridge point 是平台的身份证**，跨平台比较先比它

## 3.2 平台参数速查

| 平台 | BF16 算力 (dense) | FP8 算力 | HBM 带宽 | 显存 | ridge (BF16) | ridge (FP8) |
|---|---|---|---|---|---|---|
| A100 80GB SXM | 312 TFLOPS | — | 2.04 TB/s | 80 GB | 153 FLOP/B | — |
| H100 80GB SXM | 989 TFLOPS | 1979 TFLOPS | 3.35 TB/s | 80 GB | **295** | **591** |
| H200 SXM | 989 TFLOPS | 1979 TFLOPS | 4.8 TB/s | 141 GB | 206 | 412 |
| RTX 4090 | 165 TFLOPS | 330 TFLOPS* | 1.01 TB/s | 24 GB | 163 | 327 |
| B200 | ~2.25 PFLOPS | ~4.5 PFLOPS | ~8 TB/s | 192 GB | ~280 | ~560 |

*4090 FP8/FP16-accumulate 口径较绕，面试一般不问消费卡。

**关键观察**：带宽增速（2→3.35→4.8→8 TB/s）与算力增速基本同步，所以 ridge 十年稳定在 150~600 FLOP/B——这就是为什么 decode（AI≈1~2）**在任何 GPU 上都是 memory bound**，平台无关。

## 3.3 三类核心 kernel 的 AI 定位

| Kernel | FLOPs | 主要访存 | AI | 落点 |
|---|---|---|---|---|
| Decode GEMV（B=1, BF16） | 2NK | 2NK B 权重 | **1 FLOP/B** | 斜坡深处（差 ridge ~300×） |
| Decode GEMM（B=64, FP8） | 2NK×64 | NK B 权重 | **128 FLOP/B** | 仍偏斜坡，但接近分界 |
| Prefill GEMM（4096³, BF16） | 2MNK | 3×2MN B 矩阵 | ~1365 FLOP/B | 屋顶（compute bound） |
| Prefill attention（s=4096） | ~4s²d/头 | Q/K/V 3×sd | ~s/3 ≈ 1000+ | 屋顶（s 大时） |
| Decode attention（ctx=4096, B=1） | 4·ctx·d/头 | 2·ctx·d B KV(BF16) | **1 FLOP/B** | 斜坡（KV gather） |
| Decode attention（B=64） | ×64 | KV 不变 | ~64 FLOP/B | 斜坡→分界 |

这张表是 Day 1 结论的定量版：**decode 全链路压在斜坡上，唯一杠杆是减少字节（量化）和摊销字节（batch）**。

## 3.4 从昇腾 bound 建模到 GPU Roofline 的映射表（本周核心产出）

| 维度 | 昇腾（达芬奇架构） | NVIDIA GPU | 同一性 |
|---|---|---|---|
| 计算单元 | Cube / Vector / Scalar | Tensor Core / CUDA Core | 峰值 FLOPS 口径 |
| 存储层级 | HBM→L2→L1→L0A/L0B/L0C (+UB) | HBM→L2→SMEM/L1→Register | 都是"离算力越近越小越快" |
| 搬运引擎 | MTE2(入)/MTE1(L1→L0)/MTE3(出) | cp.async / TMA(Hopper+) | 双缓冲/乒乓 ↔ double buffering |
| 分块参数 | baseM/baseN/baseK tiling | threadblock/warp tile + MMA m,n,k | 同一个"分块凑局部性"问题 |
| bound 判定 | `CalRebalanceBlock`：L2/HBM 带宽 vs Cube 算力分界 + balanceRate≥0.9 剪枝 | Roofline：AI vs ridge point | **同一第一性原理，不同参数表** |
| L2 优化 | ASW 蛇形滑窗（S 形扫描提升 L2 命中） | tile 排布 swizzle / persistent kernel | 提升片上复用 |
| 权重驻留 | AL1/BL1 全载模板（权重驻留 L1） | 无对应（SMEM 太小）→ 用 batch 摊销 | 平台差异点，面试讲"跨平台方法论"的好素材 |
| 流水线 | Fixpipe / 无 Queue 手工流水（SetFlag/WaitFlag） | 软件流水 / async copy 多 stage | 隐藏搬运延迟 |
| 利用率指标 | Cube 利用率 / aicore cycles | SM throughput / tensor pipe active | ncu `sm__throughput` |
| Profile 工具 | msprof / Ascend Profiler | nsys（时间线）/ ncu（单 kernel 深挖） | 对应关系见 3.5 |

**叙事主线（面试用）**："我在昇腾上做的是'用 L2/HBM 带宽和 Cube 算力的比值划 bound 线、用 tiling 搜优把 kernel 推到线上方'——搬到 GPU 就是 Roofline + tile/warp 调优，公式不同、方法论同源。区别在昇腾 L1 大可以做权重驻留，GPU SMEM 小必须走 batch 摊销，这决定了两边算子形态的不同。"

## 3.5 ncu 上手实操

```bash
# 安装：CUDA toolkit 自带或 pip install ncu-nsight-cu-cli
# 1) 单 kernel 速览：SpeedOfLight 一节直接给两个百分比
ncu --section SpeedOfLight python matmul_bench.py

# 2) 只看 SM/DRAM 利用率两个指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    python matmul_bench.py

# 3) 自动画 roofline（GUI 里看散点）
ncu --set roofline -o report python matmul_bench.py && ncu-ui report.ncu-rep
```

**判读表（背下来）**：

| SM busy | DRAM busy | 结论 |
|---|---|---|
| <40% | <40% | 未饱和：latency bound（occupancy/依赖/启动开销） |
| ≥85% | 任意 | compute bound（看 tensor pipe 更准） |
| 任意 | ≥85% | memory bound（查访存量能不能砍） |
| ~60% | ~60% | 两者都半吊子：通常是分块不当，tiling/流水有问题 |

**动手任务**：写一个 10 行的 PyTorch 脚本跑 `torch.matmul`（大矩阵）和一个逐元素 `y = x + 1`（大张量），各用 ncu 抓 SM/DRAM busy，验证一个贴屋顶、一个贴斜坡。

**常见误判**：
- L2 命中会把"有效带宽"抬高，DRAM busy 低但 kernel 仍受限于 L2 带宽 → 加看 `lts__t_sectors` / L2 hit rate
- decode 场景 kernel 极短（µs 级），launch 开销占比高 → 先看时间线（nsys）再下 ncu
- 小 batch 时"看起来 memory bound"其实是 latency bound，两者药方完全不同

**产出物**：笔记《从昇腾 bound 建模到 GPU Roofline 的映射表》（以 3.4 表为骨架，每行补一句自己的话）

---

# Day 4：PagedAttention 论文精读（SOSP 2023）

- [ ] 核心问题：KV cache 的显存碎片与浪费（原方案浪费 60-80%）
- [ ] block/table/引用计数/COW 的设计动机

## 4.1 论文信息

- *Efficient Memory Management for Large Language Model Serving with PagedAttention*, Kwon et al., **SOSP 2023**（UCSB + UC Berkeley）
- vLLM 的开山之作；读原文 + 对照 V1 源码里的 block pool（W3 再深入）

## 4.2 问题：KV cache 浪费 60-80% 从哪来

朴素系统（HF / TGI / 早期 Triton Inference Server）按**最大长度**为每个请求预留连续显存：

| 浪费类型 | 机制 | 量级 |
|---|---|---|
| **内部碎片** | 预留 max_len（如 2048）实际只生成 200 token | 与 (max−actual)/max 同比 |
| **外部碎片** | 变长请求进进出出，显存空洞无法紧凑（GPU 张量要求连续 + 静态 shape） | 随时间累积 |
| **预留/抢占** | 内存满时整请求换出或重算，吞吐塌方 | 尾延迟来源 |

实际统计：**有效 KV 只占预留的 20-40%**。由于并发上限 ∝ 有效 KV 池（Day 2 公式），浪费 60-80% 显存 ≈ 吞吐直接损失 2.5-5×——这就是论文的动机：**管理好显存 = 免费的数倍吞吐**。

## 4.3 核心设计：把操作系统虚拟内存搬进 GPU

| OS 虚拟内存 | PagedAttention |
|---|---|
| 页 page | **block**（默认 16 token 的 KV，按层组织） |
| 页表 page table | **block table**（逻辑块号 → 物理块号） |
| 物理帧 | KV 池中的物理 block |
| 缺页 | 分配新 block（同步、无 lazy） |
| 引用计数 | block 的 ref count（多序列共享） |
| fork + COW | parallel sampling 共享 prompt 前缀，分叉时复制 |
| swap | 抢占时换出到 CPU（vLLM 默认反而选 recompute，见 4.5-3） |
| 碎片 | **<4%**（只剩 block 内部尾部空洞） |

**关键工程难点**：attention kernel 必须会"查表寻址"——Q 逻辑连续的 KV 实际散在物理块里，需要改 kernel 按 block table 做 paged gather（vLLV V1 中由 FlashAttention/FlashInfer 的 paged KV 接口承担，`vllm/attention/`）。

**调度收益**：块粒度分配使"几乎是零浪费"→ 同样显存塞进 2-4× 的 batch → 论文实测吞吐 **2-4×（vs TGI/Orca），对比 HF 最高 20×+**（并行采样/共享前缀场景）。

## 4.4 我标注的三个最巧设计点（模板，自己重选更好）

1. **block table 间接层**：一个"逻辑/物理解耦"就把碎片问题从分配器里消灭——而且全在用户态 GPU 侧实现，不依赖 OS/驱动。这是把 VM 思想迁移到"没有 MMU 的世界"的示范。
2. **COW + 引用计数实现 fork 语义**：parallel sampling（一次 prompt 出 n 个候选）的公共前缀零拷贝，只在分叉瞬间复制最后一个块。与进程 fork 的 COW 完全同构。
3. **抢占选 recompute 而非 swap**：与 CPU 世界直觉相反——HBM→CPU 往返带宽昂贵，而重算一段 prompt 花的是（本来空闲的）算力。这个 trade-off 在不同硬件/负载下会翻转（P/D 分离后 prefill 算力不再空闲，见 W5）。

## 4.5 局限与 V1 演进（必须知道的后续）

| 论文时代 (v0) | V1 现状 |
|---|---|
| 自研 PagedAttention CUDA kernel | FlashAttention / FlashInfer 后端原生支持 paged KV（`vllm/attention/` 插拔） |
| prefix caching 可选 | **默认开启**，block hash = f(父块 hash, token ids, salt)，链式哈希 |
| 抢占默认 recompute | 同样保留 recompute 优先 |
| block size 16 | 默认仍 16（trade-off：小 → 表项开销大；大 → 块内碎片） |

**遗留局限**：块粒度仍非 token 级（尾部浪费）、查表开销、多机场景块所有权管理（→ W5 分层 KV 的引子）。

## 4.6 精读笔记模板

```
1. 一句话问题定义（含 60-80% 这个数字怎么测的）
2. 三类浪费的机理图（自己画）
3. 数据结构：block / table / refcnt / slot 的字段级草图
4. kernel 侧：paged gather 的寻址流程
5. 实验部分重点看：Fig.6(吞吐-延迟)、并行采样加速比、共享前缀实验
6. 三个最巧设计点 + 为什么"巧"（约束下的对称性破缺）
7. 如果是我：还有什么别的解法？（对比 Sliding-window/SSTable 式 compaction…）
8. 面试 3 分钟版讲稿
```

**产出物**：论文精读笔记，标注 3 个你觉得最巧的设计点

---

# Day 5：Serving 指标体系

- [ ] TTFT、TPOT/ITL、E2E、throughput、goodput 的定义与关系
- [ ] SLO 驱动思维：为什么按 goodput 而非 raw throughput 评估

## 5.1 指标定义卡片（面试抽背版）

| 指标 | 定义 | 公式 | 由什么决定（第一性） |
|---|---|---|---|
| **TTFT** | 请求到达到首 token | 排队 + prefill | prefill 吞吐（compute bound）+ 队列长度 |
| **TPOT** | 平均每输出 token 耗时 | `(E2E − TTFT)/(n−1)` | decode 步时延（memory bound）× batch 拥挤度 |
| **ITL** | 相邻 token 间隔（逐 token） | 分布而非均值 | TPOT 的细粒度版，**抖动**看得见（抢占/chunk 边界尖刺） |
| **E2E** | 端到端时延 | `TTFT + (n−1)×TPOT` | 上两者之和 |
| **Throughput** | 系统输出速率 | output tokens/s | 随并发升，但受 TPOT 恶化反噬 |
| **Goodput** | **满足 SLO 的**吞吐 | 合格请求的 tokens/s | 曲线膝点（见 5.3） |

## 5.2 两个恒等式（白板必写）

1. `E2E = TTFT + (n−1) × TPOT`——分解归因的第一刀
2. 吞吐 ≈ `并发数 / TPOT`（稳态 Little's Law：`L = λ × W`）——容量估算的桥梁

## 5.3 SLO 与 goodput：为什么生产系统按 goodput 评估

- **SLO 举例**（对话场景典型值）：TTFT p99 ≤ 500 ms（超过用户觉得"卡住了"）；TPOT p99 ≤ 50~100 ms（阅读速度 10-20 tok/s 的体感下限）
- **raw throughput 的欺骗性**：过载后继续加并发，batch 更大、吞吐（tokens/s）还能涨，但 TTFT/TPOT 尾延迟爆炸——**用户侧体验已经崩了**
- **goodput 曲线**：横轴 load，纵轴"满足 SLO 的吞吐"——先升后降，存在**膝点（knee）**。生产容量规划 = 部署在膝点左侧略留余量，而不是吞吐峰值点
- 面试一句话："**raw throughput 优化的是机器，goodput 优化的是生意**。膝点右边每一分吞吐都是拿 SLO 换的。"

## 5.4 诊断速查表（W8 Day 51 诊断树的雏形，现在先建立直觉）

| 症状 | 优先怀疑 | 验证手段 |
|---|---|---|
| TTFT ↑、ITL 稳 | 排队堆积 / prefill 拥塞 | waiting 队列深度、prefill token budget |
| ITL 稳中有规律尖刺 | chunked prefill 混排 / 抢占重算 | preemption 计数、step 时间分布 |
| ITL 整体抬升 | batch 过大 / KV 读饱和 / 通信占比 | batch size、KV 池占用、TP 通信时间 |
| 吞吐不升反降 | 频繁抢占、prefix 失效 | preemption 计数、cache hit rate |
| OOM / 崩溃 | KV 超配 | watermark、max_num_seqs、max_num_batched_tokens |

## 5.5 动手任务

1. 把 5.1 表抄成口袋卡片（正面指标名，背面公式+决定因素）
2. 用 Day 2 的手算能力串联指标：给定 H100 + 8B BF16 + 期望并发 64 @ 8K ctx，估算 TPOT 和吞吐，再估 E2E（n=512）——这就是一道完整面试题
3. 画一张草图：x=并发，双 y 轴画 throughput（先升后平/降）与 goodput（先升后陡降），标出膝点

**产出物**：指标定义卡片

---

# Day 6：环境搭建 + 第一次压测

- [ ] 部署 vLLM（最新版），启动 Qwen3-8B 服务
- [ ] 跑 `vllm bench serve`（ShareGPT），观察 TTFT/TPOT 随并发的变化

## 6.1 准备清单

- 硬件：A100/H100 80GB 最佳；4090 24GB 也可以（Qwen3-8B 放得下，`--max-model-len` 调小到 8192）
- 软件：Python 3.10-3.12、CUDA 12.x；`nvidia-smi`、`nsys`、`ncu` 可用
- 数据：ShareGPT 对话集（vLLM 标准压测集）

## 6.2 部署 vLLM 并启动服务

```bash
python -m venv vllm-env && source vllm-env/bin/activate
pip install vllm                     # 自动带 torch 等依赖

# 网络受限时走 modelscope：
# pip install modelscope && modelscope download --model Qwen/Qwen3-8B

vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --enable-metrics                   # 旧版本默认关闭；新版本默认开
# 验证：curl http://localhost:8000/v1/models
```

## 6.3 压测

```bash
# ShareGPT 数据集（vLLM 官方 benchmarks 用的同一份）
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_clean_cod_split2.json

# 新版 CLI 入口（等价于旧版 python benchmarks/benchmark_serving.py）
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-8B \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_clean_cod_split2.json \
  --num-prompts 300 \
  --request-rate 2.0 \
  --percentile-metrics ttft,tpot,itl \
  --save-result result_r2.json
```

**实验矩阵**：`--request-rate ∈ {0.5, 1, 2, 4, 8, inf}` 各跑一轮（inf = 饱和打满），结果存 JSON。也可另跑 `vllm bench latency`（离线批量）对照在线结果。

## 6.4 观察什么：预期曲线形态（先预测，再对照）

| 区间 | 预期现象 | 第一性解释 |
|---|---|---|
| 低载（rate 0.5-1） | TTFT≈单请求 prefill 时间，ITL≈TPOT 下界 | 无排队，Day 2 公式直接命中 |
| 中载（rate 2-4） | TTFT 缓升，TPOT 缓升 | batch 增大摊销权重（Day 2 例题三） |
| 高载（rate 8-inf） | TTFT **超线性**暴涨，TPOT 明显抬升，可能见 preemption | 队列论 + KV 池逼近上限（Day 5 表） |

**辅助观察**：`curl localhost:8000/metrics | grep -E "preemption|queue|token_usage"`——把 preemption 计数与 ITL 尖刺对上号（W2 Day 13 会正式做）。

**画图**：x=request rate，y1=TTFT p99，y2=TPOT p99（两张小图并列也行）→ 这就是本周最终产出。

## 6.5 踩坑清单

- 4090 上跑 8B：`--max-model-len` 必须 ≤8192，否则 KV 池为 0 直接起不来
- `--request-rate inf` 时 num-prompts 别太大，首轮跑 200 即可
- ShareGPT 文件名/路径写错会静默 fallback 到随机数据集——看日志确认 `dataset_name=sharegpt`
- 首次启动要编译/加载权重，bench 前 curl 一次预热；实验中途别混开其他 GPU 进程
- 结果 JSON 里 `percentiles` 字段才是 p99，mean 会被长尾带偏

**产出物**：第一张性能曲线图（并发 vs TTFT p99 / TPOT p99）+ 三行结论

---

# Day 7（复盘日）

- [ ] 整理本周笔记为《LLM 推理性能的第一性原理》一篇
- [ ] 自测：不看笔记，手推 70B 模型的显存与并发上限估算

## 7.1 一页总结（初稿，自己重写才算数）

**《LLM 推理性能的第一性原理》**

1. **两阶段**：prefill compute bound（AI ∝ s），decode memory bound（AI≈1-2 FLOP/B vs ridge 150-600）
2. **两公式**：`KV/token = 2L·H_kv·d·P`；`TPOT_min ≈ (N·P + B·ctx·KV_per_token)/BW`
3. **一模型**：Roofline。ridge=算力/带宽，平台十年稳定 150-600 FLOP/B → decode 的 memory bound 是**平台无关**的宿命
4. **推论①（容量）**：并发上限 ≈ (显存−权重−预留)/(ctx·KV_per_token)——显存管理就是吞吐（PagedAttention 的 2-4× 从这来）
5. **推论②（摊销）**：batch 增大，TPOT 亚线性涨、吞吐近线性涨——continuous batching 的理论根基；batch 上限由 SLO 定
6. **推论③（量化）**：量化同时打权重字节（TPOT↓）和 KV 字节（容量↑、并发↑），是 decode 侧第一优先级杠杆
7. **推论④（prefill）**：TTFT ≈ 2Ns/(MFU·峰值算力)，杠杆是 MFU（kernel/调度/通信），与 decode 的杠杆完全不同 → P/D 分离的种子（W5）
8. **指标层**：E2E=TTFT+(n−1)·TPOT；生产看 goodput 膝点而非 raw throughput

## 7.2 自测题（合上全部材料）

1. 70B BF16，2×H100 TP=2，4K ctx：并发上限？（权重 70.6 GB/卡→ 可用≈25 GB/卡→50 GB 池→**~40 序列**）
2. 手推：为什么 decode 的 AI 在 BF16 下恰是 1 FLOP/B？
3. H100 BF16 的 ridge point？不用背，现场算。（989/3.35≈**295**）
4. batch=128、ctx=1024 时 decode 还一定 memory bound 吗？（算 KV 项 vs 算力项，短 ctx 大 batch 会贴近分界）
5. PagedAttention 把浪费从 60-80% 降到 <4%，为什么吞吐"只"提升 2-4×而不是 5×？（浪费≠全部瓶颈；decode 后期受 KV 读带宽与 SLO 约束）
6. 用户反馈"回答变慢但打字速度正常"，先查什么？（TTFT/排队，不是 TPOT）
7. 为什么 vLLM 抢占选 recompute 而非 swap？（HBM↔CPU 往返 vs 空闲算力重算的 trade-off）
8. 一句话讲清 GQA 对 serving 的意义。（KV/token 缩 H_q/H_kv 倍→容量与 KV 带宽瓶颈同步缓解）
9. 良率题：TTFT p99 500 ms、TPOT p99 60 ms，n=400，E2E p99 大约？（0.5+399×0.06≈**24.4 s**）
10. 把 Day 2 例题一完整重推一遍（含单位）。

## 7.3 归档清单

```
week1/
├── README.md            ← 本手册
├── notes/
│   ├── day1_prefill_vs_decode.md     （含张量流手画照片/扫描）
│   ├── day2_hand_calc.md             （3 道题完整推导）
│   ├── day3_roofline_ascend_map.md   （映射表）
│   ├── day4_pagedattention_notes.md  （论文精读）
│   └── day5_metrics_cards.md
├── results/
│   ├── bench_r*.json
│   └── curve_ttft_tpot.png
└── summary_first_principles.md       （Day 7 一页总结）
```

---

# 附录 A：常用模型参数速查（手算题代入用）

| 模型 | 参数量 | L | hidden | H_q | H_kv | d | KV/token BF16 |
|---|---|---|---|---|---|---|---|
| Llama-3-8B | 8.0 B | 32 | 4096 | 32 | 8 | 128 | 128 KiB |
| Llama-3-70B | 70.6 B | 80 | 8192 | 64 | 8 | 128 | 320 KiB |
| Qwen3-8B | 8.2 B | 36 | 4096 | 32 | 8 | 128 | 144 KiB |
| Qwen2.5-72B | 72.7 B | 80 | 8192 | 64 | 8 | 128 | 320 KiB |
| DeepSeek-V3 (MoE) | 671 B (激活 37 B) | 61 | 7168 | 128 | MLA | 576/128 | MLA：~70 KiB/token* |

*MLA 把 KV 压到 latent 向量（≈576 维 + 少量 rope 项），是"架构级 KV 压缩"路线，W3 讲 MLA 后端时再展开。

# 附录 B：单位与易错点

- **GiB vs GB**：手算时统一十进制（1 GB=1e9 B，1 TB/s=1e12 B/s），显存标称是 GiB（80"GB"=80 GiB≈85.9 GB）——面试报数差 7% 以内可接受，但要说明口径
- **FLOPs vs FLOPS**：FLOPs 是总量（-floating point operations），FLOPS 是速率（/s）；1 个乘加 = 2 FLOPs
- **参数量≈显存**：`N×P`；注意 lm_head 与 embedding 在大词表模型占比可观（70B：128256×8192×2×2≈4.2 GB，~6%）
- **MFU** = 实测 FLOPs / (峰值 FLOPS × 时间)；prefill 合格线 40-60%，decode 谈 MFU 无意义（看带宽利用率）
- **上下文长度**：估算一律用"当前已生成长度"而非 max_model_len；平均 ctx 用负载的期望
