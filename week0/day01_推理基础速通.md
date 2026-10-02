# Day 1 · 推理基础速通（建立共同语言）

> **总时长**：6-8 小时（上午 3.5h + 下午 2.5h + 晚上 1.5h）
> **今日目标**：面试时能把"显存/时延/并发"三类问题当场手推出来，并用自己的话讲清 prefill/decode 的本质
> **产出物**：一页纸《推理性能第一性原理》（含 3 道手算题完整推导）——这是 Day 7 面试作战包的第一块

---

## 作息建议

| 时间 | 内容 | 时长 |
|---|---|---|
| 09:00-10:30 | 模块一：Prefill vs Decode 的本质 | 1.5h |
| 10:45-12:30 | 模块二：KV Cache 机制 | 1.75h |
| 14:00-16:30 | 模块三：必考手算（公式 + 3 道例题 + 2 道自测） | 2.5h |
| 19:30-21:00 | 模块四：PagedAttention 论文导读 | 1.5h |
| 21:00-21:30 | 整理一页纸产出 + 自测清单 | 0.5h |

---

## 模块一：Prefill vs Decode 的本质（上午，1.5h）

### 1.1 两个阶段的划分

LLM 推理一个请求分两个阶段：

- **Prefill（预填充）**：一次性并行处理输入 prompt 的全部 N 个 token，产出第 1 个输出 token，同时**生成全部 prompt 位置的 KV cache**
- **Decode（解码）**：自回归逐 token 生成，每步只处理 1 个新 token（每序列），但要**读全部历史 KV cache**

### 1.2 为什么说 prefill 是 compute-bound、decode 是 memory-bound

这是今天最重要的推导，务必自己推一遍（以单序列、Transformer 一层为例）：

**计算量（FLOPs）**：对于维度 d 的隐藏层，一个 token 过一层的计算量 ≈ 常数 × d²（QKV 投影 + FFN）。prefill 处理 N 个 token 就是 N 倍；decode 每步只有 1 个 token。

**访存量（Bytes）**：无论 prefill 还是 decode，**每步都要把模型权重完整读一遍**（W 字节）。

关键区别在于**算术强度（Arithmetic Intensity = FLOPs / Bytes）**：

| 阶段 | 每步计算量 | 每步权重访存 | 算术强度 | 瓶颈 |
|---|---|---|---|---|
| Prefill（N 个 token） | ∝ N·d² | W（读一次） | 高（∝N） | **算力**（compute-bound） |
| Decode（1 个 token） | ∝ d² | W（读一次） | 极低 | **HBM 带宽**（memory-bound） |

- Prefill 时权重读一次被 N 个 token 摊销，GPU 的 Tensor Core 能喂饱 → 撞算力墙
- Decode 时每生成 1 个 token 都要把几十 GB 权重完整读一遍，计算量却很小 → 撞带宽墙，算力大量闲置

> ⚡ **昇腾翻译提示**：这就是你做 WeightQuantBatchMatmul 时"访存/计算 bound 分界模型"的同一个原理——decode（M≤256 小 M 场景）就是典型的 memory-bound GEMM，所以你当时才需要 ASW 模板提升 L2 命中率、做 AL1/BL1 全载。面试时可以直接这样挂钩。

### 1.3 三条立即有用的推论

1. **Decode 吞吐上不去的根因是带宽不是算力** → 优化方向：量化（权重字节数减半，decode 时延近似减半）、投机解码（一次前向验证多个 token，摊销权重读取）、batching（多个序列共享一次权重读取）
2. **Prefill 和 decode 的最优硬件配置相反** → 这是 Day 4「P/D 分离」的全部动机
3. **Decode 阶段 GPU 算力大量闲置** → 所以才有"用计算换访存"的投机解码（Day 4）

### ✅ 模块一检验

合上书能回答：
- 为什么 batch size 增大时 decode 吞吐近似线性提升但单序列时延几乎不变？（答：权重读取被 batch 内序列摊销，直到撞上算力墙）
- 长 prompt 场景 TTFT 由什么决定？（prefill 算力 + 排队）

---

## 模块二：KV Cache 机制（上午，1.75h）

### 2.1 KV cache 是什么、为什么必须缓存

Decode 第 t 步生成时，attention 需要当前 token 的 Q 去 attend **所有历史位置** 的 K 和 V。如果不缓存，每步都要对全部历史 token 重算 K/V 投影 → 计算量从 O(t) 暴涨到 O(t²)。KV cache 用显存换计算：把每个历史 token 在每层的 K、V 向量存下来。

**代价**：KV cache 显存随「序列长度 × 并发数」线性增长——它是 serving 系统显存的第一大头，也是 vLLM 全部调度和内存管理存在的原因。

### 2.2 每 token KV cache 显存（必背公式）

```
KV_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

- 因子 2：K 和 V 各一份
- **GQA 陷阱**：用 `num_kv_heads` 而不是 attention 头数！MHA 模型两者相等，GQA 模型 kv_heads 远小于 q_heads（这正是 GQA 省显存的原理）

### 2.3 注意力变体对 KV cache 的影响（面试常追问）

| 变体 | KV 头数 | KV cache 相对大小 | 代表模型 |
|---|---|---|---|
| MHA | = q_heads | 1×（基准） | GPT-3、Llama-2-7B |
| GQA | q_heads / 4 ~ / 8 | 1/4 ~ 1/8 | Llama-3、Qwen3 |
| MQA | 1 | 最小 | Falcon |
| MLA | 压缩为低秩潜向量 | 极小（约 1/10+） | DeepSeek-V2/V3 |

> 一句话总结：**现代模型架构演进的一条主线就是压缩 KV cache**——因为长上下文时代 KV cache 已经超过权重成为显存瓶颈。

### 2.4 KV cache 显存增长曲线（要有直觉）

以 Llama-3-70B（80 层，GQA 8 个 KV 头，head_dim 128，FP16）为例：

- 每 token：2 × 80 × 8 × 128 × 2B = **327,680 B ≈ 320 KiB**
- 一条 8K 上下文的序列：320 KiB × 8192 ≈ **2.5 GiB**
- 一条 128K 上下文的序列：**40 GiB**（相当于小半个模型的权重！）

### ✅ 模块二检验

- 为什么 vLLM 要用 block（页）管理 KV 而不是连续分配？（先凭直觉答：不同请求长度各异、动态增长、需要共享前缀——晚上论文导读验证）
- 128K 上下文时代，限制并发的首要因素是什么？（KV cache 显存，不是权重）

---

## 模块三：必考手算（下午，2.5h）

### 公式卡（背到默写）

```
① KV cache 每 token 显存 = 2 × layers × kv_heads × head_dim × dtype_bytes

② Decode 单 token 时延下界 ≈ 权重字节数 / HBM 带宽
   （memory-bound 下，计算时间可忽略；batch=1 时成立）

③ 可用 KV 显存 = (卡显存 × gpu_memory_utilization − 权重 − 激活/工作区开销) × 卡数
   最大并发序列数 = 可用 KV 显存 / (每 token KV × 平均上下文长度)
```

**常用硬件参数（记这几个数）**：

| 硬件 | 显存 | HBM 带宽 | FP16 算力 |
|---|---|---|---|
| H100 SXM | 80 GB | 3.35 TB/s | ~990 TFLOPS（稠密） |
| A100 80G | 80 GB | 2.0 TB/s | ~312 TFLOPS |
| H20 | 96 GB | 4.0 TB/s | ~148 TFLOPS |

> 顺带记住这个对比的戏剧性：H20 带宽比 H100 还高但算力只有 1/7 → H20 跑 decode 尚可、跑 prefill 很弱 → 又一条 P/D 分离的论据。

### 例题 1：KV cache 每 token 显存（基础题，必须秒答）

**题目**：Qwen3-32B，60 层（注：按 64 层估算即可，面试中先看模型卡），GQA 8 个 KV 头，head_dim 128，FP16 KV。每 token KV cache 多少？一条 32K 序列占多少？

**推导**：
```
每 token = 2 × 64 × 8 × 128 × 2 B
        = 262,144 B = 256 KiB
32K 序列 = 256 KiB × 32768 = 8 GiB
```

**面试加分点**：主动提一句"如果开 FP8 KV cache，直接减半到 4 GiB，代价是少量精度损失"——展示你知道优化的旋钮在哪。

### 例题 2：Decode 时延下界（理解题）

**题目**：70B 模型 FP8 权重，单卡 H100（3.35 TB/s），batch=1 时 decode 单 token 理论时延下界和理论吞吐上限是多少？为什么实测会更高？

**推导**：
```
权重字节数 = 70 × 10⁹ × 1 B = 70 GB
时延下界 = 70 GB / 3.35 TB/s ≈ 20.9 ms
理论上限 ≈ 48 tokens/s（单序列）
```

**为什么实测更高（时延）**：
1. 除权重外还要读 **KV cache**（长上下文时不可忽略：8K 上下文 ≈ 2.5 GB，再加 ~0.75ms）
2. kernel launch 开销与无法 100% 打满带宽（所以 decode 要用 CUDA Graph，Day 3 展开）
3. 激活、norm、embedding 等零碎访存

**面试加分点**：追问"怎么提高？"——答案链条：量化（字节数↓）→ batching（摊销）→ 投机解码（一次前向出多 token）→ 多卡 TP（带宽聚合）。这四个答案正好覆盖你这一周的全部专题，可以在回答时主动铺开。

### 例题 3：并发上限估算（综合题，白板高频）

**题目**：70B FP8 模型，8×H100（80GB）TP=8，gpu_memory_utilization=0.9，平均上下文 128K，FP16 KV（320 KiB/token，见模块二）。最多同时服务多少条序列？

**推导**：
```
每卡预算      = 80 GB × 0.9 = 72 GB
每卡权重分摊   = 70 GB / 8 = 8.75 GB
每卡开销预留   ≈ 5 GB（激活 + CUDA Graph 工作区 + 通信 buffer）
每卡 KV 可用   = 72 − 8.75 − 5 ≈ 58 GB
全集群 KV 池   = 58 × 8 ≈ 464 GB

总 token 容量  = 464 GB / 320 KiB ≈ 464×2³⁰ / 327,680 ≈ 152 万 tokens
并发序列数    = 152 万 / 131,072 ≈ 11~12 条
```

**结论与讨论（这步比算对更重要）**：
- 128K 满上下文下 8 卡只能撑 **约 12 条并发**——长上下文场景的显存经济性非常差
- 优化旋钮逐个说：FP8 KV cache（×2 → 24 条）→ 平均上下文没跑满 128K 时按实际长度分配（PagedAttention 的意义）→ P/D 分离让 decode 实例独占 KV 池 → offload 冷 KV 到 CPU/SSD
- 这就是为什么"PagedAttention 把显存浪费从 60-80% 降到 <4%"是个大事——晚上读论文时验证这个数字

### 自测题 2 道（不看答案做完再对）

**自测 1**：Llama-3-8B（32 层，8 KV 头，head_dim 128），FP16，单卡 4090（24GB，1 TB/s）。①每 token KV？②batch=1 decode 时延下界？③权重 16GB，留 2GB 开销，4K 平均上下文，最大并发？

**自测 2**：如果面试官问"同一个模型，为什么 prefill 阶段提升 batch 对吞吐几乎没帮助，decode 阶段却近似线性？"——用算术强度作答。

<details>
<summary>自测 1 参考答案</summary>

```
① 2 × 32 × 8 × 128 × 2 B = 131,072 B = 128 KiB/token
② 16 GB / 1 TB/s = 16 ms（≈ 62 tokens/s 上限）
③ KV 可用 = 24×0.9 − 16 − 2 ≈ 3.6 GB
   每条 4K 序列 KV = 128 KiB × 4096 = 512 MiB
   并发 ≈ 3.6 GB / 512 MiB ≈ 7 条
```

</details>

---

## 模块四：PagedAttention 论文导读（晚上，1.5h）

**论文**：《Efficient Memory Management for Large Language Model Serving with PagedAttention》（SOSP 2023），只读 §1、§3、§4。

### 阅读任务清单（带着问题读）

**§1 Introduction（20 min）**——抓住问题定义：
- [ ] 原方案的显存浪费数字：**60-80%**——来自哪三部分？（①为最大长度预留的冗余 ②长度不确定导致的过度预留 ③内部碎片）
- [ ] 为什么是 OS 虚拟内存启发的分页方案？

**§3 PagedAttention 设计（40 min）**——抓住三个映射：
- [ ] 分页映射：KV cache 切固定大小 **block**（vLLM 默认 16 token/block），逻辑块 → 物理块经 **block table** 间接寻址——类比你在昇腾做的 tile 寻址，只是这里间接一层
- [ ] 共享映射：同一 prompt 前缀 / parallel sampling 的多个分支共享物理 block，**引用计数**管理
- [ ] 写时复制（COW）：分支分叉时只复制最后一个 block

**§4 调度与实现（30 min）**——抓住两个机制：
- [ ] Iteration-level scheduling（continuous batching 的雏形）：每个 decoding step 重新决定 batch 组成，完成的请求立即退出、新请求立即插入
- [ ] Preemption：显存不够时逐出请求，恢复方式 recompute vs swap 的取舍

### 读完必须能画的图

合上论文，手绘：3 条请求共享一个前缀 block、各自分叉后的 block table 状态（标注每个物理 block 的引用计数）。这张图就是 Day 6 白板四件套之二的雏形。

> ⚡ **昇腾翻译提示**：block table 间接寻址 ≈ 你做 Nd2Nz 格式转换时的索引重排；PagedAttention 的"按需分配、消除预留"≈ 内存池化思想。面试讲这段时带上你在内存/数据布局上的工程直觉。

---

## 今日产出：一页纸《推理性能第一性原理》模板

照这个结构写满一页（A4），这就是作战包第 1 件：

```
1.【两阶段一图】prefill/decode 计算量-访存量对比表 + 算术强度结论
2.【三个公式】KV/token 公式 · decode 时延下界公式 · 并发估算公式
3.【三个数字直觉】H100 带宽 3.35TB/s · 70B 每 token KV 320KiB ·
   128K 单序列 KV 40GiB
4.【优化地图】每个瓶颈对应的旋钮：
   decode 慢 → 量化/batch/投机解码/TP
   显存爆 → GQA/FP8-KV/PagedAttention/offload
   TTFT 高 → chunked prefill/prefix caching/P-D 分离
5.【3 道手算题】例题 1-3 的完整推导（浓缩版）
```

---

## Day 1 收工自测清单（全绿才算完成）

- [ ] 3 分钟内手推出任意给定模型的 KV/token、decode 时延下界、并发上限
- [ ] 能用算术强度一句话解释 prefill 和 decode 的 bound 差异
- [ ] 能讲清 GQA 为什么省 KV 显存，以及公式里哪个参数体现
- [ ] 能说出 PagedAttention 解决的三类显存浪费和大致数字（60-80% → <4%）
- [ ] 能画出共享前缀的 block table + 引用计数
- [ ] 一页纸产出写完，放在明天抬眼能看到的地方

**未完成项不许带入 Day 2**——宁可压缩 Day 2 上午的架构泛读，也要把今天的公式练熟。手算题是白板第一题，翻车代价最高。
