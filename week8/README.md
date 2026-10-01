# Week 8：面试冲刺（详细展开版）

> **本周定位**：前 7 周是"输入与建构"，本周是"输出与收敛"。所有产出物在本周被打磨成面试时**带得走、讲得出、经得起追问**的形态。
> **节奏**：每天 3-4 小时，其中至少一半时间在"动嘴/动手"（录音、白板、被追问），而不是继续看新材料。
> **总原则**：本周不学新东西，只做三件事——**默写、推演、讲述**。发现漏洞记下来，当晚定点补，不发散。

---

## 本周材料与前 7 周产出的对应关系

| 冲刺材料 | 来源 | 本周动作 |
|---|---|---|
| 显存/时延手算 | W1《LLM 推理性能的第一性原理》 | Day 50 默写限时训练 |
| block table 图 | W3 KV Cache Manager 笔记 | Day 50 白板手绘 |
| 调度推演 | W2 Scheduler 笔记 + W4 mini 引擎实现 | Day 51 逐步推演 |
| 诊断树 | W2/W5 实验记录 + W7 消融报告 | Day 51 背诵 + 展开 |
| 高频问答 | W1-W5 全部笔记 | Day 52 逐题录音过堂 |
| 项目 A（vllm-ascend PR）/ B（mini 引擎）/ C（消融报告） | W3-W7 | Day 53 STAR 打磨 |
| 昇腾→GPU 方法论迁移叙事 | 简历项目一 | Day 53 主线叙事 |

---

## Day 50：白板四件套（一）——显存估算 + Block Table

### 50.1 显存估算默写（限时 3 分钟/题）

**必须肌肉记忆的两个公式**：

```
KV cache 每 token 显存 = 2 (K+V) × layers × kv_heads × head_dim × dtype_bytes
                      （GQA 用 kv_heads，不是 q_heads —— 面试最常见坑）

单 token decode 时延下界 ≈ 模型参数字节数 / HBM 带宽
                        （memory-bound 场景，忽略 KV 读与 kernel 开销）
```

**估算流程五步法**（口述模板）：

1. 权重显存 = 参数量 × dtype 字节数（70B BF16 ≈ 140GB）
2. 每 token KV 显存（上面的公式，先念 config 再代数）
3. KV 池 = 总显存 × gpu_memory_utilization − 权重 − 激活/图/框架开销（经验值 4-10GB）
4. 可缓存 token 数 = KV 池 / 每 token KV
5. 并发上限 = 可缓存 token 数 / (平均 prompt + 平均输出) × KV 利用率（~90%，留块边界余量）

**自测题 1（3 分钟）**：Qwen3-8B BF16 单卡 A100-80G，4K 上下文，能撑多少并发？

```
推导：
- config：36 层，32 Q heads / 8 KV heads（GQA），head_dim=128
- KV/token = 2 × 36 × 8 × 128 × 2B = 147,456B = 144KB
- 权重 = 8.2B × 2B ≈ 16.4GB；开销按 6GB 计
- KV 池 ≈ 80 × 0.9 − 16.4 − 6 ≈ 49.6GB（gpu_memory_utilization=0.9）
- 可缓存 token ≈ 49.6GB / 144KB ≈ 345K tokens
- 并发 ≈ 345K / 4096 ≈ 84 路（考虑 90% 利用率与块 padding，答 "80 路上下" 即可）
```

**自测题 2（3 分钟）**：Llama-3-70B FP8（权重+KV）双卡 H100-80G TP=2，4K 上下文并发上限？

```
- 70.6B × 1B = 70.6GB 权重 → TP=2 每卡 35.3GB
- KV/token（每卡，KV heads 均分 4 个）= 2 × 80 × 4 × 128 × 1B = 81,920B = 80KB
- 每卡 KV 池 ≈ 80 × 0.9 − 35.3 − 4 ≈ 32.7GB → 409K tokens/卡
- 并发 ≈ 409K / 4096 ≈ 100 路
```

**自测题 3（2 分钟，W1 Day2 原题）**：Llama-3-70B FP8 权重在 H100（3.35TB/s）上 decode TPOT 下界？

```
- 每 token 需读全部权重一次：70.6GB / 3.35TB/s ≈ 21.1ms
- → TPOT 下界 ≈ 21ms，即单序列 decode 上限 ≈ 47 tok/s
- 追问预备：BF16 则 42ms / 24 tok/s；GQA 下 KV 读远小于权重读，故忽略合理；
  实际 TPOT 高于下界的部分 = attention kernel 开销 + 调度 + 通信
```

**验收标准**：三题在 10 分钟内完成且数字误差 <10%。卡壳点（如 GQA 代错 heads、忘记 TP 分 KV）当场记录。

### 50.2 Block Table 白板手绘（含 prefix caching 命中 + COW 场景）

**场景设定**：block_size=16；请求 A（prompt 96 tokens）先运行，请求 B（相同 64-token 系统提示词 + 32-token 不同用户问题）后到达。

参考画法（面试时边画边讲）：

```
Block Pool（block_size = 16 tokens/block）

  Block0      Block1      Block2      Block3      Block4      Block5
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ sys 0-15 │ sys16-31 │ sys32-47 │ sys48-63 │ userA    │ userA    │
│ hash=h0  │ h1       │ h2       │ h3       │ h4       │ h5       │
│ ref=2    │ ref=2    │ ref=2    │ ref=2    │ ref=1    │ ref=1    │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
     ↑ ref=2：A、B 共享（B 命中 prefix cache，物理块复用，零拷贝零计算）

Req A block table: [0, 1, 2, 3, 4, 5]   ← 逻辑连续、物理离散
Req B block table: [0, 1, 2, 3, 6, 7]   ← 命中 h0..h3，B 的 32-token 问题
                                            写入新分配的 Block6/7
```

**讲解脚本要点**（60-90 秒）：

1. 逻辑块号 → 物理块的映射表即 block table，等价 OS 分页；逻辑连续物理离散
2. 块哈希 = hash(**父块哈希** + 本块 token ids + LoRA/多模态标识) → 哈希链天然防止"同内容不同位置"误命中，父链保证前缀语义正确
3. 命中路径：B 到达 → 沿哈希链逐块比对 → h0..h3 命中 → 引用计数 +1 → B 的 prefill 只需计算 32 tokens（TTFT 直接砍 2/3）
4. COW 场景：若 A、B 共享的是**未满块**且某方要继续追加写入（典型：beam search 分叉、两请求同 prompt 同步生成），写入方触发 copy-on-write——新分配一块、拷贝内容、旧块 ref−1
5. 满块不可变（内容寻址、天然可共享）；部分填充块可共享但追加时 COW
6. 内碎片上界 = 每 seq 最多 (block_size−1) 个 padding token，平均 block_size/2 —— 这就是"碎片率 60-80% → 接近 0"的答案来源

**验收标准**：不看笔记，5 分钟内画完并讲完上述 6 点。

---

## Day 51：白板四件套（二）——调度推演 + 性能诊断树

### 51.1 调度推演：10 个请求、KV 只够部分，逐步演算

**题目设定**：

```
block_size=16，KV 池 = 100 块（1600 token 槽位）
max_num_seqs = 10，max_num_batched_tokens = 512（chunked prefill 开启）
同一时刻到达 R1..R10（FCFS），每个 prompt=120，期望输出最长 160
块需求 = ceil(tokens / 16)
```

**逐步推演表**（面试时在白板上列关键快照）：

| Step | 事件 | running | waiting | budget 用途 | KV 块 | 备注 |
|---|---|---|---|---|---|---|
| 0 | 准入 R1-R4 | R1-R4 prefill(480) | R5-R10 | 480/512 | 4×8=32 | 准入校验整个 prompt 的块需求 |
| 1 | 准入 R5-R8 | R1-R4 decode(4) + R5-R8 prefill(480) | R9-R10 | 484/512 | 64 | prefill 与 decode 同 step 混排 |
| 2 | 准入 R9-R10 | R1-R8 decode(8) + R9-R10 prefill(240) | — | 248/512 | 80 | 全部 10 个进入 running |
| 3-18 | 纯 decode | R1-R10 各 +1 token/step | — | 10/512 | 90 | 每 16 步各 seq 跨块边界 +1 块 |
| 19-33 | 纯 decode | 同上 | — | 10/512 | 100 | 第 33 步末池满（若无请求完成） |
| 34 | **触发抢占** | R1-R9 | R10 | 9/512 | 90 | R10 已生成 32 tok，KV=152 tok=10 块被释放 |
| 35+ | R1-R9 decode | R1-R9 | R10 | 9/512 | ↑ | R10 状态回 waiting，num_computed_tokens 清零 |
| ~44 | R1 完成（输出 40） | R2-R9 | R10 | 8/512 | 释放 R1 的 10 块 | 队列开始排水 |
| ~45 | R10 重新准入 | R2-R10 | — | R10 需 prefill **152** tok | — | recompute：重算 prompt+已生成部分 |

**必须讲清的机制点**：

1. **抢占规则**：同优先级 FCFS → 抢**最后准入**的（最低优先级）；请求优先级字段可覆盖
2. **recompute vs swap**：V1 采用 recompute——释放全部块、重新计算（浪费 = num_computed_tokens 的 FLOPs）；V0 的 swap 把 KV 换到 CPU 内存（代价 = 2×KV_bytes 的 PCIe 传输 + CPU 内存占用，适合抢占频繁但重算贵的场景）
3. **为何不在准入时"留一手"**：准入校验的是 prompt 块需求，decode 增长是渐进的，池满才抢——这就是 preemption 计数增长的根源
4. **变体追问**：若 R11 与 R1 共享 64-token 系统提示词 → 命中 4 块，prefill 少算 64 tokens，准入块需求也减少

**验收标准**：不看笔记 10 分钟内推演完，抢占时机（step 34）与重算代价（152 tokens）能当场算出。

### 51.2 性能诊断树（完整版，要求能默写主干、按分支展开）

```
异常入口（与基线对比）
│
├─① TTFT p99 ↑（ITL 正常）
│   ├─ 看 waiting 队列深度 / queue duration
│   │   ├─ 队列深 → 准入受限
│   │   │   ├─ KV 块不足（blocks utilization ≈ 100%、preemption 计数 ↑）
│   │   │   │   → 动作：↓max_num_seqs / ↑gpu_memory_utilization
│   │   │   │             / KV cache FP8 / 限制 max_model_len / 扩容或 P/D 分离
│   │   │   └─ token budget 太小（每 step prefill 推进慢）
│   │   │       → 动作：↑max_num_batched_tokens（注意引发 ITL 尖刺，见②）
│   │   └─ 队列不深 → 单请求 prefill 本身慢（长 prompt 属正常）
│   │       → 区分 p50 与 p99：p50 也高 = 计算问题 → 量化 / 更强卡 / TP
│   └─ prefix hit rate 低 → 负载天然低复用 or 未开 prefix caching
│       → 动作：确认开关 / cache-aware routing（把同前缀路由到同实例）
│
├─② ITL/TPOT p99 ↑（TTFT 正常）
│   ├─ batch 太大 → 每 step 时间被拉长 → ↓max_num_seqs（牺牲吞吐保 SLO）
│   ├─ chunked prefill 尖刺（大块 prefill 混入 decode step）
│   │   → 动作：↓max_num_batched_tokens 或 P/D 分离
│   ├─ preemption 增长 → 走①的 KV 路径（重算同时拖垮 TTFT 与 ITL）
│   ├─ CUDA Graph 未命中（batch 超出 capture 的 bucket → 回退 eager）
│   │   → 动作：检查 capture sizes 覆盖实际并发分布
│   ├─ CPU 调度开销（nsys 显示 step 间 bubble、CPU-bound）
│   │   → 动作：开 async scheduling（调度 N+1 与执行 N 重叠）
│   └─ TP 通信占比高（nsys 看 NCCL 时间占比）
│       → 动作：↓TP 度 / 确认 NVLink 拓扑 / 检查 all-reduce 融合
│
├─③ 吞吐 ↓（同 SLO 下）
│   ├─ batch 上不去 → KV 不足 → 走①
│   └─ raw throughput ↑ 但 goodput ↓ → SLO 违约 → 刻意降 batch（反直觉但正确）
│
└─④ 精度劣化 → 对照实验：关 KV 量化 / 换 W8A16 回退 / 看 perplexity 与下游任务
```

**背诵口径**（主干一句话版）：**"TTFT 看队列，ITL 看单步，队列看准入，准入看 KV；KV 满了抢，抢占靠重算；ITL 抖看混排，通信看 nsys。"**

---

## Day 52：高频问题清单过堂（每题录音自答 3 分钟）

录音方法：每题连续答 3 分钟不许停，回听打分（结构完整 / 有数字 / 有 trade-off / 无口头禅），低于 70 分重录。

### Q1 PagedAttention 解决了什么问题？碎片率怎么算？

- **一句话**：把 KV cache 从"连续分配"改成"固定块 + 页表"，消除内/外碎片，显存利用率从 20-40% 提到 96%+
- **问题**：连续预留 max_len → 内碎片 = (max_len−actual_len)×每 token KV；多次分配释放后 → 外碎片（无连续大块）。论文测得浪费 60-80%
- **方案**：block_size=16 固定块、block table 映射、引用计数共享、COW；内碎片上界 = 每 seq (block_size−1) token，平均 block_size/2；外碎片 = 0
- **外溢收益**：内容寻址的块是 prefix caching 的基础设施；共享块省计算省拷贝
- **追问预备**：为什么不把 block_size 设大/小？→ 大：内碎片↑、共享粒度粗；小：页表开销↑、gather kernel 间接寻址成本↑。16 是权衡值

### Q2 continuous batching vs static batching？调度粒度？

- **一句话**：调度粒度从"整个序列完成"细化到"每个 iteration"，完成的立刻移出、新请求立刻插入
- **static**：凑批 → 全体跑完才换血，短序列被长序列拖住，GPU 大量空转
- **iteration-level**（vLLM V1 统一调度器 / TGI 的 in-flight batching）：每 step 执行 `schedule()` 重新组 batch；prefill 与 decode 同 step 混排（chunked prefill）
- **数字**：典型吞吐提升 2-4×，越混合长短的负载收益越大
- **追问预备**：混排的代价 = 调度器每 step 都要重算 batch 布局 → CPU 开销 → 这正是 async scheduling 与 CUDA Graph 要解决的对象

### Q3 为什么 decode 用 CUDA Graph 而 prefill 不用？

- **一句话**：decode 的瓶颈是 kernel launch 与 CPU 调度开销（几百个小 kernel），prefill 的瓶颈是计算本身
- **decode**：batch 小、每 kernel 微秒级，80 层 × 多算子 ≈ 数百次 launch，CPU 喂不饱 GPU → 捕获整图，一步一次 replay
- **prefill**：kernel 时间长，launch 占比可忽略；且 prompt 长度任意 → 形状组合爆炸，逐 bucket 捕获不划算还费显存
- **V1 进阶**：piecewise CUDA graph——只捕获 attention 之外的静态部分，attention（形状多变）留 eager，兼顾两边
- **个人叙事**（加分）：这与我做昇腾算子"编译期确定分块参数、消除运行时开销"是同一方法论——**用静态化换确定性，消灭动态分支**

### Q4 chunked prefill 的 trade-off？token budget 怎么设？

- **收益**：长 prompt 不再独占一个 step → decode 不被饿死 → **TPOT 抖动↓**；prefill 与 decode 混排提高 GPU 利用
- **代价**：prefill 被拉长 → **TTFT 略升**；实现复杂（部分计算、块分配与重算边界）；每 step 组 batch 开销
- **budget 设定逻辑**：每 step 时间 ≈ budget_tokens / prefill 吞吐 ≤ 目标 TPOT 上限。如 prefill 吞吐 20K tok/s、TPOT 目标 50ms → chunk ≤ 1000 tokens（还要扣除 decode 占用的 token 数）。经验范围 2048-8192，**必须实测**——太小 TTFT 崩，太大 ITL 尖刺
- **追问预备**：为什么 V1 默认开？→ 生产负载 prompt 长短混合，不切的 TPOT p99 不可接受

### Q5 量化对 TTFT 和 TPOT 的影响分别是什么？

- **TPOT（decode，访存 bound）**：权重读取量减半（W8/FP8）→ 理论上限 2×，实际 1.3-1.8×（KV/激活读取不变、非 GEMM 开销稀释）
- **TTFT（prefill，计算 bound）**：FP8 GEMM 峰值 2×（H100：1979 vs 989 TFLOPS）→ 实际 1.2-1.6×（attention 与 KV 写不受益）
- **二次收益**：权重省一半 → KV 池变大 → 并发上限↑ → 吞吐进一步靠近 roofline 峰值
- **失效模式**：outlier 导致精度崩（→ per-channel/block-wise scale、SmoothQuant/AWQ 思路）；KV 量化伤长上下文检索精度
- **个人叙事**（核心加分项）：直接对接 WeightQuantBatchMatmulV2 经验——"我在昇腾上做的就是量化 MatMul 的融合算子，outlier 处理、多精度指令序列选择、反量化融合这些工程问题我踩过一遍"

### Q6 P/D 分离什么时候不值得做？

- **值得做的条件**：prefill/decode 负载都重、TTFT 与 TPOT 双 SLO、规模大到能同时填满两组机器、有 RDMA
- **不值得**：①并发低/负载轻 → 分离后两组机器都半载，资源碎片化 ②prompt 短输出长 → prefill 占比小、干扰本就小 ③KV 传输成本吃掉收益（无 RDMA；或 KV 巨大：传输量 = prompt_tokens × KV/token，如 70B BF16 4K prompt = 320KB × 4K ≈ **1.3GB/请求**）④小团队扛不住双平面运维与故障域翻倍
- **数字预备**：分离收益典型 1.5-3× goodput 提升，但前提是 KV 传输能被分层流水掩盖

### Q7 TP 开到什么时候是负收益？

- **前提**：能单卡放下就别 TP（通信是纯税）
- **负收益机制**：①每层 2 次 all-reduce，通信量 ∝ batch×hidden，不随 TP 减少而计算每卡减为 1/N → 计算/通信比恶化 ②decode 小 batch 时消息小、latency 主导，带宽利用率低 ③跨机 TP 走网络更是灾难
- **经验**：NVLink 域内（≤8 卡）prefill 通常 TP=4/8 仍正收益；**decode 小 batch 下 TP=2 就可能负收益**；跨机一律考虑 PP/EP 替代
- **对照**：MoE 用 EP + all-to-all，专家并行把通信换成 token 路由——不同通信模式对应不同负载形态

### 加练题（每题 1 分钟骨架）

8. **MLA 为什么省显存**：KV 联合压缩到 latent（DeepSeek-V3：每层 576 维 ≈ 70KB/token BF16，对比传统 MHA 数倍）代价是 attention 需要权重吸收后的矩阵乘
9. **投机解码何时负收益**：接受率 α 低 + draft 成本高；期望 token 数 ≈ (1−α^(k+1))/(1−α)，α<0.5 时收益覆盖不了 draft 与验证开销
10. **async scheduling 原理**：CPU 调度 step N+1 与 GPU 执行 step N 流水重叠，消除 step 间 bubble；前提是调度决策不依赖 N 的输出（speculative scheduling）
11. **prefix caching 的哈希链为什么安全**：父哈希入参 → 位置敏感 + 链式传递；ABA 由 token ids 全量入参排除
12. **goodput vs throughput**：满足 SLO 的有效吞吐；raw 吞吐涨而 goodput 跌 = batch 过大违约，生产系统唯一该看的指标
13. **为什么 V1 用多进程拆 EngineCore**：把 GIL 与引擎循环和前端 API 进程隔离，前端崩溃不传染引擎，且便于前后端独立伸缩与异步化

---

## Day 53：项目讲述打磨

### 53.1 三个项目的 STAR 模板

统一结构：**背景 1 句 → 难点 1-2 句 → 动作（方法论）→ 量化结果 → 反思/局限**。3 分钟版砍掉细节只留主线，10 分钟版加白板图与备选深入方向。

**项目 A：vllm-ascend 源码贡献（性能优化 PR）**

| STAR | 内容（用你的真实数据填充 ▢） |
|---|---|
| S 背景 | vllm-ascend 作为 vLLM 的 NPU 后端，目标算子/路径性能落后于理论峰值 ▢% |
| T 难点 | 官方工具链不熟悉 + 现有实现瓶颈不明，需要先建基线再定位 |
| A 动作 | 用 profiler 定位热点 → 数据流/访存模式分析 → 套用 bound 建模求理论上限 → 实施 ▢ 优化 → 前后 benchmark + 精度验证 |
| R 结果 | 该路径性能提升 ▢%（P50/P99 数据），PR #▢，review 意见闭环 |
| R 反思 | 如果重做会 ▢；当前方案局限在 ▢ 场景 |

追问预测：基线怎么保证可复现？优化的收益怎么归因到你的改动（控制变量）？精度怎么验证？为什么官方没发现这个问题？

**项目 B：mini 推理引擎**

| STAR | 内容 |
|---|---|
| S 背景 | 为验证对 continuous batching / KV 管理的理解，纯 Python 复刻 vLLM 核心机制 |
| T 难点 | iteration 级调度与块管理的边界条件（抢占重算、COW、budget 竞争） |
| A 动作 | 实现固定块 KV 池 + block table + 引用计数 + chunked prefill + preemption；static batching 对照实验 |
| R 结果 | 相同负载下吞吐 ▢×（典型 2-3×），并产出调度行为可视化 |
| R 反思 | 与 vLLM 差距集中在 kernel 层（我用的是朴素 attention），调度层行为一致 |

追问预测：与 vLLM 的 scheduler 差异？为什么不做 X？如果加 prefix caching 要改哪些结构？（→ 回答引到哈希链与块池）

**项目 C：vLLM 性能消融实验报告**

| STAR | 内容 |
|---|---|
| S 背景 | 团队/社区缺少系统的"开关 × 负载"性能画像，调参靠玄学 |
| T 难点 | 变量控制与指标采集（Prometheus + Grafana 看板搭建） |
| A 动作 | 4 组消融：chunked prefill × prompt 分布 / prefix hit rate 梯度 / 投机接受率-收益曲线 / 量化吞吐-精度权衡 |
| R 结果 | 每组一张图 + 机制解释；产出调参决策表（什么负载开什么开关） |
| R 反思 | 结论受限于单卡 8B 模型，规模外推需谨慎 |

### 53.2 "从昇腾到 GPU"主线叙事（面试官最想听的部分）

**30 秒版**："我在昇腾上做了 3 年量化矩阵乘算子优化，核心方法论是——先建访存/计算 bound 分界模型，再据此刻意设计 tiling、流水线与精度路径。GPU 与 NPU 架构不同，但这套'第一性原理 + 定量建模'的方法是平台无关的，我用它重新推导了 vLLM 的性能边界并完成了 ▢ 优化。"

**概念映射表**（面试时主动画出，展示迁移能力）：

| 昇腾概念 | GPU 对应 | 共同本质 |
|---|---|---|
| Cube 单元 | Tensor Core | 矩阵乘专用算力 |
| Vector 单元 | CUDA Core (SIMT) | 逐元素/规约 |
| L1 / L0A/L0B/L0C | shared memory / 寄存器 | 片上级联与数据复用 |
| MTE2/MTE3 搬运 | cp.async / TMA | 异步搬运与预取 |
| Fixpipe | GEMM epilogue | 尾部融合（bias/量化） |
| AI Core 多核 | SM 多卡 | 负载均衡与长尾消除 |
| 双 buffer 乒乓 | double buffering | 访存计算重叠 |
| 编译期定 tiling | CUDA Graph / 模板特化 | 静态化消灭运行时开销 |
| CANN 算子库 | CUTLASS/Triton | kernel 抽象层 |

**三个可以直接引用的实战证据**（来自简历，准备好展开）：

1. ASW 蛇形滑窗 + L2 命中率优化 → 迁移话题：GPU 上的 tiling 与 L2/共享内存复用策略（Swizzling）
2. CalRebalanceBlock 基于 bound 分界模型的分块搜优 → 迁移话题：roofline 驱动的 kernel 参数选择，正是本周诊断树的底层逻辑
3. 无 Queue 手工流水线（SetFlag/WaitFlag 事件驱动）→ 迁移话题：persistent kernel、software pipeline、CUDA Graph 消除调度开销——与 Q3 呼应

---

## Day 54-55：模拟面试 × 2

### 轮次 1：技术深挖面（45 分钟）

| 时间 | 环节 | 内容 |
|---|---|---|
| 0-3min | 自我介绍 | 30 秒昇腾叙事主线 + 3 个项目一句话各带一个数字 |
| 3-18min | 项目深挖 | 面试官选项目 A，连环追问 5 层（见 53.1 追问预测），考察是否被击穿 |
| 18-33min | 基础连环 | 从 Day 52 的 Q1-Q7 抽 3 题，每题答完必追问"代价是什么/什么时候失效" |
| 33-43min | 现场诊断 | 给一组指标（TTFT p99 从 800ms 涨到 4s，ITL 稳定 40ms，preemption 计数 0，queue depth 高企）→ 要求 5 分钟内给出诊断与动作 |
| 43-45min | 反问 | 见 Day 56 反问清单 |

### 轮次 2：系统设计面（45 分钟）："为日活千万的客服机器人设计推理集群"

参考作答骨架（练到能 30 分钟画完）：

1. **需求量化（5min）**：日活千万 → 峰值 QPS 估算（假设早高峰 10% 同时在线、人均 1 条消息/分钟 → 峰值 ~1.6K QPS）；prompt 分布（系统提示词占 80%，前缀复用率高）；SLO：TTFT p99 < 1s、TPOT < 50ms
2. **容量估算（5min，手算）**：选定模型（如 8B BF16）→ 单卡并发 ~80（Day 50 结论）→ 单卡吞吐 = 80 seq × 20 tok/s ≈ 1.6K tok/s → 按人均输出 200 token 与 QPS 反推卡数 → 留 30% 冗余
3. **架构（15min）**：网关 → cache-aware 路由（相同前缀粘同实例，提升 hit rate）→ 推理层（多副本 + 前缀缓存；负载超预期再上 P/D 分离）→ 弹性伸缩（按队列深度扩容）
4. **可观测（5min）**：TTFT/TPOT/goodput 分位数、prefix hit rate、preemption 计数、队列深度 → 触发告警与自动降级（超时降 batch）
5. **多租户与安全（3min）**：cache_salt 防跨租户缓存泄漏、配额与频控
6. **演进（2min）**：分层 KV 池（GPU→CPU→SSD）、全局前缀路由、量化迭代

### 评分表（每维度 1-5 分，两轮均 ≥20 分/25 及格）

| 维度 | 评分要点 | 得分 |
|---|---|---|
| 正确性 | 概念、机制、数字无硬伤 | |
| 深度 | 追问 3 层不崩，能下钻到 kernel/数据结构 | |
| 结构化表达 | 先结论后展开、有框架不发散 | |
| 数字敏感度 | 关键结论带量级（几倍、多少 ms、多少 GB） | |
| trade-off 意识 | 每个方案主动讲代价与失效条件 | |

### 卡壳点记录表（每次模拟面试后 30 分钟内填写）

| 问题 | 卡壳类型（不知道 / 知道说不出 / 被追问击穿） | 补救动作 | 补后复测日期 |
|---|---|---|---|
| | | | |

**常见失败模式自查**：背答案感强（无眼神/语速单一）、全程无数字、不画图纯口述、被打断后逻辑散架、不确认需求就开答。命中任意一条 → 当晚针对重练。

---

## Day 56：收官——材料整理 + 面试日协议

### 56.1 最终材料清单（面试作品集）

- [ ] 《LLM 推理性能的第一性原理》笔记（W1）
- [ ] vLLM V1 架构图 + 源码走读笔记（W2-W3）
- [ ] mini 推理引擎 README（架构图 + 性能对比，W4）
- [ ] 四份 A4 专题总结：量化 / 投机解码 / P-D 分离 / 分布式（W4-W5）
- [ ] vllm-ascend PR + 前后性能数据表（W6-W7）
- [ ] 消融实验报告（W7）
- [ ] 白板四件套：显存估算模板图 / block table 图 / 调度推演表 / 诊断树（W8 本周产出）
- [ ] 三个项目 STAR 讲稿（3min + 10min 双版本）+ 昇腾→GPU 映射表（W8）
- [ ] 两轮模拟面试评分表 + 卡壳点闭环记录（W8）

### 56.2 必背数字卡（面试当天早上最后过一遍）

| 项目 | 数值 |
|---|---|
| A100-80G | 2.0TB/s，BF16 312 TFLOPS（无 FP8） |
| H100-80G SXM | 3.35TB/s，BF16 989 / FP8 1979 TFLOPS |
| H200 / B200 | 141GB 4.8TB/s / 192GB 8TB/s |
| NVLink | A100 600GB/s，H100 900GB/s |
| KV/token（BF16） | Llama-3-8B 128KB；Qwen3-8B 144KB；Llama-3-70B 320KB；DeepSeek-V3 (MLA) ≈70KB |
| 8B BF16 decode 下界（A100） | 16.4GB / 2TB/s ≈ 8ms → ~120 tok/s |
| 70B FP8 decode 下界（H100） | ≈21ms → ~47 tok/s |
| PagedAttention 显存浪费（改前） | 60-80% |
| continuous batching 吞吐收益 | 2-4× |
| P/D 分离 goodput 收益 | 1.5-3×（前提：KV 传输被流水掩盖） |
| H100 FP8 GEMM vs BF16 | 2× 峰值 |

### 56.3 面试前 24 小时协议

1. **只看自己写的东西**（数字卡 + 四件套 + 讲稿），不看任何新论文/新代码
2. 卡壳点记录表逐条口头复述一遍，未闭环的最后确认
3. 白板四件套完整默写一遍计时（总时长 ≤25 分钟）
4. 准备 3 个反问：团队当前最痛的性能瓶颈是什么 / 新人前三个月的期望产出 / 算子层与系统层团队的协作边界
5. 早睡。面试当天上午只过数字卡，不过任何推导

---

## 本周打卡表

- [ ] Day 50：三道手算题 10 分钟内完成 + block table 白板过关
- [ ] Day 51：调度推演 10 分钟内完成 + 诊断树主干默写过关
- [ ] Day 52：7 道高频题录音各 3 分钟，全部 ≥70 分
- [ ] Day 53：三项目 3 分钟版讲稿脱稿 + 昇腾→GPU 叙事 30 秒版流畅
- [ ] Day 54：模拟面试轮次 1 完成，评分 ≥20/25，卡壳点闭环
- [ ] Day 55：模拟面试轮次 2（系统设计）完成，评分 ≥20/25，卡壳点闭环
- [ ] Day 56：材料清单全勾 + 数字卡最后过一遍

> **收官自检**：如果明天就是终面，你现在最心虚的一页是什么？——那就是今天最后要补的东西。补完就收工。
