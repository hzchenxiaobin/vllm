# 推理系统优化专家岗 · 八周按天打卡计划

> **目标**：通过 LLM 推理系统优化（vLLM 方向）专家岗位面试
> **基础**：昇腾 NPU 算子优化经验（Tiling / 流水线 / 量化）+ 高并发分布式架构经验
> **节奏**：每天 2-4 小时，每周末半天复盘；全程以 vLLM V1 架构为准（V0 已移除，不要读旧代码）
> **环境**：建议租 1 张 A100/H100（或本地 4090），装好 vLLM、PyTorch、nsys/ncu、Prometheus + Grafana

---

## 第 1 周：推理基础与性能建模（第一性原理）

**本周目标**：把 LLM 推理变成能手算的数学题，建立 GPU 版 Roofline 思维。

- [ ] **Day 1**：Transformer 推理机制
  - 精读 prefill vs decode 的本质区别：prefill 计算密集（compute-bound），decode 访存密集（memory-bandwidth-bound）
  - 搞懂 KV cache 的生成与复用过程，手画一次 decode 迭代的张量流
  - 产出：笔记《prefill/decode 计算与访存量推导》
- [ ] **Day 2**：显存与时延的手算公式（面试必考）
  - KV cache 每 token 显存 = `2 × layers × kv_heads × head_dim × dtype_bytes`（注意 GQA 用 kv_heads）
  - Decode 单 token 理论时延下界 ≈ 模型参数字节数 / HBM 带宽
  - 练习题：Llama-3-70B FP8 在 H100（3.35TB/s）上的 decode 时延下界是多少？
  - 产出：3 道手算题完整推导过程
- [ ] **Day 3**：Roofline 模型（GPU 版）
  - 对照你做过的"访存/计算 bound 分界模型"，理解 GPU 上 arithmetic intensity 的判定
  - 用 ncu 跑一个小 kernel 看 SM busy / DRAM busy 指标
  - 产出：笔记《从昇腾 bound 建模到 GPU Roofline 的映射表》
- [ ] **Day 4**：PagedAttention 论文精读（SOSP 2023）
  - 核心问题：KV cache 的显存碎片与浪费（原方案浪费 60-80%）
  - block/table/引用计数/COW 的设计动机
  - 产出：论文精读笔记，标注 3 个你觉得最巧的设计点
- [ ] **Day 5**：Serving 指标体系
  - TTFT、TPOT/ITL、E2E latency、throughput、goodput 的定义与关系
  - SLO 驱动思维：为什么生产系统按 goodput 而非 raw throughput 评估
  - 产出：指标定义卡片（面试随时抽背）
- [ ] **Day 6**：环境搭建 + 第一次压测
  - 部署 vLLM（最新版），启动 Qwen3-8B 服务
  - 跑 `vllm bench serve`，用 ShareGPT 数据集，观察 TTFT/TPOT 随并发的变化
  - 产出：第一张性能曲线图（并发 vs TTFT p99 / TPOT p99）
- [ ] **Day 7（复盘日）**：
  - 整理本周笔记为《LLM 推理性能的第一性原理》一篇
  - 自测：不看笔记，手推 70B 模型的显存与并发上限估算

---

## 第 2 周：vLLM V1 源码精读（上）——调度链路

**本周目标**：讲清一个请求从进来到出 token 的完整生命周期。

- [ ] **Day 8**：V1 架构总览
  - 读官方博客/文档中 V1 架构设计（分离的 EngineCore 进程、多进程结构）
  - 梳理进程关系：`AsyncLLM` → `Processor` → `EngineCore` → `Executor/Worker` → `ModelRunner`
  - 产出：V1 进程架构图（自己画）
- [ ] **Day 9**：请求入口链路
  - 源码走读：`vllm/v1/engine/` 下 processor 与 llm_engine
  - tokenize → Request 对象构造 → 加入 scheduler 的过程
  - 起一个服务，打日志跟踪一个请求（贯穿本周每天验证）
- [ ] **Day 10**：Scheduler（一）——队列与 budget
  - `vllm/v1/core/scheduler.py`：waiting/running 队列、FCFS 与优先级
  - `max_num_batched_tokens` token budget 如何决定每步调度多少请求
- [ ] **Day 11**：Scheduler（二）——chunked prefill
  - 长 prompt 如何被切块，切块与 decode 如何混排在同一个 step
  - 思考：为什么 chunked prefill 能降低 TPOT 抖动？代价是什么？
  - 产出：一页总结《chunked prefill 的收益与代价》
- [ ] **Day 12**：Scheduler（三）——preemption
  - 抢占的触发条件（KV block 不足）、recompute vs swap 两种模式
  - 什么指标说明系统在频繁抢占？怎么调？
- [ ] **Day 13**：动手验证调度行为
  - 用压测构造：长 prompt 洪峰（观察 chunked prefill）、高并发挤爆 KV（观察 preemption）
  - 对照 `/metrics` 中 preemption 计数、queue time 变化
  - 产出：实验记录（现象 → 源码机制 → 指标表现 三段对照）
- [ ] **Day 14（复盘日）**：
  - 画出"一个请求在 scheduler 中的状态机"
  - 自测：口头推演"10 个请求、KV 只够 6 个"的完整调度过程

---

## 第 3 周：vLLM V1 源码精读（下）——KV 管理与执行

**本周目标**：吃透 KV cache 管理与模型执行层，开始同步动手写 mini 引擎。

- [ ] **Day 15**：KV Cache Manager（一）
  - `vllm/v1/core/kv_cache_manager.py` + block pool + free block 队列
  - block table 的数据结构，allocate/free/append 的路径
- [ ] **Day 16**：KV Cache Manager（二）——prefix caching
  - block hash 机制：hash 包含父前缀 + token ids + 多模态/LoRA 标识
  - COW（copy-on-write）与引用计数；`enable_prefix_caching` 开关行为
  - 实验：构造高重复前缀负载，观察 cache hit rate 与 TTFT 改善
- [ ] **Day 17**：Attention 后端抽象
  - FlashAttention / FlashInfer / Triton backend 的插拔接口（`vllm/attention/`）
  - paged KV 的 gather kernel 如何读取非连续 block
  - 思考：如果让你给新硬件写 backend，要实现哪些接口？（和你的昇腾经验对接）
- [ ] **Day 18**：CUDA Graph
  - full CG vs piecewise CG 的区别与取舍
  - 为什么 decode 阶段必须用 CUDA Graph；capture 的 batch size bucket 策略
  - 实验：`-O` 不同 compilation level 下 TPOT 对比
- [ ] **Day 19**：Async scheduling 与 CPU 开销隐藏
  - 调度与执行如何流水重叠；为什么 V1 把 async scheduling 设为默认方向
  - 用 nsys 看一次 decode step 的 CPU/GPU 时间线，找 bubble
- [ ] **Day 20**：mini 引擎开工（项目 B，持续 3 天）
  - 实现：固定大小 block 的 KV 池 + block table + 引用计数（纯 Python）
- [ ] **Day 21（复盘日 + 项目）**：
  - mini 引擎：实现 iteration 级 continuous batching 调度器（waiting/running + token budget）
  - 复盘：画 vLLM V1 完整数据流大图（请求 → 调度 → KV 分配 → 执行 → 输出）

---

## 第 4 周：进阶专题（一）——量化与投机解码

**本周目标**：两个高频专题达到"原理 + 场景 + 权衡 + 失效模式"四段式水平。

- [ ] **Day 22**：量化基础串讲
  - W8A8 / W4A16 / FP8（per-tensor / per-channel / block-wise）的区别
  - 激活量化的难点（outlier）与 SmoothQuant/AWQ/GPTQ 思路
  - 结合你的 WeightQuantBatchMatmul 经验：整理一页《昇腾量化算子 vs GPU 量化 GEMM 对照》
- [ ] **Day 23**：KV cache 量化
  - FP8/INT4 KV cache 的精度代价与收益；什么负载适合开
  - 实验：Qwen3 FP8 + KV cache FP8，测吞吐/显存变化 + 简单精度对比
- [ ] **Day 24**：量化动手 + 总结
  - 用 llm-compressor 或官方 FP8 checkpoint 跑一次完整量化 serving
  - 产出：专题 A4 总结《量化：原理/场景/权衡/失效模式》
- [ ] **Day 25**：Speculative Decoding 原理
  - 为什么投机解码是"用计算换访存"（联系 decode 访存 bound 本质）
  - draft model / MTP / EAGLE-3 三条路线的差异
  - 接受率的决定因素；收益公式：加速比 ≈ f(接受率, 草稿成本)
- [ ] **Day 26**：投机解码实验
  - vLLM 开启 MTP/EAGLE 投机解码，分别用代码补全负载（高接受率）与开放对话负载（低接受率）测试
  - 观察 `num_speculative_tokens` 调参的影响；验证低接受率下吞吐反而下降的失效模式
- [ ] **Day 27**：mini 引擎收尾（项目 B）
  - 实现 chunked prefill 与 preemption
  - 写一个 benchmark 对比 static batching vs 你的 continuous batching
  - 产出：项目 README（架构图 + 性能对比数据）——面试作品集素材
- [ ] **Day 28（复盘日）**：
  - 产出：专题 A4 总结《投机解码》
  - 自测：口头回答"投机解码什么时候是负收益？"

---

## 第 5 周：进阶专题（二）——P/D 分离与分布式

**本周目标**：掌握生产级推理系统的架构设计话题（专家岗区分度最高的部分）。

- [ ] **Day 29**：P/D 分离（一）——为什么
  - prefill 与 decode 硬件特性相反、混跑互相干扰的本质
  - SLO 视角：为什么分离后 goodput 可提升 1.5-3 倍
- [ ] **Day 30**：P/D 分离（二）——怎么做
  - 同节点分离（NVLink 传 KV）vs 跨节点分离（RDMA + 分层流水隐藏传输）
  - 读 Mooncake / NVIDIA Dynamo / llm-d 的设计文章，对比三者取舍
- [ ] **Day 31**：P/D 分离实验
  - 用 vLLM 的 disaggregated serving 示例（或 Dynamo）搭一个最小 P/D 分离部署
  - 产出：专题 A4 总结《P/D 分离》+ 部署记录
- [ ] **Day 32**：分布式并行（一）
  - TP 的通信模式（all-reduce）与 NVLink 拓扑约束；为什么"能单卡放下就别上 TP"
  - PP 的 bubble 问题；MoE 模型的 EP 与 all-to-all
- [ ] **Day 33**：分布式并行（二）实验
  - 双卡 TP=2 vs 单卡对比实验：吞吐、TPOT、通信占比（nsys 看 NCCL 时间）
  - 产出：专题 A4 总结《分布式推理》
- [ ] **Day 34**：prefix caching 与路由
  - cache-aware routing：把相同前缀的请求路由到同一实例
  - 多租户隔离：cache_salt 防跨租户缓存泄漏
  - 大规模集群视角：全局 KV 池、分层存储（GPU→CPU→SSD）的方向
- [ ] **Day 35（复盘日）**：
  - 整理第 4-5 周共 4 份 A4 专题总结，模拟白板互讲一遍
  - 自测：设计题"为一个日活千万的客服机器人设计推理集群架构"

---

## 第 6 周：项目 A —— vLLM-Ascend / 源码贡献（上）

**本周目标**：选定贡献点，打通开发流程。（项目 A 是简历级产出，贯穿两周）

- [ ] **Day 36**：选型与调研
  - 浏览 vllm-ascend（或 vLLM 主仓）的 open issues / roadmap，找性能相关切入点
  - 候选方向：paged attention kernel 优化、量化 GEMM tiling、MLA 后端、调度器小改进
  - 产出：选题备忘录（问题、影响面、预计工作量）
- [ ] **Day 37**：环境打通
  - 搭好 vllm-ascend 开发环境，跑通单测与 benchmark 基线
  - 记录基线性能数据（改之前先量化）
- [ ] **Day 38-40**：性能剖析定位瓶颈
  - 用 profiler（Ascend 侧工具或 nsys）定位目标路径的热点
  - 画出热点的数据流与访存模式，套用你的 tiling/bound 建模方法分析理论上限
  - 产出：瓶颈分析报告（现状数据 → 理论上限 → 优化空间）
- [ ] **Day 41-42**：实施优化（第一轮）
  - 编码实现，每步保持可测试
  - 周末复盘：整理中途记录，确保思路可对外讲述

---

## 第 7 周：项目 A（下）+ 项目 C 消融实验

- [ ] **Day 43-45**：优化迭代 + 验证
  - 完整 benchmark 前后对比；边界 case 与精度验证
  - 整理为 PR 提交（即使不合并，review 过程本身就是材料）
  - 产出：PR + 前后性能数据对比表
- [ ] **Day 46**：项目 C 启动——实验设计
  - 设计 4 组消融：①chunked prefill 开关 × prompt 长度分布；②prefix caching 命中率梯度；③投机解码接受率-收益曲线；④量化吞吐-精度权衡
  - 写好压测脚本与指标采集（Prometheus + Grafana 看板）
- [ ] **Day 47-48**：跑实验 + 出报告
  - 产出：《vLLM 性能消融实验报告》（图表 + 结论 + 机制解释）——求职作品集核心
- [ ] **Day 49（复盘日）**：
  - 把项目 A/B/C 整理成简历 bullet 与面试讲稿（背景 → 动作 → 量化结果）

---

## 第 8 周：面试冲刺

**本周目标**：白板四件套 + 模拟面试 + 表达打磨。

- [ ] **Day 50**：白板四件套（一）
  - 默写显存估算：任意给定模型/精度/上下文长度，3 分钟内算出 KV 显存与并发上限
  - 手绘 block table：含 prefix caching 命中、COW 分裂场景
- [ ] **Day 51**：白板四件套（二）
  - 调度推演：给定 10 个请求和 KV 容量，推演每个 step 的 running batch、抢占与恢复
  - 性能诊断树：背熟并能展开——TTFT 升/ITL 稳 → 队列与 prefill 拥塞；ITL 升 → batch 过大或通信瓶颈；preemption 增长 → KV 超配调 `max_num_seqs`
- [ ] **Day 52**：高频问题清单过堂（每个问题录音自答 3 分钟）
  - PagedAttention 解决了什么问题？碎片率怎么算？
  - continuous batching vs static batching？in-flight batching 的调度粒度？
  - 为什么 decode 用 CUDA Graph 而 prefill 不用？
  - chunked prefill 的 trade-off？token budget 怎么设？
  - 量化对 TTFT 和 TPOT 的影响分别是什么？
  - P/D 分离什么时候不值得做？
  - TP 开到什么时候是负收益？
- [ ] **Day 53**：项目讲述打磨
  - 每个项目用 STAR + 量化结果讲 3 分钟版 / 10 分钟版各一遍
  - 准备好"从昇腾到 GPU"的叙事主线（方法论跨平台迁移）
- [ ] **Day 54-55**：模拟面试 × 2
  - 找同行或用模拟面试工具，重点练：追问下的 trade-off 讨论、数字敏感度、诊断思路
  - 记录卡壳点，当晚补漏
- [ ] **Day 56（收官）**：
  - 整理最终材料：架构图、诊断树、四份专题总结、项目讲稿
  - 查漏补缺，面试前最后一天只看自己写的总结，不看新东西

---

## 附：每周固定产出物清单（面试作品集）

| 产出物 | 完成周 |
|---|---|
| 《LLM 推理性能的第一性原理》笔记 | W1 |
| vLLM V1 架构图 + 源码走读笔记 | W2-W3 |
| mini 推理引擎（项目 B） | W3-W4 |
| 量化 / 投机解码 / P-D 分离 / 分布式 四份 A4 总结 | W4-W5 |
| vllm-ascend PR + 性能数据（项目 A） | W6-W7 |
| 消融实验报告（项目 C） | W7 |
| 白板四件套 + 项目讲稿 | W8 |

## 打卡规则建议

1. 每天结束打勾并写一句话收获（哪怕是"今天没搞懂 X，明天继续"）
2. 允许±1 天弹性，但复盘日不许跳过——复盘产出的图和总结才是面试时真正带得走的东西
3. 第 6-7 周如果 vllm-ascend 环境有客观阻碍，降级方案：在 GPU 上用 Triton 做项目 D（paged attention decode kernel + ncu roofline 分析），同样能讲"跨平台方法论"
