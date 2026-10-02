# 第 0 周：推理系统优化专家岗 · 7 天冲刺计划

> **适用场景**：距面试仅 1 周
> **核心策略**：放弃"全面深入学习"，改为"面试产出最大化"——不追源码细节、不做大实验，把昇腾算子经验翻译成推理系统语言 + 补齐调度层必考知识 + 打磨表达
> **每天投入**：6-8 小时（全职冲刺）

---

## 冲刺原则（先读这个）

1. **砍掉的**：vllm-ascend PR（来不及）、消融实验报告、分布式深挖、CUDA kernel 手写——这些只留概念，能讲思路即可
2. **保留的**：显存/时延手算、scheduler 与 KV cache 管理机制、白板四件套、项目讲稿
3. **杠杆点**：你做过量化 GEMM 优化——面试时每个话题都试着挂上这条经验，把"我在学 vLLM"讲成"我在另一个硬件上做过同构问题"

---

## Day 1：推理基础速通（建立共同语言）

- [ ] 上午：prefill（compute-bound）vs decode（memory-bound）的本质；KV cache 生成与复用机制
- [ ] 下午：**必考手算**（练到 3 分钟内能默推）：

- KV cache 每 token 显存 = `2 × layers × kv_heads × head_dim × dtype_bytes`
- decode 单 token 时延下界 ≈ 模型字节数 / HBM 带宽
- 练习题：70B FP8 模型、H100、128K 上下文，能开多少并发？

- [ ] 晚上：PagedAttention 论文只读 §1/§3/§4（动机 + block 设计 + 调度），其余跳过
- [ ] **产出**：一页纸《推理性能第一性原理》（含 3 道手算题推导）

## Day 2：vLLM V1 架构 + 跑起来压测

- [ ] 上午：读 V1 架构官方博客/文档，掌握进程结构（AsyncLLM → Processor → EngineCore → ModelRunner）和请求生命周期
- [ ] 下午：源码只读两个文件（不求全懂，抓住主干）：

- `vllm/v1/core/scheduler.py`：waiting/running 队列、token budget、chunked prefill 切块、preemption 触发
- `vllm/v1/core/kv_cache_manager.py`：block 分配/释放、prefix caching 的 block hash

- [ ] 晚上：起 Qwen3-8B 服务，跑 `vllm bench serve`，看 TTFT/TPOT/吞吐随并发的曲线（2 小时内必须跑完，不纠结环境细节，租卡解决）
- [ ] **产出**：一张手绘 V1 数据流图 + 第一组压测数据

## Day 3：核心机制四连（面试主战场）

每个机制 1.5-2 小时，按"原理 → 解决什么问题 → trade-off → 什么时候失效"四段式过：

- [ ] **Continuous batching**：iteration 级调度 vs static batching；为什么吞吐能提升数倍
- [ ] **Chunked prefill**：长 prompt 切块与 decode 混排；收益（TPOT 不抖动）与代价（prefill 变慢、调度复杂）
- [ ] **Prefix caching**：block hash + 引用计数 + COW；命中率驱动的收益；多租户隔离（cache_salt）
- [ ] **CUDA Graph**：为什么 decode 必须抓图（kernel launch 开销 vs decode 单步时延）；piecewise CG 解决什么问题
- [ ] 晚上：压测验证——开/关 chunked prefill 看 TPOT 抖动差异（一组实验即可）
- [ ] **产出**：四张"四段式"卡片（每张不超过半页）

## Day 4：专题速通（只读不做）

每个专题读 1-2 篇高质量综述/官方文档，整理一页 A4：

- [ ] **量化**（1.5h，重点，直接对接你的经验）：W8A8/W4A16/FP8 的区别；KV cache 量化的收益与精度代价

- ⚡ 准备一段对照叙述：昇腾 WeightQuantBatchMatmul 的 tiling 搜优 ↔ GPU 上量化 GEMM 的 bound 分析，方法论同构

- [ ] **Speculative decoding**（1.5h）：MTP / EAGLE / draft model 三路线；接受率决定收益；低接受率负收益的失效模式
- [ ] **P/D 分离**（2h）：为什么分（prefill/decode 硬件特性相反、SLO 互相干扰）；同节点 NVLink vs 跨节点 RDMA；知道 Mooncake/Dynamo 的名字和大致设计即可
- [ ] **分布式并行**（1h）：TP 通信开销、"能单卡放下就别上 TP"、EP 用于 MoE——概念级即可
- [ ] **产出**：4 页 A4 专题速查卡

## Day 5：动手压缩版（二选一，只花一天）

**选项 1（推荐，白板利器）**：150-300 行 Python 写 mini 调度器

- 固定 block 的 KV 池 + block table + 引用计数 + iteration 级 continuous batching
- 不做 chunked prefill、不接真模型，用假 token 流演示调度行为即可
- **产出**：能在白板画出它的架构，并讲"vLLM 在 X 处比这复杂得多，因为……"

**选项 2（如果时间紧）**：读不写

- 精读 vllm-ascend 仓库的 attention backend 实现，整理"新硬件接入 vLLM 要实现哪些接口"
- **产出**：一页《vLLM 硬件后端接入指南》笔记——面试差异化亮点

## Day 6：白板四件套 + 高频题过堂

- [ ] **白板四件套**（各练 3 遍，计时）：

1. 显存估算（任意模型/精度/上下文长度）
2. 手绘 block table（含 prefix caching 命中、COW 分裂）
3. 调度推演（10 个请求、KV 只够 6 个，逐步推演 running batch 与抢占恢复）
4. 性能诊断树（TTFT↑且ITL稳→队列/prefill拥塞；ITL↑→batch过大或通信；preemption↑→KV超配调 `max_num_seqs`）

- [ ] **高频题录音自答**（每题 3 分钟，说完回听找问题）：

1. PagedAttention 解决了什么问题？碎片率怎么算？
2. continuous batching vs static batching？
3. 为什么 decode 用 CUDA Graph 而 prefill 不用？
4. chunked prefill 的 trade-off？token budget 怎么设？
5. 量化对 TTFT 和 TPOT 分别什么影响？
6. P/D 分离什么时候不值得做？
7. TP 什么时候是负收益？
8. TTFT 高怎么排查？TPOT 高怎么排查？

- [ ] **产出**：自己的答题录音 + 修正稿

## Day 7：模拟面试 + 收尾

- [ ] 上午：完整模拟面试 ×1（找同行或用模拟面试工具），重点练追问下的 trade-off 表达
- [ ] 下午：打磨叙事主线（3 分钟版 + 10 分钟版各讲一遍）：

> "我在昇腾做量化 GEMM kernel 优化，单核算力利用率 85%+，方法是 tiling 搜优 + 访存/计算 bound 建模 + 手工流水线。LLM 推理优化是同一套方法论的放大：KV cache 是内存问题，continuous batching 是调度问题，P/D 分离是把 compute-bound 和 memory-bound 负载拆开各自优化。"

- [ ] 晚上：只看自己这一周写的卡片和总结，**不看任何新资料**；早睡
- [ ] **产出**：最终面试作战包 = 1 页第一性原理 + V1 数据流图 + 4 张机制卡 + 4 页专题卡 + 白板四件套 + 项目讲稿

---

## 一周冲刺 vs 八周计划的取舍对照

| 内容 | 八周版 | 七天版 |
| --- | --- | --- |
| 显存/时延手算 | 完整推导 | ✅ 保留（必考） |
| vLLM 源码 | 全链路精读 | 只读 scheduler + kv_cache_manager |
| 压测实验 | 系统性消融 | 只跑 2 组关键对比 |
| 量化专题 | 动手量化 | ✅ 保留（挂你的昇腾经验） |
| 投机解码/P-D 分离 | 实验 + 部署 | 只读概念 |
| 分布式 | 双卡实验 | 概念级 |
| 项目 A（PR 贡献） | 完整执行 | ❌ 砍掉 |
| 项目 B（mini 引擎） | 完整实现 | 150 行核心调度器 |
| 白板四件套 + 模拟面试 | 1 周 | ✅ 保留 2 天（性价比最高） |

## 风险提示

- 一周方案能让你**通过初中级深度的问题 + 在系统设计题上出彩**，但遇到"读过 vLLM 某模块源码细节"的深追会露怯——被问到没读过的部分，诚实说"还没读到这层，但我的理解是……"然后给推理，比硬编强得多
- 如果面试确认通过、后续还有二面/三面，立刻切回八周计划的第 6-7 周（项目 A），把短板补上