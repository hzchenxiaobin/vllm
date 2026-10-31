# 第 5 周：进阶专题（二）—— P/D 分离与分布式（Day 29-35）

> **本周目标**：掌握生产级推理系统的架构设计话题（专家岗区分度最高的部分）。
> **本周主线**：从"单实例内怎么跑得快"（W1-W4）升级到"一个集群怎么跑得好"——P/D 分离、并行策略、KV 路由与多租户。
> **本周产出物**：
> - [ ] A4 总结《P/D 分离》（Day 31）
> - [ ] A4 总结《分布式推理》（Day 33）
> - [ ] P/D 分离最小部署记录（Day 31）
> - [ ] TP1 vs TP2 + DP 对比实验数据（Day 33）
> - [ ] 设计题完整推演：日活千万客服机器人推理集群（Day 35）

---

## 0. 为什么这一周是你的"主场"（经验地图）

面试官面对你的简历，最想验证的是：**算子层的深度能不能升维到系统层的判断力**。本周两个专题恰好是两条经验主线的交汇点：

| 你的经验 | 推理系统的对应物 | 面试叙事 |
|---|---|---|
| decode（M≤256）场景做 A/B 矩阵 L1 全载模板，prefill 大 M 场景做 ASW 流水 | 同一颗芯片上，prefill GEMM 计算密集、decode GEMM 访存密集，最优 tiling/模板完全相反 | "P/D 干扰的微观根源我在算子层见过：两类 GEMM 形态相反，混跑时谁也跑不到自己的最优点" |
| 无 Queue 手工流水线：SetFlag/WaitFlag 管理多级乒乓，首 tile 半载隐藏 MTE2 延迟 | P/D 分离的 KV 分层传输：逐层推送 + decode 侧流水接收，把 RDMA 传输藏进 prefill 计算 | "把掩盖搬运延迟的流水线思想从 L1/DDR 尺度放大到机间网络尺度" |
| L2/HBM 带宽与 Cube 算力的访存/计算 bound 分界模型 | GPU Roofline、chunk 大小的 compute-bound 交叉点、P/D 分离收益的定量论证 | "bound 建模方法论跨平台一致：先算理论下界，再谈优化空间" |
| 钉钉 400+ 台引擎服务器、RocketMQ 削峰填谷、Redis 多级缓存 | 推理集群的副本扩展、请求队列、cache-aware 路由、分层 KV 存储 | "推理集群 = 我做过的 OLTP 高并发架构 + KV cache 这个新的状态管理问题" |
| VPC Endpoint 私网连接 + 路由对账 | KV 传输链路、全局 KV 目录与元数据一致性 | "分布式系统的路由、对账、故障域隔离是通用功" |

**本周反复练习一句话**："P/D 分离本质上是把'算子的流水线设计'放大成'集群的架构设计'，把'cache 局部性优化'放大成'全局 KV 资产管理'。"

---

## Day 29：P/D 分离（一）—— 为什么

### 1.1 先把"干扰"算出来（定量，不背结论）

**第一步：两阶段的资源画像。**

| 维度 | prefill | decode |
|---|---|---|
| 负载形态 | 大量 token 一次过（几百~几千） | 每 step 每 seq 1 token |
| bound | compute-bound（GEMM 大 M，算力是瓶颈） | memory-bound（权重+KV 每 token 全读一遍，带宽是瓶颈） |
| 理想 batch | 越大越好（MFU 上升） | 越大越好（带宽摊薄），但受 KV 显存限制 |
| 理想 kernel 形态 | 大 M 大 K，走满 Cube/SM | M=1~256 的瘦 GEMM，L1 全载/权重驻留类模板 |
| 关键指标 | TTFT | TPOT/ITL |

**第二步：算 chunked prefill 也救不了的窗口（手算题，面试可直接画）。**

以 Llama-3-70B FP8、单卡 H100（HBM 3.35TB/s，FP8 dense 算力 ~1979 TFLOPS）为例：

- **decode 侧约束（ITL SLO）**：设 TPOT SLO = 80ms，混批时一个 step 的时间 ≈ chunk 计算时间 + decode 计算时间。40% MFU 下 prefill 速度 ≈ `0.4 × 1979e12 / (2 × 70e9)` ≈ 5654 tok/s，则 chunk ≤ 80ms × 5654 ≈ **452 tokens**（还没算 decode 部分，实际更小）。
- **prefill 侧约束（compute-bound 交叉点）**：GEMM 每 token 读一遍权重（FP8 即 70GB），计算量 2×70e9 FLOP/token。计算时间 = 访存时间的临界 M = `peak / (2 × BW)` = `1979e12 / (2 × 3.35e12)` ≈ **295 tokens**（BF16 下 ≈148）。
- **结论**：chunk 必须 ≥295 才不伤 prefill 效率，≤452 才不破 TPOT SLO——**窗口存在但极窄，且没有任何余量**；负载一波动就穿。BF16 更糟。这正是"调参调不出来的结构性矛盾"，也是 P/D 分离的根本动机。
- **推论（面试加分）**：交叉点 `peak/(2×BW)` 只与硬件有关、与模型无关——硬件越偏算力（H100→B200），这个矛盾越尖锐。顺带解释了为什么 GPU 越新、P/D 分离越流行。

**第三步：混跑干扰的三条路径（能分层说清）。**

1. **批同步排队**：iteration-level batching 下，混在一个 step 里的 decode token 必须等整个 step（含 chunk 的 kernel）结束——ITL 出现与 chunk 等长的尖刺。
2. **访存争抢**：prefill 大 GEMM 打满 HBM 带宽与 SM，decode 的访存密集 kernel 排队；即使分 step，prefill step 期间 decode 整体停摆。
3. **形态互斥**：混批张量形状随 chunk 波动，对 kernel 选择/CUDA Graph/算子模板都不友好——你在昇腾上给 decode 和 prefill 分别选不同模板，就是因为"一套模板通吃两类形态必然双输"。

### 1.2 SLO 视角：为什么分离后 goodput 可提升 1.5-3 倍

- **goodput 定义**（W1 复习）：满足 SLO（TTFT p99 < X 且 TPOT p99 < Y）的吞吐，单位 req/s 或 token/s。
- **耦合问题**： colocated 系统里，TTFT 与 TPOT 的最优配置互相打架：
  - 想压 TTFT → prefill 优先、大 chunk → TPOT 抖动；
  - 想稳 TPOT → 小 chunk / decode 优先 → prefill 排队、TTFT 涨。
  - 两个 SLO 挤在同一条调度队列上，**无论怎么调参都是折中**。
- **分离后的三个独立自由度**：
  1. prefill 实例：大 chunk、奔满 MFU，用队列长度控制 TTFT；
  2. decode 实例：稳定 batch + CUDA Graph，TPOT 平稳；
  3. **容量配比独立扩展**：prefill:decode 算力比按负载画像（输入/输出 token 比）配，而不是被锁死在 1:1。
- **论文口径**：DistServe（OSDI'24）报告 goodput 最高 ~4.5×（负载与 SLO 依赖；计划里 1.5-3× 是更保守的工程口径）。
- **诚实的另一面（面试官必追问）**：Sarathi-Serve（OSDI'24）证明在很多区间内，仅 chunked prefill（stall-free batching）就能拿到大部分收益。**P/D 分离的增量收益出现在：模型大（交叉点矛盾尖锐）、TPOT SLO 严格、负载重、要控成本的时候**。能主动说出"什么时候不需要分离"比只会说"分离好"更加分。

### 1.3 什么时候不值得做 P/D 分离（背下来）

- prompt 短、输出短（如分类/抽取）：prefill 占比小，干扰本来就少；
- 低负载/小模型：单卡放得下且 TPOT 余量大；
- 没有高带宽网络（跨节点 KV 传输反而成为新瓶颈，见 Day 30 手算）；
- prefix 复用率极高的负载：KV 在实例内复用价值大于搬运价值；
- 运维成本敏感：分离引入路由、KV 传输、双池容量管理三个新故障域。

### 产出与自测

- 产出：手算推导一页（chunk 窗口 + 干扰路径图），夹进 A4《P/D 分离》
- 自测（不看笔记）：
  1. 为什么 compute-bound 交叉点与模型大小无关？
  2. "混跑时 decode 的 ITL 尖刺长度约等于什么？"（答：当前 step 的 chunk 计算时间 + decode 时间）
  3. colocated 下"先 prefill 后 decode"和"decode 优先"分别牺牲什么 SLO？

---

## Day 30：P/D 分离（二）—— 怎么做

### 2.1 分离系统的三个子问题（先建框架再看系统）

1. **路由与调度**：谁决定这个请求在哪台 prefill、去哪台 decode？（请求拆两段，路由要先于执行）
2. **KV 传输**：prefill 算出的 KV 怎么搬到 decode？（本日重点）
3. **异构配置**：两类实例各自的最优 batch / chunk / 并行度 / 量化策略不同。

### 2.2 KV 传输：先算账，再看架构

**传输量手算（Llama-3-70B，GQA kv_heads=8，head_dim=128，L=80）：**

- KV/token（BF16）= `2 × 80 × 8 × 128 × 2B` = 320KB/token；FP8 减半 = 160KB
- 4K prompt：BF16 → 1.31GB，FP8 → 655MB
- 400Gbps RDMA（≈50GB/s 有效）：BF16 传 26ms，FP8 传 13ms
- 对照：该 4K prompt 在 TP8 上 prefill（40% MFU）约 91ms → **传输时间 < 计算时间，可以完全藏进流水线**

**分层流水（核心设计，对应你的"手工流水线掩盖 MTE2"经验）：**

- 朴素做法：prefill 全算完 → 整段 KV 一次性传 → decode 才开始。TTFT 额外 + 全量传输时间。
- 流水做法：**逐层（甚至逐 block）推送**——layer i 的 attention 算完即异步发出，与 layer i+1 的计算重叠；decode 侧收齐全部 L 层即可启动。
- 边界收益：理想情况下传输几乎零增量延迟（只剩最后一层的传输尾巴 + 元数据同步）。
- 块粒度：按 KV block（如 256 tokens）切块传，兼顾 RDMA 效率与流水粒度。

**Push vs Pull：**

- **prefill-push（producer 主动发）**：路由必须先于执行（router 或调度器先选好 decode 实例）；延迟最低，控制面重。
- **decode-pull（consumer 按需拉）**：控制面轻，但拉取串行化会暴露传输延迟；可用"元数据先行 + 预取"缓解。
- 同节点（NVLink/CUDA IPC）传输快，push/pull 差异小；跨节点才是设计难点。

**传输载体谱系：**

| 载体 | 典型带宽 | 适用 |
|---|---|---|
| 同节点 NVLink / IPC | 数百 GB/s | 同机 P/D 池，传输基本免费 |
| 跨节点 NCCL（borrow 现成集合通信） | 取决于网卡 | 快速原型（vLLM PyNcclConnector） |
| 专用 RDMA 传输引擎（Mooncake TransferEngine / NIXL） | 400Gbps 级，零拷贝 | 生产级，拓扑感知调度 |

### 2.3 三个生产系统对比（Mooncake / NVIDIA Dynamo / llm-d）

| 维度 | Mooncake | NVIDIA Dynamo | llm-d |
|---|---|---|---|
| 出身 | 月之暗面 Kimi 生产系统（FAST'25 论文） | NVIDIA 开源推理框架 | 社区 + Red Hat 等，CNCF 沙箱项目 |
| 核心理念 | **KVCache-centric**：全局 KV 池（DRAM+SSD+远端）为中心，算力围绕 KV 转 | **分层调度框架**：router + KV-aware scheduler + 传输层（NIXL），后端可插 vLLM/SGLang/TRT-LLM | **K8s 原生**：以标准 K8s 组件（Gateway API、Deployment、Envoy）拼出分布式 vLLM |
| P/D 分离方式 | prefill/decode 分离 + KV 存储分层 | prefill/decode worker 分池，KV 经 NIXL 搬运 | PrefillPool/DecodePool 分池 + KV-aware 路由打分 |
| 路由 | KV 亲和（cache 命中优先） | KV-aware + goodput-aware router | PrefixCacheScorer（前缀匹配打分）+ 负载 |
| 特色 | TransferEngine（RDMA 零拷贝、拓扑感知）；论文口径：同 SLO 下吞吐提升最高 ~75%（Kimi 生产负载） | 组件化：可只用其 router/传输层；号称路由开销亚毫秒 | 充分复用 K8s 生态（HPA、多租户、网关）；轻量侵入 |
| 取舍 | 深度绑定存储思维，运维复杂 | 组件多、概念多 | 依赖 K8s 成熟度；单机极致性能非首要目标 |

**读法建议**：先读 Mooncake 论文的架构图与动机部分（最好懂、最成体系），再用 Dynamo 的 compose 部署建立体感，最后浏览 llm-d 的设计文档看 K8s 生态怎么接入。**面试时按"路由 / KV 传输 / 容量配比"三栏去拆任何一个系统**，就不会乱。

**论文谱系（一句话版本，防止混）：**

- DistServe（OSDI'24）：P/D 分离 + 按 SLO 独立配比容量的系统化论证；
- Splitwise（ISCA'24）：按 phase 划分资源池 + KV 分层传输（微软）；
- Sarathi-Serve（OSDI'24）：不分离，用 chunked prefill + stall-free batching 折中解决干扰；
- Mooncake（FAST'25）：从"分离"进化到"KV cache 全局资产化"。

### 2.4 vLLM 里的落点（源码/配置地图）

- **KVConnector 抽象**（V1）：`vllm/v1/kv_connector_interface.py`（`KVConnectorBase_V1`）——把"KV 从哪来/到哪去"从执行路径解耦出来；现有实现：PyNcclConnector（跨节点原型）、SharedStorageConnector、LMCacheConnector、MultiConnector 等。
- 启动参数：`--kv-transfer-config`（JSON：role、connector、rdma 等字段，**字段名随版本演进，以当版示例脚本为准**）。
- 示例位置：仓库 `examples/online_serving/disaggregated_prefill.py`（NCCL 路线，单机双实例 + 简易 router 的最小可跑示例）与 `examples/other/` 下的 LMCache 路线脚本。
- 思考题（对接项目 A）：vllm-ascend 的 P/D 分离仍在快速演进——昇腾侧 HCCL/HCCS 与主机侧 DDR/网卡拓扑下，KVConnector 的传输实现有哪些可切入点？

### 产出与自测

- 产出：一页《KV 传输账本》（传输量公式 + 分层流水示意图 + push/pull 对比），并入 A4
- 自测：
  1. 4K prompt、70B、FP8 KV、200Gbps 网卡（≈25GB/s）：传输要多久？还能藏进流水线吗？（26ms→52ms，仍 < 91ms 计算，勉强能）
  2. 为什么 layer-wise 推送后 TTFT 增量接近零？剩下的尾巴是什么？
  3. 三个系统各用一句话说清差异。

---

## Day 31：P/D 分离实验（动手日）

### 3.1 最小可行部署（路线 A：vLLM 官方示例）

```bash
# 1. 找到当前版本的示例（路径随版本变化）
ls examples/**/disaggregated* examples/online_serving/

# 2. 示例脚本通常拉起三个进程：prefill 实例 + decode 实例 + 简易 router
#    思路（字段以当版脚本为准）：
# prefill 实例
vllm serve Qwen/Qwen3-8B --port 8001 \
  --kv-transfer-config '{"kv_role":"producer","kv_connector":"PyNcclConnector"}'
# decode 实例
vllm serve Qwen/Qwen3-8B --port 8002 \
  --kv-transfer-config '{"kv_role":"consumer","kv_connector":"PyNcclConnector"}'
# 3. 压测打 router 端口，不是打 8001/8002
vllm bench serve --backend vllm --model Qwen/Qwen3-8B \
  --dataset-name sharegpt --dataset-path <路径> \
  --request-rate 4 --percentile-metrics ttft,tpot,itl
```

### 3.2 对照实验设计（有对照才有结论）

三组部署，同负载各跑一遍：

| 组 | 部署 | 观察 |
|---|---|---|
| A（基线） | 单实例 colocated（默认 chunked prefill） | TTFT / TPOT 基线 |
| B（分离） | prefill 实例 + decode 实例 | TTFT 应≈基线或更好；TPOT 尾部应更稳（p99 改善最明显） |
| C（可选） | B + 调大 prefill 侧 chunk | 看 TTFT 进一步下降而 decode 不受影响——这是分离的独有自由度 |

**重点看 p99 而不是均值**：分离的收益集中在尾部稳定性（TPOT p99），均值可能只差几个百分点——这本身就是一条面试结论。

### 3.3 常见坑（提前列出来省半天）

- 两实例的模型、版本、dtype 必须完全一致；
- `kv_role` / `kv_connector` 字段名随版本变化，**以示例脚本为唯一真源**，报错先 diff 示例；
- NCCL 初始化 hang：多机/多网卡环境设置 `NCCL_SOCKET_IFNAME`；
- 压测必须打到 router；打到 prefill 端口会得到"只有 TTFT 没有 decode"的假数据；
- 当前限制要心里有数：KV 单向流动（prefill→decode）、与 prefix caching / LoRA / 多模态的组合支持仍在演进（版本相关）——面试时能主动说出限制，比假装完美更加分。

### 3.4 A4 总结模板《P/D 分离》（四段式，一页纸）

1. **原理**：两阶段资源画像相反 → 混批干扰三条路径 + chunk 窗口手算（贴 Day 29 推导）；
2. **方案**：三个子问题（路由/传输/配比）+ 分层流水 + push/pull + 载体谱系；
3. **权衡**：收益条件（大模型/严 SLO/重负载/控成本）vs 不值得做的五种情况 + 新增故障域；
4. **失效模式**：KV 传输成新瓶颈（网络不够）、容量配比失衡（prefill 排队或 decode 闲置）、路由与执行不匹配（选错 decode 实例）、低命中负载白搬 KV。

### 产出

- [ ] 部署记录（三组对照的 TTFT/TPOT p99 数据表 + 一段结论）
- [ ] A4《P/D 分离》v1

---

## Day 32：分布式并行（一）—— 原理

### 4.1 TP（Tensor Parallel）

- **切法**：attention 按 head 切（o_proj 行并行），MLP gate/up 列切 + down 行切 → 每层前向 **2 次 all-reduce**（推理只有前向，无反向通信）。
- **通信量公式（背）**：decode 每 step 每 seq 的 AR 消息量 = `2 × L × hidden × dtype_bytes`；
  ring all-reduce 实际搬运 = `2(N-1)/N ×` 消息量。
  例：Llama-3-70B BF16，`2 × 80 × 8192 × 2B` ≈ 2.6MB/token；batch 256 → 671MB/step；TP8 ring 系数 1.75 → 每 GPU 搬 ~1.17GB/step。
- **带宽需求**：NVLink 域内（H100 NVSwitch，数百 GB/s 有效 busbw）→ ~2-3ms，占 decode step（约 50ms）的 ~5%，可接受；**跨节点 PCIe/IB（几十 GB/s 且多卡共享）→ 通信占比暴涨 + 每层同步锁步**。
- **为什么"能单卡放下就别上 TP"**：
  1. 跨 NVLink 域后每层 AR 延迟叠加，decode TPOT 下降不线性；
  2. 所有 rank 锁步执行，最慢的卡拖垮全局（木桶效应）；
  3. TP 的收益（显存分摊）用"换更小的卡/量化"往往也能拿到；
  4. 副本（DP）扩展无通信、故障域小、运维简单——**吞吐优先用 DP，延迟才用域内 TP**。
- **TP 的理想收益**：decode 是访存 bound → 权重分摊到 N 卡 → 理论 TPOT ≈ 1/N（域内）；prefill 计算分摊 → TTFT ≈ 1/N；**跨机后两者都迅速劣化**。

### 4.2 PP（Pipeline Parallel）

- 用途：模型大到 TP 域内都放不下时才考虑；推理 PP 的"micro-batch"就是 continuous batching 里的并发请求流。
- **bubble 公式（背）**：`bubble 占比 = (p-1)/(m+p-1)`（p 段，m 个 micro-batch）。
  例：p=4, m=8 → 27%；m=64 → 4.5%。**推理请求数天然是流水填充器，高并发下 bubble 可压小；低并发下 PP 很亏**。
- 推理 vs 训练的差异：推理无反向、无 activation 重算，PP 的工程难点在调度（变长请求跨段路由）与 KV 管理（每段各自持有本段 KV）。
- 注意：vLLM V1 的 PP 支持在快速演进，使用前查当版 release notes / 文档。

### 4.3 EP（Expert Parallel，MoE 专用）

- 切法：expert 分片到各卡，token 按 router 选 top-k 专家 → **dispatch（all-to-all）→ expert 计算 → combine（all-to-all）**，每 MoE 层两次 a2a。
- 通信特征与 TP 的 AR 不同：a2a 的量 ≈ `batch × hidden × topk/EP` 量级（每个 token 发往 k 个专家所在卡），**对 batch 大小极其敏感**——小 batch 时 a2a 延迟占比失控，必须靠大并发摊平。
- DeepSeek 路线的启示：MLA 压缩 KV + 大 EP + DP attention（KV 不随 TP 复制），是"为 MoE 推理重新设计并行拓扑"的范本；DeepEP 提供正常/低时延两种 a2a 模式分别服务 prefill/decode。
- vLLM 入口：`--enable-expert-parallel`。

### 4.4 选型决策树（面试画这个）

```
模型放得进单卡？
├─ 是 → 不要任何模型切分，用 DP 副本横向扩（吞吐）/ 域内 TP（单请求延迟）
└─ 否 → MoE？
    ├─ 是 → EP（+ DP attention），域内 TP 补显存
    └─ 否 → 先量化/量化 KV 压显存 → 仍放不下：
        ├─ 能接受吞吐损失 → 跨机 TP（延迟敏感）
        └─ 高并发 → PP（bubble 可摊薄）
```

**与昇腾对照**（简历对接点）：把 NVLink/NVSwitch ↔ HCCS 总线、NCCL ↔ HCCL、NVSwitch 全互联 ↔ 昇腾集群拓扑逐项映射写成一页小表；面试讲"我在昇腾上理解的是'通信带宽 ÷ 计算密度'决定并行方案，这在 GPU 上完全同构"。

### 产出与自测

- 产出：决策树 + 通信量公式卡 + 昇腾/GPU 对照表（并入 A4《分布式推理》）
- 自测：
  1. 70B BF16 TP8 batch 256，一步 decode 的每 GPU 通信量？（答：~1.17GB，能现场推）
  2. TP 什么时候负收益？（跨 NVLink 域 / 小 batch 下 AR 延迟占比高 / 锁步木桶）
  3. PP 在推理里为什么高并发才划算？

---

## Day 33：分布式并行（二）—— 实验

### 5.1 实验设计：2 张卡的三种用法（核心洞察实验）

同负载（固定 dataset 与 request-rate 扫档）跑三组：

| 组 | 部署 | 预期 |
|---|---|---|
| 1 | 单卡 TP1 × 1 实例 | 基线 |
| 2 | TP2 × 1 实例 | TPOT 更低（权重分摊带宽）、单请求更快；总吞吐 ↑ 但 < 2× |
| 3 | TP1 × 2 实例 + router | **总吞吐最高（无通信）**；但单请求 TPOT 不变 |

**这个实验的结论就是 Day 52 面试题"TP 开到什么时候是负收益"的实证版**：同样的 2 张卡，"更快"和"更多"是两种花法——延迟选 TP，吞吐选副本。

```bash
# 组2
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 2 --port 8000
# 组3：两个实例 + 简易转发（round-robin 即可，或用 vllm-project/production-stack 的 router）
vllm serve Qwen/Qwen3-8B --port 8001 & vllm serve Qwen/Qwen3-8B --port 8002 &

# 压测（扫并发）
vllm bench serve --backend vllm --model Qwen/Qwen3-8B \
  --dataset-name sharegpt --dataset-path <路径> \
  --request-rate 1,2,4,8,16 --percentile-metrics ttft,tpot,itl
```

### 5.2 用 nsys 看 NCCL 占比

```bash
# 包住服务进程，延迟 60s（等预热完）采 60s
nsys profile -o tp2_trace --trace=cuda,nvtx,osrt --delay 60 --duration 60 \
  vllm serve Qwen/Qwen3-8B --tensor-parallel-size 2

# 统计 NCCL kernel 占比
nsys stats -r cuda_gpu_kern_sum tp2_trace.nsys-rep | grep -i nccl
```

- 找 `ncclDevKernel*` 的合计时间 ÷ step 总时间 = 通信占比；域内预期 <5%。
- 轻量替代：vLLM 自带 profiler——设 `VLLM_TORCH_PROFILER_OUTPUT_DIR=/tmp/prof` 启动，压测中 `curl :8000/start_profile` → 睡 30s → `/stop_profile`，用 chrome://tracing 打开看 NCCL 段。
- 进阶观察：TP2 下 decode step 时间线里 AR 与 GEMM 是否有间隙（同步锁步的 bubble）；对比 TP1 的 kernel 形态差异。

### 5.3 A4 总结模板《分布式推理》

1. **原理**：TP/PP/EP 各自的切法、通信原语与通信量公式；NVLink 域的物理含义；
2. **场景**：决策树 + "单卡放得下就别 TP"的四个理由；
3. **权衡**：TP（延迟/域内）vs DP 副本（吞吐/扩展性）vs PP（超大模型/高并发）vs EP（MoE）；跨域通信代价；
4. **失效模式**：跨机 TP 通信放大、小 batch 下 EP a2a 失控、低并发 PP bubble、锁步木桶效应、（附）实验数据表佐证。

### 产出

- [ ] 三组对比数据表（并发 × {TTFT, TPOT, 吞吐}）+ nsys 通信占比截图
- [ ] A4《分布式推理》v1

---

## Day 34：prefix caching 与路由（集群视角）

### 6.1 从"实例内命中"到"全局命中"（问题升级）

W3 学的是单实例 block hash 命中；集群里出现新问题：**相同前缀的请求被 router 均衡到了没有缓存的实例** → 全局命中率塌方。这就是 cache-aware routing 要解的问题。

### 6.2 路由方案谱系（按复杂度递进）

| 方案 | 思路 | 问题 |
|---|---|---|
| round-robin / least-loaded | 只看负载 | 命中率≈0（前缀随机分布） |
| 会话亲和（session affinity） | 同一会话粘同一实例 | 实现最简单，客服/多轮场景性价比极高 |
| 前缀一致性哈希 | 对 prompt 前 k 个 token 的 digest 做一致性哈希 | 简单有效；热点前缀会打爆单实例（需负载兜底） |
| 前缀匹配打分 | router 维护各实例"最近 block hash 集合"（bloom filter / Redis），选最长匹配 | production-stack 与 llm-d（PrefixCacheScorer）的做法；精度/开销权衡 |
| 全局 KV 目录 + KV 可迁移 | "请求找缓存"变"缓存可搬迁"（KV 搬到有空闲的实例） | Mooncake 的终极形态；控制面最重 |

- 现成参考：vllm-project/production-stack（Helm 部署 + prefix-aware router + 监控全家桶）、SGLang router（cache-aware 变体，"leftovers" 思路：先按前缀匹配，落选者再按负载兜底）、llm-d（K8s 原生打分器）。
- **对账思维（对接你的 VPC 经验）**：router 记录的"实例↔前缀"视图是异步近似，与实例真实状态存在漂移——需要 TTL、容量上限、定期对账/降级，这就是你做过的"路由对账"在 KV 域的翻版。

### 6.3 多租户：cache_salt

- 机制：vLLM API 传入 `cache_salt`（如 `extra_body={"cache_salt": "tenant-42"}`），salt 的 hash 参与每层 block hash → **不同租户相同 prompt 物理隔离**，防跨租户缓存泄漏（合规硬要求）。
- 代价：租户间不共享 → 公共 system prompt 无法复用，全局命中率下降——**安全与效率的正交权衡**，面试能讲清这一条就很出彩。

### 6.4 分层 KV：GPU → CPU → SSD → 远端

- 动机：KV 是"算出来的资产"，evict 丢弃 = 白白烧掉 prefill 算力；下沉到便宜介质，命中时回迁远比重算便宜（前提：读带宽 > 等效重算速度）。
- 成本账（一算就有说服力）：命中 4K token 的 70B FP8 KV = 省下 `2 × 70e9 × 4096 / (0.4 × 1979e12)` ≈ 725ms 单卡 prefill 算力（TP8 ~91ms），而 655MB 从 SSD（10GB/s）回读只要 ~65ms。
- 生态：LMCache（CPU/SSD/远端，vLLM KVConnector 集成）、Mooncake 的全局 KV 存储（DRAM+SSD+对象存储分层）、vLLM V1 内建的 hybrid KV cache manager（CPU offload 与 prefix cache 联动，full block 换出）。
- 分层策略要点：热的（频繁命中）留 HBM，温的驻 DRAM，冷的下沉 SSD/远端；命中率 × 回迁带宽决定分层是否赚钱。

### 6.5 动手实验（半天）

1. 构造高重复前缀负载：ShareGPT 改造或自造 dataset——同一 system prompt + 递增多轮历史（模拟客服会话）；
2. 对比开关 `--enable-prefix-caching` 的 TTFT 分布；观察 `/metrics` 中 `vllm:gpu_prefix_cache_hits` / `vllm:gpu_prefix_cache_queries` 命中率；
3. 开 salt 再压一遍（`cache_salt` 每请求随机），验证命中率归零、TTFT 回落——亲手复现 6.3 的权衡。

### 产出与自测

- 产出：命中率 vs TTFT 曲线 + salt 对照数据（并入 A4《分布式推理》"集群扩展"小节）
- 自测：
  1. 会话亲和 vs 前缀打分，各自的失效场景？
  2. cache_salt 防的是什么攻击/事故？（答：跨租户 KV 泄漏——共享前缀下 B 租户可"预热"A 租户的缓存）
  3. 什么时候 KV 下沉到 SSD 反而亏？（回迁带宽 < 等效重算速度，或命中率低）

---

## Day 35：复盘日 —— 4 份 A4 互讲 + 集群设计题

### 7.1 上午：A4 互讲

- 把 W4-W5 的 4 份 A4（量化 / 投机解码 / P/D 分离 / 分布式推理）各讲 5 分钟，录音回听；
- 每份按"原理 → 场景 → 权衡 → 失效模式"四段互查，缺一段当场补；
- 检查每个专题是否都有一道**手算锚点**（量化：KV 显存；投机：加速比公式；P/D：chunk 窗口；分布式：AR 通信量）。

### 7.2 下午：设计题完整推演（白板模拟）

**题目：为一个日活千万的客服机器人设计推理集群架构。**

**Step 1 需求量化（先算再画，面试官就看这个）：**

- 假设：DAU 10M，人均 2 会话/天，每会话 6 轮；峰值小时占全天 20%，分钟级再 ×2 安全系数
- 峰值 RPS ≈ `10M × 2 × 0.2 / 3600 × 2` ≈ **2200 req/s**（每轮 = 1 请求）
- 画像：RAG 客服 → 输入长（system prompt + 检索文档 + 多轮历史，均值 ~1.5K token）、输出短（~150 token）
- SLO：TTFT p99 < 1.5s，TPOT p99 < 80ms（流式可感知）
- 并发（Little 定律）：驻留时间 ≈ TTFT + 150 × 60ms ≈ 9.5s → 并发 decode 流 ≈ 2200 × 9.5 ≈ **21K 路**

**Step 2 模型与并行选型（用 Day 32 决策树）：**

- 选 ~32B 级 instruct 模型 FP8：权重 ~33GB → **单卡 80G 放得下 → 不上 TP**（讲清这四个字的理由是加分项）
- KV/token（FP8 KV，L=64, kv_heads=8, head_dim=128）= `2×64×8×128×1B` = 128KB；均长 1.65K → ~210MB/请求
- 单卡并发上限：KV 预算 ~40GB → ~180 路；decode TPOT 估算：权重 33GB/3.35TB/s ≈ 10ms + KV 读 ~12ms → ~40ms，满足 SLO 有余量

**Step 3 容量估算：**

- decode：21K 路 ÷ 180 路/卡 ≈ **~117 卡**（先按无前缀复用算，作为上界）
- prefill：2200 × 1500 tok/s = 3.3M tok/s；单卡 @40% MFU ≈ 12.4K tok/s → 267 卡？——**这一步就是转机**：多轮历史 + 共享 system prompt 使 prefix caching 命中率 60-70%，新增 token 降到 ~500/请求 → prefill 需求砍到 ~90 卡
- 结论链条（面试亮点）：**prefix caching + cache-aware 路由把集群规模砍掉一半以上** → 这直接论证了 Day 34 为什么是独立专题

**Step 4 架构分层（画图）：**

```
客户端 → 网关（限流/鉴权） → router（会话亲和 + 前缀打分 + 负载兜底）
                                   ├─ prefill 池（FP8、大 chunk、奔 MFU）
                                   │      └─(KV 逐层推送)─┐
                                   └─ decode 池（稳定 batch + CUDA Graph）←┘
        全局：分层 KV（HBM→DRAM→SSD）· cache_salt 多租户 · Prometheus/Grafana（goodput 看板）
```

- P/D 要不要分离？——按 Day 29 判据：TPOT SLO 严 + prefill/decode 算力比 1:1.3 失衡 + 规模够大 → **分离或动态配比**，并保留 colocated 弹性缩容预案（夜间负载低谷合并池）
- 弹性：按 goodput（非 raw throughput）做 HPA；峰值分钟级突增靠队列削峰（对接你 RocketMQ 削峰经验）
- 兜底与降级：小模型分级路由（简单意图走 8B）、KV 池满时的 preemption 监控（`vllm:preemption_reqs` 类指标）、多可用区容灾
- 成本三板斧收尾：FP8 权重 + FP8 KV + prefix caching（每个都给出上面算过的量化收益）

**Step 3 分钟版讲稿骨架**：需求量化（1min）→ 选型决策（1min，突出"为什么不上 TP"）→ 容量与架构（1min，突出 cache 权衡与弹性）。

### 7.3 本周自测清单（全部口头过一遍）

1. P/D 分离的定量动机是什么？chunk 窗口怎么算？
2. 什么时候不值得做 P/D 分离？（背 5 条）
3. KV 逐层推送为什么能把传输延迟藏掉？push 和 pull 差在哪？
4. Mooncake / Dynamo / llm-d 一句话对比？
5. 70B BF16 TP8 batch 256 每 GPU 每 step 通信量现场推？
6. TP 什么时候负收益？PP 为什么高并发才划算？EP 的 a2a 为什么怕小 batch？
7. 2 张卡：TP2 快还是 2 副本多？（答：延迟选 TP、吞吐选副本，用 Day 33 数据背书）
8. cache_salt 防什么？代价是什么？
9. KV 下沉 SSD 什么时候反而亏？
10. 设计题 3 分钟版讲一遍并录音。

---

## 附录 A：本周高频面试题速查（对应 Day 52 过堂清单）

| 问题 | 答题锚点 |
|---|---|
| P/D 分离什么时候不值得做？ | 短输入短输出 / 低负载小模型 / 无高带宽网络 / 高前缀复用 / 运维成本敏感 |
| TP 开到什么时候是负收益？ | 跨 NVLink 域；小 batch；锁步木桶；有更便宜的显存手段（量化）时 |
| 为什么"能单卡放下就别上 TP"？ | 副本无通信、故障域小、扩展线性；TP 只有延迟收益且仅限域内 |
| 混跑时 decode 的 ITL 尖刺由什么决定？ | 当前 step 的 chunk 计算时间（批同步 + 访存争抢） |
| 分离后 goodput 为什么能提升 1.5-3×？ | 双 SLO 解耦 + 容量配比独立 + 各自最优 batch/配置 |
| KV 传输量怎么估？ | `2 × L × kv_heads × head_dim × bytes × prompt_len`，先算账再谈架构 |

## 附录 B：参考资料

**论文**
- DistServe（OSDI'24）：<https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>
- Sarathi-Serve（OSDI'24）：<https://www.usenix.org/conference/osdi24/presentation/agrawal>
- Splitwise（ISCA'24）：<https://dl.acm.org/doi/10.1145/3620666.3651335>
- Mooncake（FAST'25）：<https://www.usenix.org/conference/fast25/presentation/qin>

**系统与代码**
- Mooncake：<https://github.com/kvcache-ai/Mooncake>
- NVIDIA Dynamo：<https://github.com/ai-dynamo/dynamo>
- llm-d：<https://github.com/llm-d/llm-d>
- vLLM disaggregated 示例：<https://github.com/vllm-project/vllm/tree/main/examples/online_serving>
- vLLM production-stack（prefix-aware router）：<https://github.com/vllm-project/production-stack>
- LMCache：<https://github.com/LMCache/LMCache>
- DeepEP（MoE a2a）：<https://github.com/deepseek-ai/DeepEP>

---

## 打卡记录

| Day | 日期 | 完成打勾 | 一句话收获 |
|---|---|---|---|
| 29 | | [ ] | |
| 30 | | [ ] | |
| 31 | | [ ] | |
| 32 | | [ ] | |
| 33 | | [ ] | |
| 34 | | [ ] | |
| 35 | | [ ] | |

> 复盘日不许跳过——本周的 4 份 A4 与设计题推演，是面试"专家岗区分度"环节真正带得走的东西。
