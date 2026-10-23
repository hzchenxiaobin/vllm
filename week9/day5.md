# Day 5：分布式并行 —— Attention DP / balanced CP、MoE EP + EPLB

> **Week 9 · DeepInfer 源码精读 · Day 5**
> 前置知识：Day 2（进程模型与 gloo CPU 组）、Day 3（CP-mirror 字段与 metadata 广播）、Day 4（MoE 三路径与 EPLB 的 gate 接口）；vLLM 的 TP/PP/DP（Week 2）、PD 分离（Week 5）。
> 源码基线：commit `9bd9cf2`（2026-10-21）。今日主角：`parallel_utils/`（5 文件 706 行）、`eplb/`（3 文件 792 行）、`tilelang_ops/prepare_cp_mirror.py`（394 行）、`compute_seqlens_k_zos.py`（121 行）、`deep_cluster/topology/`。
> 今日用时：4~5 小时（源码走读 3h + 手算 1h + 自测 0.5h）

---

## 今日学习目标

| # | 目标 | 验收标准 |
|---|---|---|
| 1 | 讲清 Attention DP 的本质 | 对照 vLLM 的 TP/PP/DP，解释"为什么 MLA 适合 attention DP"、DP 组内为什么必须步型一致 |
| 2 | 推导 balanced CP 的均衡性 | 手排 zigzag 例子（8 卡、序列长 56），证明每 rank 的 Σ(query 位置) 恒等；说出"balanced"的三层含义 |
| 3 | 讲清 EPLB 全链路 | 统计（窗口/计数）→ 策略（两阶段贪心）→ 迁移（两种 shuffle）→ 热切换（gate 吃映射表），触发条件与迁移粒度脱口而出 |
| 4 | 手算 EP 切分 | 给定 64 卡 decode 超节点，推出 physical expert 总数与每卡分布（对照 Q&A 的"每卡 7 个"） |
| 5 | 产出并行策略矩阵图 | DP / CP / EP / ENP × 切什么维度 × 通信原语 × 代码文件，一页纸 |

今日阅读清单（对应 week9 README）：

| 文件 | 行数 | 主题 |
|---|---|---|
| `examples/offline_inference.py --ep/--dp/--cp` + `examples/README.md` | — | 参数语义（Day 2 已读，今日复盘） |
| `parallel_utils/parallel_state.py` | 230 | 并行组建立 |
| `parallel_utils/{communication_op,cp,userbuffer,nccl_memory}.py` | 82+54+240+100 | 通信原语与缓冲 |
| `eplb/{__init__,policy,shuffle}.py` | 434+140+218 | EPLB 全链路 |
| `distributed/shm_broadcast.py` | 461 | Day 2 已精读，今日只回顾角色 |
| `deep_cluster/topology/bind.py` | 99 | NUMA 绑定 |
| `tilelang_ops/prepare_cp_mirror.py`、`compute_seqlens_k_zos.py` | 394+121 | balanced CP 的两个证据 kernel |
| `doc/inference_deployment_qa.md` EP/CP 章节 | — | 此时应该能看懂细节了 |

---

## 一、并行全景：一个 world，四种切分

### 1.1 组的建立：`create_process_groups` 的 rearrange 语义（`parallel_state.py:70-126`）

Day 2 读过"每个设备组都有 gloo 镜像"；今天补上**组怎么切**。`world = max(enp, dp × cp)` 个 rank（每个 rank 一个 GPU 进程：DP rank 的引擎 + 其 CP worker），四组用 einops 风格的 rearrange 定义：

```python
ranks = torch.arange(max(enp_size, dp_size * cp_size))
# DP："(replica dp cp) -> (replica cp) dp" —— 固定 cp 位、跨 dp 的 rank 成组
dp_ranks = rearrange(ranks, "(replica dp cp) -> (replica cp) dp", dp=dp_size, cp=cp_size)
# CP："(dp cp) -> dp cp" —— 固定 dp 位（同一引擎）、跨 cp 的 rank 成组
cp_ranks = rearrange(ranks, "(dp cp) -> dp cp", cp=cp_size)
# EP："(edp ep) -> edp ep" —— 连续 ep_size 个 rank 成组
ep_ranks = rearrange(ranks, "(edp ep) -> edp ep", ep=ep_size)
# ENP："(replica enp) -> replica enp" —— engram 表切分
```

用一个贯穿全天的例子（**world=8，dp=4，cp=2，ep=8**，即 4 个引擎各带 1 个 CP slave）：

```
rank:      0     1     2     3     4     5     6     7
engine:    └─0─┘     └─1─┘     └─2─┘     └─3─┘      （cp_rank 0 = driver：0,2,4,6）
CP 组:     {0,1}     {2,3}     {4,5}     {6,7}      同一引擎的 driver↔slave
DP 组:     {0,2,4,6}（driver 们，调度投票组）  {1,3,5,7}（slave 们）
EP 组:     {0,...,7}（= world，MoE all-to-all 跨所有引擎与 CP slot）
```

三个立即可读出的事实：

1. **DP 组把各引擎的 driver 连起来**（cp=1 时 DP 组=整个 world）。Day 2 的调度投票（`all_reduce(MIN)`）和 `has_unfinished`（`all_reduce(MAX)`）就跑在 driver 的 DP 组上——**attention DP 的唯一"通信"是调度面的一致性协议，不传数据**；
2. **CP 组是引擎内部的**（driver↔自己的 CP slave），通信走 shm 广播（方法分发）+ DeepEP BucketBuffer（数据）；
3. **EP 组是连续切分的，可以跨引擎、跨 CP slot**——MoE 的 all-to-all 天然横跨 attention 的 DP/CP 结构。这正是"EP 与 DP 共享同一批卡（#DP = #EP）"的物理形态：attention 按请求独立算，MoE 把大家的 token 汇到 expert 所在的卡上再还回来。

还有两个细节：`ep_in_dp` 与 EP 同切分但**独立通信域**（避免不同集合通信交错）；`init_device_groups()` 对所有 NCCL/HCCL 子组各跑一次 `all_reduce` 预热——注释解释了原因（默认 PG 换 gloo 后设备组变成 lazy 初始化，"必须先跑一次 collective 才会真正建 comm"，且 engram 的 deep_ep 会复用 comm 指针，空指针直接段错误）。

### 1.2 参数语义复盘（对照 examples 三种跑法）

| 跑法 | dp | cp | ep | 语义 |
|---|---|---|---|---|
| `--ep 8` | 8（=ep/cp） | 1 | 8 | 8 个引擎、EP=world：**生产 decode 的形状**（EP64/DP64 的缩小版） |
| `--ep 2 --dp 8` | 8 | 1 | 2 | 8 个引擎，每 2 个引擎共享一组 expert：**生产 prefill 的形状**（EP2–8） |
| `--ep 2 --cp 8` | ... 注意 `dp = ep/cp` 需整除，此跑法实际是 `--ep 2 --dp 8`（README 的例子）＋ CP 实验 | 8 | 2 | balanced CP：每引擎 8 个 CP worker，序列 zigzag 切分 |

（`offline_inference.py:27-29`：`dp_size = max(ep_size // cp_size, 1)`；`world = dp × cp`；`assert world % ep == 0`。）

---

## 二、Attention DP：为什么 MLA 适合（README 关键问题 1）

### 2.1 本质区别：切"请求"而不是切"张量"

| 维度 | vLLM V1（TP/PP/DP） | DeepInfer（Attention DP） |
|---|---|---|
| 切分对象 | TP 切 **head 维**（QKV 权重按 head 分卡），PP 切 **层**，DP 切**请求副本** | 只切**请求**：每 rank 完整持有全部层、全部 head，处理互不相交的请求集合 |
| 每 step 通信 | TP：2 次 all-reduce/层；PP：P2P 激活 | **零数据通信**（DP 组只有调度投票的 all_reduce） |
| 权重 | TP 下每卡 1/TP 的 attention 权重 | **每卡全量 attention 权重**（重复） |
| 前提 | 通用（任意模型） | KV 必须小到"不值得切" |
| MoE 怎么办 | TP/EP 混合 | EP 独立成组，all-to-all 汇聚 |

**为什么 MLA 让这个前提成立**：MLA 把 KV 压成 latent（`kv_lora_rank + rope_head_dim` ≈ 576 元素/token，且 MQA `num_key_value_heads == 1`），一条请求的 KV 总量比 GQA 小一到两个数量级。TP 切 KV 的收益（省显存）消失，成本（每层 2 次 all-reduce + KV 块表跨卡）全在——**不如每卡全量权重、按请求分片**。Q&A 的原话就是"attention uses data parallelism (#DP = #EP)"：attention 的 DP 副本数对齐 EP，MoE 层把 DP 副本的 token 通过 all-to-all "借"给 expert 所在的卡，算完再还——**attention DP 不是为了省通信而存在，是为了和 Wide-EP 组合成"attention 免费、MoE 聚合"的拓扑**。

这也解释了 Day 2 遗留的"DP 组内步型必须一致"：MoE 的 all-to-all 是 EP 组级别的集合通信，EP 组横跨多个 DP rank——如果一个 rank 做 prefill、另一个做 decode，all-to-all 的语义（buffer 形状、token 数上限）就对不上。于是有了 ALL/ANY 投票。

### 2.2 生产与代码的错位

Q&A 说 "TP and CP are not used for now"（生产），但代码完整支持 CP（今日第三节）。错位的原因读 `benchmark.py --cp 8 --prefill_batch_size 65536` 的存在就能理解：**CP 是为超大 prefill chunk 准备的实验/超节点能力**——128K 上下文的 prefill 单卡放不下激活时，用 CP 把序列切开；线上 prefill 用 EP2–8 的小机器 + chunked prefill 已经够用，CP 留作超节点形态的储备。

---

## 三、balanced CP：zigzag 镜像切分（README 关键问题 2）

### 3.1 通信基础设施：`CPManager`（`parallel_utils/cp.py`，54 行）

```python
class CPManager:
    def __init__(self, group, num_bytes):
        self._buffer = BucketBuffer(group, num_bytes, explicitly_destroy=True)   # DeepEP 的桶缓冲
    def session(self): ...                      # BucketSession： gathered 张量别名 buffer 存储，下个 session 复用
    def get_inplace_tensor(self, shape, dtype):
        region = self._session.allocate((self._size, *shape))    # 全组共享区域
        return region[self._rank]                                # ★ 本 rank 的连续分片
    def all_gather(self, x) -> Callable:
        handle = self._buffer.all_gather(x)
        return lambda: handle.wait().view(self._size, *x.shape).flatten(0, 1)   # wait 后沿 token 轴拼接
```

CP 的数据面就这一个原语：**all_gather 返回惰性 getter**（Day 4 见过它被用来做"通信先行"），gathered 结果是各 rank 分片沿 token 轴的拼接。缓冲大小由 `allocate_cp` 估算：`max_tokens × per-token 字节 × cp_size + 8 × RDMA 对齐`。

### 3.2 zigzag 的定义：从 `prepare_cp_mirror_ref` 逐行读（`prepare_cp_mirror.py:126-157`）

```python
ps = ceil(sq / (2 * cp_size))                    # 每序列切成 2P 个 block，每 block ps 个 token
for rank in range(cp_size):
    for half, block in enumerate((rank, 2*cp_size - 1 - rank)):   # ★ 镜像配对：rank r 拿第 r 块和倒数第 r+1 块
        for j in range(ps):
            local_idx = ps * block + j
            if local_idx < sq:
                reverse_index[cu_q[i] + local_idx] = rank * total_per_rank + cu_elems[i] + half * ps + j
            if rank == cp_rank:
                mirror_index.append(cu_q[i] + local_idx if local_idx < sq else -1)   # -1 = padding
                delta = sk - sq + local_idx + 1
                ks[ri].append(cu_k_cn[ri][i]); ke[ri].append(cu_k_cn[ri][i] + delta // ratio)
```

**手排例子（今日实验 1 的预演）**：cp_size=4（P=4，2P=8 块），序列长 sq=56，ps=7：

```
块号:      0    1    2    3    4    5    6    7
token:   0-6  7-13 14-20 21-27 28-34 35-41 42-48 49-55
rank 0:  ─block0─┐                                   ┌─block7─┐   位置和 = 21+364 = 385
rank 1:        └─block1─┐                       ┌─block6─┘         70+315 = 385
rank 2:              └─block2─┐           ┌─block5─┘              119+266 = 385
rank 3:                    └─block3─┐ ┌─block4─┘                  168+217 = 385
```

**"balanced" 的三层含义**（README 关键问题 2 的完整答案）：

1. **token 数均衡**：每 rank 每序列固定拿 2 块，任意序列长度下 rank 间 token 数差 ≤ ps；短序列照样切（ps 可以是 1，`_MIRROR_SEQS` 里有 `[1, 0, 7, ...]` 的用例）——对比朴素"按 token 轮转"（切点不齐、kernel 形状碎）和"连续均分"（最后一块尾巴偏斜）；
2. **attention 工作量均衡（zigzag 的本质）**：causal attention 里块 b 的成本 ∝ (b+1)（query 要看前面所有块）。镜像配对 (b, 2P−1−b) 满足 **(b+1) + (2P−b) = 2P+1 恒定**——上例每 rank 的 Σ(query 位置) 都是 385。V4.1 的实际度量是压缩 key 数：`ke − ks = (sk − sq + local_idx + 1) // ratio`，kernel 直接按 per-row 工作量输出（DFlash/稀疏层的调度也用它）；
3. **静态形状均衡**：`output_len = Σ 2·ps` 对**每个 rank 相同**（缺的 token 填 -1，`valid_token_mask = mirror_cp_index != -1` 屏蔽）——mirror 缓冲形状固定，对 CUDA graph 和 kernel launch 都友好。

### 3.3 CP 数据流全链（把 Day 3/4 的碎片接起来）

```
prepare_for_mirror_cp()（metadata，Day 3）          prepare_cp_mirror（TileLang，一次 kernel 出 4 个索引）
  │ mirror_cp_index / reverse / ks / ke                │
  ▼                                                    ▼
forward 开头: hidden = gather(hidden, mirror_cp_index)        ← 本 rank 的镜像布局
每层: 各 query-aligned 字段 gather 到 mirror 布局；attention 用 mirror 版 ks/ke
      userbuffer.cp_session() 内: all_gather（通信先行，Day 4 的 ④）
层尾(deepseek.py:241-244):
      gathered = userbuffer.cp.all_gather(hidden)()           ← wait 后 [P × cp_mirror_len] 拼接
      hidden = gather(gathered, mirror_cp_reverse_index)      ← 还原原始 varlen 顺序
```

**YoCo × CP 的叠加**（Day 3 已见调用点）：顺序是"**先 compact 再 CP**"——`get_consumer_view` 里用 compact 版 `cu_seqlens_q` 重新跑一遍 `prepare_cp_mirror`（`sk − sq_c = q_start + clip_prefix`，推导注释原文在 yoco.py:136-139），再用 `yoco_cp_boundary_fused` 做"各 rank full-mirror 保留行 → all-gather 拼接缓冲 → compact-mirror 行"的边界重分布。**ZOS × CP 的叠加**：`compute_seqlens_k_zos`（README 点名的另一个 kernel）在 GPU 上做 `seqlens_k += seqlens_q; seqlens_k −= (nextn − prev_accepted)`——decode 的 per-row KV 长度由上一步接受数修正，**CP 的均衡度量（seqlens_q/k）在 ZOS 下同样要逐 rank 一致地修正**，这就是它作为"线索"的含义：CP 的账本和 ZOS 的账本是同一套 per-row 量。

### 3.4 控制面广播：`broadcast_tensor_dict`（`communication_op.py:51-82`）

Day 3 只说了"follower 重建"；今天看清双通道：**非张量（形状/type/标量）走 `cp_broadcast_object`（shm 环形缓冲，Day 2），CUDA 张量走逐个 `dist.broadcast`（NCCL）**——`InputMetadata.broadcast()` 递归收集（带引用环防护），按路径写回。注释里留着 "TODO: Maybe use deepep multi_tensor_broadcast later"。

---

## 四、MoE EP 的通信缓冲：`userbuffer.py`

`_UserBufferManager` 是**进程级通信缓冲的唯一持有者**（一个 world 一份，`userbuffer` 单例），管理四种 buffer：

| Buffer | 库 | 用途 | 大小估算 |
|---|---|---|---|
| `EPBuffer` | DeepEP | dispatch/combine（非 mega 路径） | `num_max_tokens_per_rank × hidden × topk`（+FP8 dispatch） |
| `CPManager`（BucketBuffer） | DeepEP | CP all_gather | `max_tokens × bytes/token × cp_size` + RDMA 对齐 |
| `EngramBuffer` | DeepEP | engram 表通信 | `get_storage_size_hint`（可 CPU RDMA storage） |
| `SymmBuffer` | DeepGEMM | Mega MoE 的对称缓冲 | gate_up/down × tokens × topk（首个 spec 必须最大，后续是视图） |

三个生产级细节：

1. **通信 SM 数自动推导**（`allocate_ep`）：`USERBUFFER_COMM_SMS=0` 时，`num_sms = EPBuffer.get_theoretical_num_sms(...)`，再夹到 `[理论值, 设备 SM 数 − min(32, SM/4)]`——**给计算 kernel 永远留至少 1/4 的 SM**。Day 4 的 `limit_num_sms` 上下文在共享 expert 计算时临时再削 deep_gemm/tile_kernels 的 SM，让位给 combine；
2. **`allow_multiple_reduction = not (BATCH_INVARIANT and REARRANGE_INTERVAL > 0)`**——一行浓缩了两个特性的冲突：**多副本归约的顺序不确定会破坏 batch-invariant，而 EPLB 恰恰制造多副本**。batch-invariant + EPLB 同时开时强制单归约路径；
3. **`nccl_memory.py`**：定期采样所有已初始化 NCCL communicator 的显存占用并按别名（组名）打日志——分布式显存泄漏的监控仪表。

---

## 五、EPLB：统计 → 策略 → 迁移 → 热切换（README 关键问题 3）

### 5.1 状态与触发（`eplb/__init__.py`）

`ExpertLoadBalancer` 持有（每 MoE 层一行的表）：

```python
expert_count          # [L, num_local_physical]：每 step 清零，由 dispatch 累计（Day 4）
physical_to_logical   # [L, num_physical]
logical_to_physical   # [L, num_logical, num_redundant+1]，无效位 -1 —— gate 的路由表
logical_count         # [L, num_logical]：各逻辑 expert 的副本数
```

**触发条件**（`_update_rearrangement`）：

- 步进计数 `rearrangement_step >= DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL`（**默认 0 = 关闭**！examples 显式设 30，生产 prefill 1000 / decode 10000）；
- 窗口约束：`0 < EXPERT_COUNT_WINDOW_SIZE（默认 100） ≤ interval`（违反直接 raise）；
- **首次提前**：初始计数 = `interval − interval/4`——"尽早纠正冷启动的不均衡"（新部署的初始布局是盲目的）；
- NORMAL 引擎的两个模式开关：`REARRANGE_AFTER_PREFILL`（prefill 后下一个非 prefill 步立即 rearrange）/ 否则 prefill 步跳过窗口累计；
- 每步 `expert_count.zero_()`（`step()` 由 `model.sample` 调用，Day 4 见过）。

**迁移粒度**：**physical expert 槽位级，逐层独立**——每层的 `physical_to_logical` 单独重排（层间热点分布不同）。迁移的是**权重张量**（gate_up/down 两个矩阵），逻辑 expert 的副本数可变（`logical_to_physical` 的第二维就是冗余度+1）。

### 5.2 策略：两阶段贪心（`policy.py`）

`rebalance_experts_global(tokens_per_expert[window, L, num_logical], ...)`：

**阶段一：冗余副本分配**（`num_redundant` 轮，每轮复制一个 expert）：

```python
score = tokens_per_expert / logical_count          # 每副本负载
score_added = tokens_per_expert / (logical_count + 1)   # 加一个副本后的每副本负载
# 把 argmax 位置的 score 替换成 score_added → values[j] 即"副本给 j"时的最大每副本负载
redundancy_indices = values.sum(0).argmin(-1)      # 按 step 求和取 argmin → 本轮复制谁
```

注释里有个防并列的 hack：`tokens_per_expert + arange * 1e-4`——"NOTE: Values in score must be different from each other"（argmax 并列时行为未定义，加扰动强制唯一）。

**阶段二：槽位到 rank 的贪心放置**（`_batch_greedy_indices`）：

```python
# 优化目标（注释原文）：minimize sum_{step i} max_{rank j} (sum_{expert e 放在 j} e 在 step i 的负载)
# 贪心：按每副本历史总负载降序逐个放置，每次尝试所有 rank，选使目标最小的
for job_index in range(num_physical):
    candidates = max(loads + vector, peaks)        # 放到 rank j 后各 step 的最大负载
    scores = candidates.sum(-1)                    # Σ_step max
    scores.masked_fill_(occupancy == capacity, inf)  # 满员 rank 排除
    rank = scores.argmin(-1)                       # 同分取最小 rank ID（确定性！）
```

**目标函数就是 moe_lb 的最小化**：`Σ_step max_rank load` 的最小化 = 让"最慢 rank"尽量不慢。全程张量化（`num_physical` 次迭代的批量 GEMM-ish 操作），没有 Python 双层循环。

### 5.3 迁移：两种 shuffle（`shuffle.py`）

| 路径 | 条件 | 机制 |
|---|---|---|
| `expert_model_parallel_shuffle_inplace_intranode` | `is_within_nvl_domain(EP) and is_gpu` | **symm_mem IPC 镜像**：`_initialize_shuffle` 只为**一层**的权重建对称镜像（逐层刷新复用），每个 rank 拿到全组 rank 的 buffer 句柄，直接读对端权重——**零通信原语，纯显存拷贝**。Q&A 的 "expert weights are shuffled within the group over NVLink symm_mem" 的代码实体 |
| `expert_model_parallel_shuffle_inplace` | 跨节点 / NPU | **NCCL batched P2P**：先做卡内拷贝（`_plan_intra_copies`），跨卡的按 `(对端 rank 差 mod 组数)` 分桶（`REARRANGE_P2P_GROUPS` 默认 1）批量 `batch_isend_irecv`；`_paired_sender` 保证收发两端配对一致（TODO 注释："可以考虑优先机器内通信和 broadcast"） |

配对算法的细节值得一提：同一 expert 有多个副本时，senders/receivers 按 `_paired_sender(j) = j // n`（前 n×senders 个均分、其余轮转）配对——**纯函数、两端算出同样结果**，不需要额外协商。

### 5.4 热切换与监控

- **路由切换是零成本的**：新 `logical_to_physical_map` 写进 `ExpertLoadBalancer`，下一个 dispatch 的 gate kernel 直接读（Day 4 的 `to_physical_map` 参数）——**权重搬完的瞬间即生效，没有 barrier**；
- `moe_lb` 的来源：`_update_load_balance_stats` 每 `MOE_LB_STAT_INTERVAL`（默认 128）步把每 rank 的 token 数 `all_gather_into_tensor`，`(mean, max)` 求和后非阻塞搬 CPU——benchmark 的 `moe_lb = Σmax/Σ(Σavg)`（Day 1 的 1.074/1.045/1.090 就是这里产出的）；
- `_log_rearrangement_metrics`：rank_balance / logical_balance / peak 负载 / 耗时，`DEEPINFER_SHOW_EPLB_DETAIL` 还能打**逐层 × 逐 expert** 的负载表（tabulate）。

### 5.5 与 Q&A / examples 的参数对照

| 参数 | Q&A（生产） | examples（离线） | environ 默认 |
|---|---|---|---|
| 冗余 expert 下限 | prefill 4/卡、decode 1/卡 | `OVERRIDE_NUM_EXTRA_EXPERTS=32`（下限） | — |
| 统计窗口 | prefill 100 步、decode 1000 步 | 3（省显存） | 100 |
| 重排间隔 | prefill 1000 步、decode 10000 步 | 30 | **0（关闭）** |
| 实测 moe_lb | 1.074(EP4DP4) / 1.045(EP2DP2) / 1.090(EP64) | — | — |

注意引擎默认**不开** EPLB——它是部署决策（重排有代价：权重搬运期间的带宽占用），生产按 prefill/decode 的稳定性差异选不同窗口。

---

## 六、deep_cluster：拓扑感知如何影响切分

Day 1 读过 `hardware/`（探测）；今天补 `topology/bind.py`（99 行）和它的消费方：

```python
def which_numa(i_gpu=None, i_ib=None) -> int:
    # GPU：PCI 设备号 → /sys/bus/pci/devices/<bus>/numa_node（IB：/sys/class/infiniband/mlx5_N/device/numa_node）
def bind_numa(node: int) -> int:
    # libnuma：numa_sched_setaffinity（CPU 亲和）+ set_mempolicy(MPOL_BIND)（内存落位）
```

消费方清单（谁在绑）：

| 消费方 | 代码 | 为什么 |
|---|---|---|
| worker 初始化 | `worker/base.py: _bind_numa(local_rank)` + `bind_cpu(gpu_id)` | H2D/D2H 走本地 NUMA 的内存（失败告警："内存落位退化为 first-touch，可能导致 H2D 性能下降"） |
| SO 编译线程池 | `engine/llm_engine.py: get_idle_cpus → SOMatcherBuilder(bind_cpus=...)` | 编译不抢引擎核 |
| 通信域判定 | `PG.is_within_nvl_domain` → `deep_ep.get_physical_domain_size` | **Mega MoE 与 EPLB intranode shuffle 的开关**（NPU 恒 True） |
| engine 启动 | `get_bf16_tflops_limit`（Day 2，成本模型） | 按实际卡型估算 |
| engram CPU RDMA | `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE`（默认 NPU 强制 GPU）+ `expose_all_gpus_to_cuda_driver` | CPU RDMA storage 需要正确的 NUMA/设备可见性 |

**拓扑 → 并行的传导链**：PCI/NVLink 拓扑决定 `is_within_nvl_domain` → 决定 Mega MoE（NVL 域内一次调用）还是 dispatch/combine → 决定 EPLB 用 IPC 镜像还是 P2P。**并行策略不是静态配置，是拓扑的函数**——这是 Day 1"deep_cluster 独立成包"的真正原因：探测结果要喂给引擎、worker、kernel 三层。

---

## 七、并行策略矩阵图（今日核心产出，SVG）

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 960" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect width="1120" height="960" fill="#fafbfc"/>
  <text x="560" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">DeepInfer 并行策略矩阵（示例：world=8，dp=4，cp=2，ep=8）</text>

  <!-- rank layout -->
  <rect x="50" y="48" width="1020" height="150" rx="10" fill="#f8fafc" stroke="#94a3b8"/>
  <text x="70" y="72" font-size="13" font-weight="700" fill="#0f172a">rank 布局与四种组的覆盖（ranks 0-7；引擎 = cp_rank 0 的进程）</text>
  <g font-size="12" font-weight="700" text-anchor="middle">
    <rect x="90"  y="88" width="86" height="34" rx="6" fill="#dbeafe" stroke="#2563eb"/><text x="133" y="109">rank 0 ★</text>
    <rect x="190" y="88" width="86" height="34" rx="6" fill="#e2e8f0" stroke="#94a3b8"/><text x="233" y="109">rank 1</text>
    <rect x="290" y="88" width="86" height="34" rx="6" fill="#dbeafe" stroke="#2563eb"/><text x="333" y="109">rank 2 ★</text>
    <rect x="390" y="88" width="86" height="34" rx="6" fill="#e2e8f0" stroke="#94a3b8"/><text x="433" y="109">rank 3</text>
    <rect x="490" y="88" width="86" height="34" rx="6" fill="#dbeafe" stroke="#2563eb"/><text x="533" y="109">rank 4 ★</text>
    <rect x="590" y="88" width="86" height="34" rx="6" fill="#e2e8f0" stroke="#94a3b8"/><text x="633" y="109">rank 5</text>
    <rect x="690" y="88" width="86" height="34" rx="6" fill="#dbeafe" stroke="#2563eb"/><text x="733" y="109">rank 6 ★</text>
    <rect x="790" y="88" width="86" height="34" rx="6" fill="#e2e8f0" stroke="#94a3b8"/><text x="833" y="109">rank 7</text>
  </g>
  <text x="905" y="106" font-size="10.5" fill="#475569">★ = driver</text>
  <!-- CP group braces -->
  <path d="M133,126 v10 h100 v-10" fill="none" stroke="#16a34a" stroke-width="2.5"/>
  <text x="183" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">CP 组 {0,1}</text>
  <path d="M333,126 v10 h100 v-10" fill="none" stroke="#16a34a" stroke-width="2.5"/>
  <text x="383" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">{2,3}</text>
  <path d="M533,126 v10 h100 v-10" fill="none" stroke="#16a34a" stroke-width="2.5"/>
  <text x="583" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">{4,5}</text>
  <path d="M733,126 v10 h100 v-10" fill="none" stroke="#16a34a" stroke-width="2.5"/>
  <text x="783" y="150" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">{6,7}</text>
  <!-- DP group line -->
  <path d="M133,160 q200,26 400,0 t400,0" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="7,4"/>
  <text x="533" y="192" text-anchor="middle" font-size="11" font-weight="700" fill="#1e40af">DP 组（drivers）：{0,2,4,6} —— 调度投票；slaves {1,3,5,7} 另一组</text>
  <!-- EP group bracket -->
  <path d="M90,58 h786" fill="none" stroke="#dc2626" stroke-width="3"/>
  <text x="483" y="54" text-anchor="middle" font-size="11.5" font-weight="700" fill="#991b1b">EP 组 = world {0..7}：MoE all-to-all 横跨所有引擎与 CP slot</text>

  <!-- matrix -->
  <text x="70" y="228" font-size="14" font-weight="700" fill="#0f172a">并行策略矩阵：每个维度切什么、怎么通信、代码在哪</text>
  <g font-size="11.5">
    <!-- header -->
    <rect x="70" y="240" width="130" height="30" fill="#e2e8f0" stroke="#94a3b8"/><text x="135" y="260" text-anchor="middle" font-weight="700" fill="#0f172a">维度</text>
    <rect x="200" y="240" width="200" height="30" fill="#e2e8f0" stroke="#94a3b8"/><text x="300" y="260" text-anchor="middle" font-weight="700" fill="#0f172a">切什么</text>
    <rect x="400" y="240" width="270" height="30" fill="#e2e8f0" stroke="#94a3b8"/><text x="535" y="260" text-anchor="middle" font-weight="700" fill="#0f172a">通信原语</text>
    <rect x="670" y="240" width="380" height="30" fill="#e2e8f0" stroke="#94a3b8"/><text x="860" y="260" text-anchor="middle" font-weight="700" fill="#0f172a">代码文件 / 关键类</text>
    <!-- DP row -->
    <rect x="70" y="270" width="130" height="76" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
    <text x="135" y="298" text-anchor="middle" font-weight="700" fill="#1e3a8a">Attention DP</text>
    <text x="135" y="316" text-anchor="middle" fill="#1e40af" font-size="10.5">（生产 #DP=#EP）</text>
    <rect x="200" y="270" width="200" height="76" fill="#eff6ff" stroke="#2563eb"/>
    <text x="212" y="290" font-size="10.5" fill="#1e40af">请求数据流：每 rank 全量</text>
    <text x="212" y="306" font-size="10.5" fill="#1e40af">权重，独立 batch；MLA 的</text>
    <text x="212" y="322" font-size="10.5" fill="#1e40af">latent KV 小到不值得切</text>
    <rect x="400" y="270" width="270" height="76" fill="#eff6ff" stroke="#2563eb"/>
    <text x="412" y="290" font-size="10.5" fill="#1e40af">数据面零通信；控制面：</text>
    <text x="412" y="306" font-size="10.5" fill="#1e40af">步型投票 all_reduce(MIN)×2、</text>
    <text x="412" y="322" font-size="10.5" fill="#1e40af">unfinished all_reduce(MAX)</text>
    <text x="412" y="338" font-size="10.5" fill="#1e40af">（gloo CPU 组，不碰 CUDA）</text>
    <rect x="670" y="270" width="380" height="76" fill="#eff6ff" stroke="#2563eb"/>
    <text x="682" y="292" font-size="10.5" fill="#1e40af" font-family="Consolas, monospace">parallel_state.py（PG.dp / cpu_dp_group）</text>
    <text x="682" y="308" font-size="10.5" fill="#1e40af" font-family="Consolas, monospace">csrc/engine/scheduler.hpp（_dp_allreduce 投票）</text>
    <text x="682" y="324" font-size="10.5" fill="#1e40af" font-family="Consolas, monospace">llm_engine.py:665（has_unfinished）</text>
    <!-- CP row -->
    <rect x="70" y="346" width="130" height="76" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
    <text x="135" y="374" text-anchor="middle" font-weight="700" fill="#14532d">balanced CP</text>
    <text x="135" y="392" text-anchor="middle" fill="#166534" font-size="10.5">（prefill 序列维）</text>
    <rect x="200" y="346" width="200" height="76" fill="#f0fdf4" stroke="#16a34a"/>
    <text x="212" y="366" font-size="10.5" fill="#166534">序列 token 的 zigzag 镜像：</text>
    <text x="212" y="382" font-size="10.5" fill="#166534">rank r 拿第 r 块 + 倒数第 r+1</text>
    <text x="212" y="398" font-size="10.5" fill="#166534">块；(b+1)+(2P−b)=2P+1 恒定</text>
    <text x="212" y="414" font-size="10.5" fill="#166534">→ 每 rank 工作量均衡</text>
    <rect x="400" y="346" width="270" height="76" fill="#f0fdf4" stroke="#16a34a"/>
    <text x="412" y="366" font-size="10.5" fill="#166534">all_gather（DeepEP BucketBuffer，</text>
    <text x="412" y="382" font-size="10.5" fill="#166534">惰性 getter 通信先行）+ gather 重排；</text>
    <text x="412" y="398" font-size="10.5" fill="#166534">控制面：shm 广播（对象）+</text>
    <text x="412" y="414" font-size="10.5" fill="#166534">NCCL broadcast（张量）</text>
    <rect x="670" y="346" width="380" height="76" fill="#f0fdf4" stroke="#16a34a"/>
    <text x="682" y="368" font-size="10.5" fill="#166534" font-family="Consolas, monospace">parallel_utils/cp.py（CPManager）</text>
    <text x="682" y="384" font-size="10.5" fill="#166534" font-family="Consolas, monospace">tilelang_ops/prepare_cp_mirror.py（zigzag）</text>
    <text x="682" y="400" font-size="10.5" fill="#166534" font-family="Consolas, monospace">input_metadata.py（mirror 字段 + broadcast）</text>
    <text x="682" y="416" font-size="10.5" fill="#166534" font-family="Consolas, monospace">userbuffer.py（allocate_cp / cp_session）</text>
    <!-- EP row -->
    <rect x="70" y="422" width="130" height="86" fill="#fef2f2" stroke="#dc2626" stroke-width="1.8"/>
    <text x="135" y="450" text-anchor="middle" font-weight="700" fill="#7f1d1d">MoE EP</text>
    <text x="135" y="468" text-anchor="middle" fill="#991b1b" font-size="10.5">（Wide-EP：decode 64-128）</text>
    <text x="135" y="484" text-anchor="middle" fill="#991b1b" font-size="10.5">prefill 2-8</text>
    <rect x="200" y="422" width="200" height="86" fill="#fef2f2" stroke="#dc2626"/>
    <text x="212" y="442" font-size="10.5" fill="#991b1b">expert 权重按 physical slot</text>
    <text x="212" y="458" font-size="10.5" fill="#991b1b">均分到 EP 组；token 按 top-k</text>
    <text x="212" y="474" font-size="10.5" fill="#991b1b">路由到 expert 所在卡 → 每 step</text>
    <text x="212" y="490" font-size="10.5" fill="#991b1b">每卡读的 expert 权重 1/P</text>
    <text x="212" y="504" font-size="10.5" fill="#991b1b">（Wide-EP 的收益来源）</text>
    <rect x="400" y="422" width="270" height="86" fill="#fef2f2" stroke="#dc2626"/>
    <text x="412" y="442" font-size="10.5" fill="#991b1b">NVL 域内：Mega MoE（SymmBuffer，</text>
    <text x="412" y="458" font-size="10.5" fill="#991b1b">gate+dispatch+GEMM+combine 融合）</text>
    <text x="412" y="474" font-size="10.5" fill="#991b1b">域外：EPBuffer.dispatch/combine</text>
    <text x="412" y="490" font-size="10.5" fill="#991b1b">（all-to-all，USERBUFFER_COMM_SMS</text>
    <text x="412" y="504" font-size="10.5" fill="#991b1b">限流，共享 expert 与 combine 重叠）</text>
    <rect x="670" y="422" width="380" height="86" fill="#fef2f2" stroke="#dc2626"/>
    <text x="682" y="444" font-size="10.5" fill="#991b1b" font-family="Consolas, monospace">v4_modules/mlp.py（DeepseekMoE 三路径）</text>
    <text x="682" y="460" font-size="10.5" fill="#991b1b" font-family="Consolas, monospace">userbuffer.py（EPBuffer / SymmBuffer / SM 推导）</text>
    <text x="682" y="476" font-size="10.5" fill="#991b1b" font-family="Consolas, monospace">parallel_state.py（ep / ep_in_dp 组）</text>
    <text x="682" y="492" font-size="10.5" fill="#991b1b" font-family="Consolas, monospace">runtime_config.py（NCCL env 定制）</text>
    <text x="682" y="504" font-size="10.5" fill="#991b1b" font-family="Consolas, monospace">tilelang_ops/compute_seqlens_k_zos.py（ZOS 账本）</text>
    <!-- EPLB row -->
    <rect x="70" y="508" width="130" height="76" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
    <text x="135" y="536" text-anchor="middle" font-weight="700" fill="#7c2d12">EPLB</text>
    <text x="135" y="554" text-anchor="middle" fill="#9a3412" font-size="10.5">（EP 的运行时重排）</text>
    <rect x="200" y="508" width="200" height="76" fill="#fff7ed" stroke="#ea580c"/>
    <text x="212" y="528" font-size="10.5" fill="#9a3412">expert 的物理放置：热点逻辑</text>
    <text x="212" y="544" font-size="10.5" fill="#9a3412">expert 加副本（physical slot 级、</text>
    <text x="212" y="560" font-size="10.5" fill="#9a3412">逐层独立）；gate 经映射表路由，</text>
    <text x="212" y="576" font-size="10.5" fill="#9a3412">权重搬完即热切换</text>
    <rect x="400" y="508" width="270" height="76" fill="#fff7ed" stroke="#ea580c"/>
    <text x="412" y="528" font-size="10.5" fill="#9a3412">统计：dispatch 累计 + all_reduce /</text>
    <text x="412" y="544" font-size="10.5" fill="#9a3412">all_gather（moe_lb）；迁移：机内</text>
    <text x="412" y="560" font-size="10.5" fill="#9a3412">symm_mem IPC 镜像（零通信原语）/</text>
    <text x="412" y="576" font-size="10.5" fill="#9a3412">跨机 NCCL batched P2P</text>
    <rect x="670" y="508" width="380" height="76" fill="#fff7ed" stroke="#ea580c"/>
    <text x="682" y="530" font-size="10.5" fill="#9a3412" font-family="Consolas, monospace">eplb/__init__.py（ExpertLoadBalancer 状态机）</text>
    <text x="682" y="546" font-size="10.5" fill="#9a3412" font-family="Consolas, monospace">eplb/policy.py（两阶段贪心：副本分配+放置）</text>
    <text x="682" y="562" font-size="10.5" fill="#9a3412" font-family="Consolas, monospace">eplb/shuffle.py（IPC 镜像 / P2P 分桶）</text>
    <text x="682" y="578" font-size="10.5" fill="#9a3412" font-family="Consolas, monospace">v4_modules/mlp.py（gate 的 to_physical_map）</text>
    <!-- ENP row -->
    <rect x="70" y="584" width="130" height="56" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.5"/>
    <text x="135" y="608" text-anchor="middle" font-weight="700" fill="#581c87">ENP</text>
    <text x="135" y="626" text-anchor="middle" fill="#7e22ce" font-size="10.5">（engram 表切分）</text>
    <rect x="200" y="584" width="200" height="56" fill="#fdf4ff" stroke="#a855f7"/>
    <text x="212" y="604" font-size="10.5" fill="#7e22ce">ngram 嵌入表跨 rank 分片，</text>
    <text x="212" y="620" font-size="10.5" fill="#7e22ce">token 的 ngram 哈希查表</text>
    <rect x="400" y="584" width="270" height="56" fill="#fdf4ff" stroke="#a855f7"/>
    <text x="412" y="604" font-size="10.5" fill="#7e22ce">EngramBuffer（DeepEP，GPU 或</text>
    <text x="412" y="620" font-size="10.5" fill="#7e22ce">CPU RDMA storage，sl3 流量隔离）</text>
    <rect x="670" y="584" width="380" height="56" fill="#fdf4ff" stroke="#a855f7"/>
    <text x="682" y="606" font-size="10.5" fill="#7e22ce" font-family="Consolas, monospace">userbuffer.py（allocate_engram）</text>
    <text x="682" y="622" font-size="10.5" fill="#7e22ce" font-family="Consolas, monospace">v4_modules/engram.py（EngramBuffer 消费）</text>
    <!-- no TP row -->
    <rect x="70" y="640" width="130" height="40" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="5,3"/>
    <text x="135" y="658" text-anchor="middle" font-weight="700" fill="#475569">TP / PP</text>
    <text x="135" y="673" text-anchor="middle" font-size="10.5" fill="#64748b">（不存在）</text>
    <rect x="200" y="640" width="470" height="40" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="5,3"/>
    <text x="435" y="658" text-anchor="middle" font-size="10.5" fill="#475569">Q&amp;A："TP and CP are not used for now"（生产）；MLA latent KV 使 TP 无收益，all-reduce 纯开销</text>
    <text x="435" y="673" text-anchor="middle" font-size="10.5" fill="#64748b">CP 代码完整保留（超节点/超长 prefill 储备）</text>
    <rect x="670" y="640" width="380" height="40" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="5,3"/>
    <text x="860" y="664" text-anchor="middle" font-size="10.5" fill="#64748b">无对应代码 —— 这是与 vLLM 最大的结构性差异</text>
  </g>

  <!-- bottom: dataflow -->
  <rect x="50" y="700" width="1020" height="240" rx="10" fill="#fffdf5" stroke="#ca8a04" stroke-width="1.5"/>
  <text x="70" y="726" font-size="13.5" font-weight="700" fill="#78350f">一个 prefill step 的两种数据流（CP=4 示意）</text>
  <!-- CP flow -->
  <text x="70" y="752" font-size="11.5" font-weight="700" fill="#166534">① CP 数据流（attention，每层两次方向）</text>
  <g font-size="10" text-anchor="middle">
    <rect x="70" y="764" width="120" height="40" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="130" y="781" fill="#14532d">varlen hidden</text><text x="130" y="795" fill="#14532d">[total_q, H]</text>
    <text x="205" y="788" fill="#475569">─gather(mirror)→</text>
    <rect x="290" y="764" width="130" height="40" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="355" y="781" fill="#14532d">mirror 布局</text><text x="355" y="795" fill="#14532d">[2·Σps, H]</text>
    <text x="440" y="788" fill="#475569">─all_gather(惰性)→</text>
    <rect x="540" y="764" width="150" height="40" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="615" y="781" fill="#14532d">全组拼接</text><text x="615" y="795" fill="#14532d">[P × mirror_len, H]</text>
    <text x="710" y="788" fill="#475569">─gather(reverse)→</text>
    <rect x="810" y="764" width="130" height="40" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="875" y="781" fill="#14532d">还原 varlen 顺序</text><text x="875" y="795" fill="#14532d">（deepseek.py:244）</text>
  </g>
  <text x="70" y="824" font-size="10" fill="#92400e">层内：ks/ke（mirror 版压缩 key 计数）喂稀疏 attention；通信先行与 Q 投影重叠（attn.py:783-785）</text>
  <!-- MoE flow -->
  <text x="70" y="852" font-size="11.5" font-weight="700" fill="#991b1b">② MoE 数据流（EP=4 示意，非 mega 路径）</text>
  <g font-size="10" text-anchor="middle">
    <rect x="70" y="864" width="110" height="40" rx="6" fill="#fee2e2" stroke="#dc2626"/><text x="125" y="881" fill="#7f1d1d">各 rank token</text><text x="125" y="895" fill="#7f1d1d">（attention DP 副本）</text>
    <text x="195" y="888" fill="#475569">─gate(top-k)→ dispatch(all-to-all)→</text>
    <rect x="360" y="864" width="160" height="40" rx="6" fill="#fee2e2" stroke="#dc2626"/><text x="440" y="881" fill="#7f1d1d">expert 所在卡</text><text x="440" y="895" fill="#7f1d1d">grouped GEMM ×2 + swiglu</text>
    <text x="540" y="888" fill="#475569">─combine(all-to-all, bias=共享)→</text>
    <rect x="740" y="864" width="160" height="40" rx="6" fill="#fee2e2" stroke="#dc2626"/><text x="820" y="881" fill="#7f1d1d">各 rank 归位</text><text x="820" y="895" fill="#7f1d1d">（EPLB expert_count 顺手累计）</text>
  </g>
  <text x="70" y="924" font-size="10" fill="#92400e">NVL 域内整段替换为 fp8_fp4_mega_moe 一次调用（SymmBuffer）；dispatch/combine 的 SM 数被 USERBUFFER_COMM_SMS 限流，计算侧 limit_num_sms 让位</text>
</svg>
```

---

## 八、动手实验

### 实验 1（README 指定手算）：EP 切分与 EPLB 迁移

**第一步：64 卡 decode 超节点的 EP 切分。** 设 V4.1 的 routed expert 数为 E（从 HF config 读，仓库不含具体值），decode 生产配置"每卡 1 个冗余 expert 起步、总数向上取整到 ep_size 的倍数"：

```
physical = E + r，其中 r = 满足 (E + r) % 64 == 0 且 r ≥ 64 的最小值
每卡 = physical / 64
Q&A 实测锚点：EP64 时"每卡 7 个（不含共享 expert）"→ physical = 448
  → 若 E = 384，r = 64 = 每卡恰好 1 个冗余（自洽解）
权重显存：每卡 7 个 expert × 2 矩阵 × L 个 MoE 层（FP4：见 Day 4 的 per-32 MXFP4 打包）
共享 expert 单独复制（每卡 1 份，不参与 EPLB）
```

**第二步：EPLB 迁移前后的负载差异。** 玩具版（3 rank、6 physical、2 step 统计，跑一遍 `_batch_greedy_indices` 的目标）：

```
初始（无 EPLB，均匀放置）：expert 每副本负载（step1/step2）
  rank0: e0=100/100, e1=10/10      → 峰值 100/100，Σmax = 200
  rank1: e2=60/60,  e3=60/60      → 峰值 120/120，Σmax = 240  ← 最慢
  rank2: e4=30/30,  e5=30/30      → 峰值 60/60
moe_lb = max/mean = 120 / 90 = 1.33
EPLB 第一步（加冗余副本给最热的 e0）：score = 负载/副本数，e0 加一副 → 50/副本
  重排后（贪心最小化 Σ_step max_rank）：
  rank0: e0, e0' = 50+50 = 100 → 100/100
  rank1: e2, e3      = 120      → 120/120   仍最慢，但已无更优放置（容量 2/卡）
  rank2: e1, e4, e5?（容量限制下近似）
→ 生产实测的收敛点就是 moe_lb ≈ 1.09（EP64）：贪心 + 冗余把 1.3~2.0 的原始倾斜压到 9% 以内
```

（`Σ_step max_rank load` 正是 `_batch_greedy_indices` 的目标函数；生产窗口 1000 步 × 每 step 每 expert 计数，张量化贪心在 GPU 上一次跑完。）

**第三步：zigzag 手排**（3.2 节的 sq=56 例子）：算出 4 个 rank 的 token 数（均 14）、Σ 位置（均 385）、`ke−ks`（ratio=2 时 = (sk−sq+local_idx+1)//2，压缩 key 数也均衡）。再把 sq 改成 50，观察块 7 只有 1 个 token 时 rank 0 的失衡（70 vs 385）——**边界截断是 zigzag 均衡的唯一误差源**，也是 `_MIRROR_SEQS` 测试用例 `[4096, 4095]` 的构造意图。

### 实验 2：对照 `prepare_cp_mirror` 的测试矩阵

`_CP_MIRROR_TEST_CASES` 的 `short_seqs`（一半序列 < 2·cp_size）与 `[1, 0, 7, 129, 0]`（含零长序列）用例——验证"短序列也切、零长序列不出块"的边界。**读 `yoco_cp_boundary_fused_ref` 的三返回值**（contribs / gathered_reverse / contrib_pad_len），把 Day 3 的"边界重分布"叙述对到变量。

### 实验 3（有硬件）：DP rank 的输出组织

```bash
python examples/offline_inference.py --ep 2 --dp 8
```

观察每个 DP rank 打印自己的 `DP {rank}; INPUT: ...` 行（输出按 rank 分散在各进程 stdout）；再对照 `tests/models/offline_decode_ppl.json` 的 key 结构（`platform → workload → dp_rank`）——**PPL 基线按 DP rank 分 key 正是因为各 rank 的请求互不相交、采样流独立**（attention DP 的直接后果）。

---

## 九、面试高频问题

**Q1："Attention DP + balanced CP" 与 vLLM 的 TP/PP/DP 有什么本质区别？为什么 MLA 适合 attention DP？**

A：本质区别是**切分对象**：vLLM 的 TP 切 head 维（每层 2 次 all-reduce）、PP 切层（P2P 激活）、DP 切请求副本；DeepInfer 的 attention 只切请求——每 rank 全量权重、独立 batch，数据面零通信，DP 组只做调度面投票（gloo CPU 组的 all_reduce）。MLA 适合的原因：KV 是 latent（MQA、576 元素/token），一条请求的 KV 比 GQA 小一到两个量级，TP 切 KV 的显存收益消失而 all-reduce 成本仍在；每卡全量 attention 权重的代价可控（权重大头在 MoE，由 EP 分摊）。attention DP 不是孤立的：#DP=#EP，MoE 层用 all-to-all 把各 DP 副本的 token 汇到 expert 所在卡——"attention 免费、MoE 聚合"是组合拓扑。PP 同样不存在：PD 分离（PROMPT/COMPLETION 引擎）取代了它的角色。

**Q2：balanced CP 的 balanced 体现在哪？**

A：三层。① token 数均衡：每序列切 2P 块、每 rank 固定拿 2 块（第 r 块 + 倒数第 r+1 块），任意长度下 rank 间差 ≤ 块长，短序列照样切；② attention 工作量均衡（zigzag 本质）：causal 下块 b 的成本 ∝ b+1，镜像配对满足 (b+1)+(2P−b)=2P+1 恒定——手排 sq=56、P=4 可验证每 rank 的 Σ query 位置都是 385；V4.1 的实际度量是 `prepare_cp_mirror` 输出的 ke−ks（压缩 key 数，per-row）；③ 静态形状均衡：每 rank 的 mirror 缓冲同长（Σ 2·ps），缺 token 填 -1 用 valid_token_mask 屏蔽——形状固定对 CUDA graph 友好。`compute_seqlens_k_zos` 补的线索是 decode 侧：per-row 的 seqlens_q/k 在 ZOS 下要按上一步接受数修正，CP 的账本与 ZOS 的账本是同一套 per-row 量。

**Q3：EPLB 的触发条件与迁移粒度？**

A：触发：步进计数达到 `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL`（引擎默认 0=关；examples 30；生产 prefill 1000 / decode 10000 步），统计窗口默认 100 步且必须 ≤ interval；首次重排提前到 1/4 周期（纠正冷启动失衡）；NORMAL 引擎可选 prefill 后立即重排。粒度：**physical expert 槽位级、逐层独立**——策略先给热点逻辑 expert 分冗余副本（每轮 argmin"加副本后的最大每副本负载"，1e-4 扰动防并列），再用贪心把所有物理槽位放到 rank（最小化 Σ_step max_rank load，即 moe_lb 的目标函数）；迁移时机在权重搬运完成后即热切换（gate 直接读新的 logical_to_physical 表，无 barrier）。机内迁移走 symm_mem IPC 镜像（零通信原语，只建一层权重的镜像逐层刷新），跨机走 NCCL batched P2P 按 rank 差分桶。

**Q4：为什么 DeepInfer 的 EP 组要横跨 DP/CP 结构？**

A：EP 组是 world 内连续 ep_size 个 rank，天然包含多个引擎的 driver 与 CP worker。这是设计而非巧合：attention DP 的各副本互不相交，但 MoE 需要"聚合"——EP 组把大家的 token 通过 all-to-all 集中到 expert 所在的卡上，让 grouped GEMM 吃到大 batch。decode 的 Wide-EP（EP64=world）就是这个形状的极限：64 个 attention 副本共享一个 EP 域，每 step 每卡只读 1/64 的 expert 权重。副作用是 EP 组内步型必须一致——所以调度器要在 DP 组（= EP 组的 driver 子集）投票。

**Q5：Mega MoE、EPLB intranode shuffle、NCCL env 这些决策的共同依据是什么？**

A：硬件拓扑。`PG.is_within_nvl_domain` 用 `deep_ep.get_physical_domain_size` 判断组是否在 NVLink 域内（NPU 恒 True）——它决定 Mega MoE（域内一次调用融合 gate/dispatch/GEMM/combine）vs 经典 dispatch/combine、EPLB 的 IPC 镜像 vs P2P。`deep_cluster` 把这层探测独立成包（GPU 的 PCI/NVLink 拓扑、NUMA 节点、IB 设备），worker 初始化按 `bind_numa` 绑 CPU 与内存落位。**并行策略是拓扑的函数**，不是静态配置。

**Q6（开放题）：如果给你 128 卡但 NVLink 域只有 72 卡，DeepInfer 会怎么跑？**

A：EP 组必须放进拓扑约束里考虑。若坚持 EP128：跨 NVLink 域，Mega MoE 不可用（is_within_nvl_domain=False），走 DeepEP dispatch/combine（RDMA/IB），每 step all-to-all 跨域带宽成为瓶颈，EPLB 迁移也退化为 P2P；更合理的是 EP72 之类的域内配置（或按 deep_cluster 探测的物理域大小取整），牺牲一点切分规整性换域内融合路径。这正是 Q&A 说 decode 用"64–128 卡超节点"、benchmark 也强调"each within one scale-up domain"的原因——**EP 的上限由 scale-up 域决定，不是由总卡数决定**。

---

## 十、今日总结（三分钟电梯版）

> DeepInfer 的并行是"一个 world、四种切分、零 TP/PP"：**Attention DP** 切请求（MLA 的 latent KV 小到不值得切，每 rank 全量权重独立 batch，数据面零通信，DP 组只做调度投票）；**balanced CP** 切 prefill 序列（zigzag 镜像：每 rank 拿第 r 块+倒数第 r+1 块，(b+1)+(2P−b)=2P+1 保证 attention 工作量精确均衡，mirror 缓冲同长对 graph 友好；通信是 DeepEP BucketBuffer 的惰性 all_gather + gather 重排，控制面是 shm 对象广播 + NCCL 张量广播）；**MoE EP** 切 expert（连续切分横跨 DP/CP，NVL 域内 Mega MoE 一次调用、域外 dispatch/combine 且共享 expert 与通信重叠，通信 SM 数自动限流）；**EPLB** 是 EP 的运行时重排（窗口统计 → 两阶段贪心：副本分配+槽位放置，目标即 moe_lb → 机内 symm_mem IPC 镜像/跨机 P2P 迁移 → 权重就位即热切换）。所有拓扑相关决策（Mega、IPC、NUMA 绑定）由 deep_cluster 的探测驱动——并行策略是拓扑的函数。与 vLLM 的本质分歧：vLLM 通用（TP/PP/DP 组合覆盖任意模型），DeepInfer 专用（为 MLA+MoE 的 V4.1 找到"attention 免费、MoE 聚合"的极限拓扑）。

---

## 十一、今日自测题

1. world=12、dp=6、cp=2 时，DP 组、CP 组、EP=4 的 EP 组各包含哪些 rank？driver 是谁？
2. MLA 的哪两个特性让 attention DP 优于 TP？（提示：`num_key_value_heads`、latent 维度）
3. zigzag 配对为什么是 (r, 2P−1−r) 而不是 (r, P−1−r)？后者均衡吗？
4. `prepare_cp_mirror` 输出的四个张量分别是什么？哪个是"per-row 工作量"的度量？
5. CP 的控制面广播为什么分两个通道（shm + NCCL）？各传什么？
6. `compute_seqlens_k_zos` 的修正公式是什么？它与 CP 有什么关系？
7. EPLB 引擎默认开吗？首次重排为什么提前？窗口大小有什么约束？
8. `_batch_greedy_indices` 的目标函数是什么？它与 `moe_lb` 是什么关系？
9. EPLB 的两种 shuffle 各自的适用条件？机内路径为什么"零通信原语"？
10. `allow_multiple_reduction=False` 在什么组合下被强制？为什么？

<details>
<summary>参考答案（先自测再展开）</summary>

1. rank = replica·(dp·cp) + d·cp + c 布局（本例无 replica 维：world=dp×cp=12）。DP 组 = 同 c 跨 d：{0,2,4,6,8,10}（c=0，driver 们）与 {1,3,5,7,9,11}（c=1，slaves）。CP 组 = 同 d：{0,1},{2,3},...,{10,11}。EP=4 连续切：{0,1,2,3},{4,5,6,7},{8,9,10,11}——横跨引擎与 CP slot。driver = 各引擎 cp_rank 0：0,2,4,6,8,10。
2. ① MQA（`num_key_value_heads == 1`，mla_attn.py 断言）——KV 无 head 维可切；② latent 压缩（kv_lora_rank + rope ≈ 576 元素/token）——每请求 KV 总量小一到两个数量级，TP 切 KV 的显存收益消失，all-reduce 成了纯开销。
3. (r, P−1−r) 时配对和为 (b+1)+(P−b)=P+1，看似也恒定——但那样每 rank 只有 1 块（P 块分给 P 个 rank），块粒度粗一倍（ps 翻倍），边界截断误差和短序列浪费更大；且 2P 块的配对让 mirror 布局每 rank 恰好两个连续段（头半+尾半），gather/all_gather 的分片更规整。均衡性两者等价，2P 版本在形状与局部性上更优。
4. `mirror_cp_index`（varlen→本 rank 镜像布局的 gather 索引，-1=padding）、`mirror_cp_reverse_index`（逆映射，还原原始顺序）、`ks`/`ke`（每行在镜像布局下的压缩 key 起止）——`ke − ks = (sk − sq + local_idx + 1) // ratio` 是 per-row attention 工作量的度量。
5. 对象（形状/dtype/标量/metadata 键）无法走 NCCL broadcast，走 shm 环形缓冲（Day 2 的 MessageQueue，pickle 进共享内存）；CUDA 张量走 NCCL broadcast（`broadcast_tensor_dict` 里 `TensorMetadata` 走前者、张量列表走后者）。两个通道解耦后，张量广播可以被捕获进 CUDA graph（Day 3 的 `input_metadata_device.broadcast()` 在图内）。
6. `seqlens_k += seqlens_q; seqlens_k −= nextn − prev_accepted[remap[i]]`（remap=-1 的新请求不减）。它是 ZOS 的 per-row KV 账本；CP 的均衡度量（seqlens_q/k）正是同一套 per-row 量——ZOS 修正必须与 CP 镜像逐 rank 一致地执行，这就是它作为 balanced CP"线索"的含义。
7. 默认关（`EXPERT_REARRANGEMENT_STEP_INTERVAL` 默认 0）。首次提前到 interval−interval/4：新部署的初始放置是盲目的，尽早纠正冷启动失衡。约束 `0 < window ≤ interval`（窗口必须能在一个重排周期内攒满，否则统计与触发错位）。
8. `minimize Σ_step max_rank Σ_{e∈rank} load(e, step)`——即所有历史 step 里"最慢 rank"的负载总和。`moe_lb = Σmax / Σmean`（benchmark 口径）正是这个目标的归一化形式：贪心直接最小化分子。
9. 机内：`is_within_nvl_domain(EP) and is_gpu`——symm_mem rendezvous 建 IPC 镜像，每个 rank 拿到全组 buffer 句柄后**直接读写对端显存**（NVLink 内 IPC 映射），不需要任何通信原语；镜像只建一层的量，逐层刷新复用。跨机/NPU：NCCL batched P2P（isend/irecv 按 rank 差分桶）。
10. `BATCH_INVARIANT=1 且 REARRANGEMENT_INTERVAL>0`。多副本归约（一个逻辑 expert 的多个副本各自算一部分再合并）的合并顺序不确定，会破坏 batch-invariant；EPLB 制造多副本，所以两者同开时强制单归约路径（userbuffer.py:79）。

</details>

---

## 十二、今日产出物

1. **并行策略矩阵图**（第七节 SVG，README 指定产出）：DP / CP / EP / ENP × 切分维度 × 通信原语 × 代码文件，含 8-rank 组覆盖图与两种数据流。
2. **手算记录**（实验 1）：EP64 切分（physical=448、每卡 7）、EPLB 迁移前后 moe_lb（1.33 → 1.09 量级）、zigzag 手排（sq=56 与 50 的对照）。
3. **差异清单 v1.4**（在 Day 4 的 21 条之上补充 3 条）：

| # | 维度 | vLLM V1 | DeepInfer | 源码证据 |
|---|---|---|---|---|
| 22 | attention 并行 | TP（head 切分，每层 2 次 all-reduce） | DP（请求切分，零数据通信 + 调度投票） | parallel_state.py、scheduler.hpp |
| 23 | 序列并行 | 无独立 CP（context 拆分依赖 TP） | balanced CP：zigzag 镜像 + 惰性 all_gather + 双通道广播 | prepare_cp_mirror.py、cp.py |
| 24 | expert 负载均衡 | 无（或靠 EP 路由的天然均衡） | EPLB 全链路：窗口统计→两阶段贪心→IPC/P2P 迁移→热切换 | eplb/ 三文件 |

**明日预告（Day 6）**：高级特性四连——DSpark 投机解码与 confidence-scheduled verification（`mtp/dflash.py`、`schedule_mtp_varlen` 全量、`mtp_info_pool`）、SWA bounded replay 的 140 手算（`yoco.py` 前向路径）、batch-invariant 与确定性采样（`random_sampling`/`sum_smaller_probs` 收口）、结构化输出（`structure_output/` 与 mask 施加点）。今天读过的 `logical_to_physical_map`、`fused_sample` 的 reject 分支、`compute_attn_info` 的展开都将在明天各归其位。

