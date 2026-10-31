# 第 3 周：vLLM V1 源码精读（下）——KV 管理与执行（扩展版）

> 本文档是根目录《八周按天打卡计划》第 3 周（Day 15-21）的逐天扩展。
> **本周目标**：吃透 KV cache 管理与模型执行层，开始同步动手写 mini 引擎（项目 B）。
> **前置依赖**（W1/W2 产出，开始前自检）：
> - 能手算任意模型的 KV cache 显存与并发上限（W1 Day 2）
> - 讲得清 Scheduler 的 waiting/running 队列、token budget、chunked prefill（W2 Day 10-11）
> - 有一个能跑的 vLLM 服务环境 + `vllm bench serve` 压测脚本（W1 Day 6）
>
> **版本纪律**：全程以 vLLM V1 架构为准（V0 已删除，不要读旧代码/旧博客）。vLLM 迭代快，本文提到的类名/文件位置以你 `pip install` 的版本为准——**每周开始时先 `git log --oneline -5`（或看 PyPI 版本号）记录你的基线版本**，类名若有出入，以"职责"对齐而不是死抠名字。
> **本周期望投入**：每天 2-4 小时；Day 20/21 可能需要 4 小时以上（写代码）。

---

## 0. 本周知识地图

先建立全局视图，再逐天下钻。

### 0.1 一个 decode step 的完整数据流（本周各天的位置）

```
请求到达（W2 已学）
    │
    ▼
Scheduler.schedule()                          ── W2 已学
    │  决定本步调度哪些请求、每个请求多少 token
    ▼
KVCacheManager.allocate_slots()               ── Day 15/16（本周核心①）
    │  prefix 命中检查 → 分配新 block → 更新 block table
    ▼
SchedulerOutput（block table 增量、token ids、位置信息）
    │
    ▼
GPUModelRunner.execute_model()                ── Day 17/18/19（本周核心②）
    │  ├─ 构建 AttentionMetadata（slot_mapping / block_table / seq_lens）
    │  ├─ Attention 后端写 KV + 算 attention（Day 17）
    │  └─ CUDA Graph replay / torch.compile（Day 18）
    ▼
采样 → 输出 → 下一轮调度与执行重叠             ── Day 19
```

### 0.2 源码目录速查表（读代码前打印贴墙上）

| 路径 | 内容 | 对应天 |
|---|---|---|
| `vllm/v1/core/kv_cache_manager.py` | KVCacheManager：请求视角的 KV 分配入口 | D15/16 |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`、`FreeBlockQueue`、hash 工具、KV cache 布局工具 | D15/16 |
| `vllm/v1/core/block_pool.py` | `BlockPool`：物理块池 + prefix caching 索引 | D15/16 |
| `vllm/v1/core/scheduler.py`（新版在 `v1/core/sched/`） | 调度器，调用 KVCacheManager 的上游 | 复习 |
| `vllm/attention/layer.py` | `Attention` 层：模型代码看到的统一接口 | D17 |
| `vllm/attention/registry.py` | 后端注册表（enum → builder） | D17 |
| `vllm/attention/backends/flash_attn.py` 等 | 各后端 Impl + MetadataBuilder | D17 |
| `vllm/platforms/interface.py` + `cuda.py` | 平台声明支持哪些后端、如何选择 | D17 |
| `vllm/v1/worker/gpu_model_runner.py` | 执行层核心：input 构建、CUDA Graph、采样 | D17/18/19 |
| `vllm/v1/worker/gpu_worker.py` | Worker：显存 profiling、KV cache 初始化 | D15 |
| `vllm/v1/engine/core.py` | EngineCore 主循环（async scheduling 在此体现） | D19 |

**读源码技巧**（贯穿本周）：不要从第一行读到最后一行。每个文件先看 `class` 定义和公开方法签名，猜职责 → 再挑 1-2 个方法读实现验证猜想。配合 debug：`python -m pdb`/`breakpoint()` 打在目标函数，跑一次单请求生成，观察调用栈与实参。

---

## Day 15：KV Cache Manager（一）——block pool 与物理内存管理

### 目标

讲清三个问题：① KV cache 的物理内存是怎么组织的；② 一个请求怎么拿到 block、怎么归还；③ 系统怎么知道"还剩多少 KV 显存"。

### 核心概念：三层数据结构

KV cache 管理本质是**操作系统虚拟内存分页**的翻版，先建立这张对照表（面试可直接用）：

| OS 虚拟内存 | vLLM V1 | 数据结构 |
|---|---|---|
| 物理页帧 | KV block（固定 token 数） | `KVCacheBlock`（`block_id` 唯一） |
| 页表 | block table（每请求 → block id 列表） | `KVCacheBlocks`（request_id → blocks） |
| 空闲页链表 | free block 队列 | `FreeBlockQueue`（双向链表） |
| 页帧分配器 | block 池 | `BlockPool` |
| 缺页/换出 | preemption、prefix 驱逐 | 调度器 + `FreeBlockQueue` 头部 |
| 页共享 + COW | prefix caching + copy-on-write | `ref_cnt` + hash 链（Day 16） |

三层职责划分：

1. **`KVCacheManager`**（请求视角）：对外只暴露 `allocate_slots / free / get_computed_blocks` 这类语义接口；内部持有 `BlockPool` 和每请求的 block table。
2. **`BlockPool`**（全局物理视角）：管理所有 `KVCacheBlock`；prefix caching 开启时同时维护 `cached_block: dict[BlockHash, KVCacheBlock]` 索引（Day 16 细讲）。
3. **`FreeBlockQueue`**（可分配序）：双向链表。**关键洞察：这个队列同时就是驱逐优先级队列**——空闲块从头部被取走，被释放的块回到尾部，所以"最老的空闲块"最先被复用，天然 LRU。

### 关键数据结构与伪代码

`KVCacheBlock`（`kv_cache_utils.py`）的字段值得逐个理解：

```python
class KVCacheBlock:
    block_id: int
    ref_cnt: int              # 被多少请求引用（prefix 共享的来源）
    block_hash: Optional[BlockHash]  # None = 私有块；非 None = 缓存的可共享块
    token_ids: list[int]      # 该块存的 token（hash 校验/驱逐后复用判断用）
    prev_block / next_block   # FreeBlockQueue 的链表指针
```

`KVCacheManager.allocate_slots()` 的主干逻辑（读源码时对照）：

```python
def allocate_slots(request, num_tokens):
    # 1. 已有 block 中还能装多少（本请求自己最后一个未满块）
    num_new_tokens = num_tokens - num_cached_tokens_in_last_block
    # 2. 计算需要几个新块（向上取整）
    num_new_blocks = ceil(num_new_tokens / block_size)
    # 3. 不够则抛出 NoFreeBlocksError → 上游 Scheduler 触发 preemption（W2 Day 12）
    # 4. 从 BlockPool 取块，登记到该请求的 block table
    # 5. prefix caching 开启时：对新满块计算 hash 并入库（Day 16）
```

一个请求的 KV 生命周期：

```
prefill:   allocate_slots(prompt_len)        → 一次性拿 prompt 所需块
decode:    allocate_slots(1)（每步）          → 大多数步复用已有块，块满时才 +1
结束/抢占:  free(request)                     → ref_cnt-1，归零则回 FreeBlockQueue 尾部
```

### 物理内存：KV cache tensor 长什么样

- 每个 KV cache tensor 形如 `[num_blocks, block_size, num_kv_heads, head_dim]`（K 和 V 各一，或拼在 head 维），**dtype = `kv_cache_dtype`**（默认跟随模型，可设 fp8——W4 Day 23 专题）。
- 布局有两种流派：所有层共享一个大池（block-major，`[num_blocks, ...]`，每层用同 block_id 的不同偏移）vs 每层独立 tensor（layer-major）。vLLM 有工具函数按模型结构（是否 hybrid/sliding window）自动选择，还会把"同组 attention 层"打包成 `KVCacheGroup`。
- **读代码任务**：在 `gpu_model_runner.py` 里找 `initialize_kv_cache` / kv cache tensor 创建处，画出你的版本最终创建的 tensor 形状。

**容量怎么定**：启动时 Worker 做一次 profile run（模拟 `max_num_batched_tokens` 的满载 forward），得到"扣除权重/激活/临时 buffer 后的可用显存"，再算 `num_gpu_blocks = 可用显存 / 每 block 字节数`。启动日志里直接打印了这个数——**拿日志数字和你 W1 Day 2 的手算对一遍**。

### 手算验证（今天必做）

Qwen3-8B（36 层，GQA 8 个 KV head，head_dim=128，BF16 权重 + BF16 KV，`block_size=16`）部署在 H100 80G：

- 每 token KV 显存 = `2 × 36 × 8 × 128 × 2B = 147456B ≈ 144KB`
- 每 block 显存 = 144KB × 16 = **2.25MB**
- 权重约 16GB，系统预留 + 激活后按 ~55GB 可用算：`55GB / 2.25MB ≈ 24500 blocks ≈ 39 万 token`
- 即：并发 100 个请求、平均 4K 上下文时 KV 占用约 40 万 token——刚好在临界点，会触发 preemption 的量级。

对照你实际启动日志的 `# GPU blocks` 验证。算错的常见原因：GQA 用了 `num_attention_heads` 而不是 `num_key_value_heads`（W1 Day 2 提过的坑）。

### 源码走读路线（建议 2 小时）

1. `kv_cache_utils.py`：`KVCacheBlock` → `FreeBlockQueue`（重点看 `pop_head/push_tail` 维护链表的细节）
2. `block_pool.py`：`allocate_block / free_block`，重点验证：**`free` 一个 `block_hash != None` 的块时，它是否真的从 `cached_block` 里删除？**（答案决定 Day 16 的驱逐语义）
3. `kv_cache_manager.py`：`allocate_slots` 全流程 + `free` / `update`（chunked prefill 的块追加走这里）
4. 断点实验：单请求跑起来，在 `allocate_slots` 打断点，看实参 `num_tokens` 与分配到的 block 数。

### 验证点（读代码时自问，答不出说明没读懂）

1. `ref_cnt > 0` 的块会出现在 `FreeBlockQueue` 里吗？什么情况下会？
2. preemption 时（W2 Day 12），被抢占请求的块是被 `free` 了还是被保留了？和 recompute 模式怎么配合？
3. `block_size` 默认多少？它同时是哪几样东西的单位？（提示：分配粒度、hash 粒度、**attention 算子 gather 的 tile 粒度**——Day 17 呼应）

### 与昇腾经验的连接

- block table = **间接寻址表**，和你做 tiling 时用偏移表管理多级 buffer（L1/L0A/L0B）是同构问题：都以"小表 + 大池"换取分配灵活性。
- KV cache 布局（哪个维度连续）直接决定 paged attention 算子搬运是否合并成大 burst——这在 NPU 上等价于 `block_size × head_dim × dtype` 是否构成高效 DMA 粒度（你对齐 32B/64B 的经验直接适用）。

### 产出

- [ ] 笔记《KV cache 三层数据结构图》（对照 OS 分页表）
- [ ] 手算 Qwen3-8B 的 block 数 vs 启动日志对照
- [ ] 3 个验证点的答案

---

## Day 16：KV Cache Manager（二）——prefix caching

### 目标

讲清：① 前缀复用怎么判定（hash 链）；② 共享块怎么不被写坏（COW + 引用计数）；③ 缓存什么时候被驱逐（LRU 藏在哪）；④ 做实验拿到第一组 hit rate 数据。

### hash 链机制：为什么 hash 必须包含父块 hash

每个**满块**的 hash 计算为：

```
block_hash = hash(parent_block_hash, tuple(token_ids_of_this_block), extra_keys)
```

- `parent_block_hash` 参与运算 → hash 天然编码了"从根到这里的完整路径"。两个请求命中同一 block，意味着**之前的整条前缀链都相同**，可以整段复用。
- `extra_keys` 至少包含：多租户 `cache_salt`（防跨租户缓存泄漏，W5 Day 34 伏笔）、LoRA id（不同 adapter 的 KV 不能混）、多模态输入的 hash。
- **只有满块才缓存**：未满块内容还在变，hash 语义不稳定。请求结束后其最后一个未满块会被补算 hash 入库（读代码验证你版本的行为：找 `cache_partial_blocks` / 请求 finish 路径）。

命中路径（`get_computed_blocks`）：给定请求的 token ids，从头开始逐块算 hash 查 `cached_block`，沿链走到底。命中块 `ref_cnt += 1`，直接挂到该请求的 block table，**这些 token 不用重算 prefill** → TTFT 大幅下降。

### COW：共享的最后一块被写入时

场景：请求 A 和 B 共享系统提示词（共享块 ref_cnt=2）。B 开始生成，新 token 要写入共享链的最后一个块，而 A 还在用 → 不能直接写。

处理：`BlockPool.fork(last_block)`——B 分配一个新块继承 token_ids 与 hash 语义，原块留给 A（ref_cnt-1）。这就是 copy-on-write 的"写时复制"。

**读代码验证点（重要，别背错）**：fork 之后，**物理 KV 数据的拷贝发生在哪一层、什么时机**？是 CPU 侧立刻下发一次 device-to-device 拷贝，还是靠"新块本就未写过 KV、直接往里写"来避免拷贝？在 `block_pool.py` 的 fork 相关代码 + `gpu_model_runner.py` 里找证据。面试被追问 COW 时，能答出"拷贝（或不拷贝）的准确时机"是区分度。

### 驱逐：没有独立 evictor，驱逐 = 分配

这是 vLLM 设计里最优雅的一点：**prefix caching 的 LRU 驱逐不发生在"内存不够"的回调里，而是发生在正常分配路径上**。

- 请求结束 → 块归还（ref_cnt=0）→ 块**留在 `cached_block` 索引里**，同时进 `FreeBlockQueue` 尾部 → "evictable"状态
- 新请求分配 → 从 `FreeBlockQueue` 头部取块 → 取到的若是带 hash 的缓存块，从 `cached_block` 删除 → 最老前缀被"驱逐"
- 结论：**`FreeBlockQueue` 的顺序 = 驱逐优先级顺序**（Day 15 埋的伏笔在此回收）

### 实验：构造高重复前缀负载

```bash
# 1. 正常启动（V1 默认开启 prefix caching；老版本显式加 --enable-prefix-caching）
vllm serve Qwen/Qwen3-8B

# 2. 压测：固定长系统提示词（如 2-4K token 的文档摘要），变化用户问题
#    用你 W1 Day 6 的脚本改造：system prompt 相同、user 内容不同
# 3. 观察指标（/metrics）：
#    vllm:prefix_cache_hits / vllm:prefix_cache_queries   → hit rate
#    TTFT p50/p99 对比"无重复前缀"基线
```

预期：hit rate 接近 `共享前缀长度 / 总 prompt 长度`，TTFT 显著下降且与命中长度近似线性。

实验陷阱：
- 共享前缀长度最好 ≥ 数个 block_size（如 ≥256 token），否则命中收益被块粒度截断；
- 多次压测间前一轮已把前缀"焐热"，注意区分冷/热启动（先打一轮 warmup 再采集）；
- 想验证 salt：请求 header 加 `cache_salt` 不同值，hit rate 应归零。

### 思考题（面试高频）

1. prefix caching 为什么对"多轮对话续写"收益极高，对"每次独立问题"收益趋零？hit rate 的上限由什么决定？
2. hash 冲突怎么处理？（读代码：`cached_block` 命中后是否校验 `token_ids`？）
3. chunked prefill 把 prompt 切成多段，中间被打断的请求已算好的块怎么办？（W2 Day 11 × 本天的交叉点：未完成请求的块保留不释放，避免重算——找 `cache_unfinished` 相关路径）
4. 如果两个请求共享前缀但 `block_size=16`，前缀长 100，能共享几个块？（6 个满块，剩 4 token 不能共享——解释"块粒度截断"）

### 与昇腾经验的连接

你做 ASW 模板提升 L2 命中率的本质是**让重复访问的数据驻留高带宽层**；prefix caching 是同一思想在系统层的实现——"重复计算 → 命中即跳过"。面试可以主动把这层映射讲出来：算子层的 reuse 优化 vs 引擎层的复用优化，都是"以便宜资源换昂贵资源"。

### 产出

- [ ] hash 链 + COW + 驱逐三张时序/状态图（手画）
- [ ] 实验数据表：hit rate 与 TTFT 改善（≥3 组前缀长度梯度）
- [ ] 思考题答案

---

## Day 17：Attention 后端抽象

### 目标

讲清：① vLLM 怎么把 FlashAttention/FlashInfer/Triton 后端"插拔"进同一套模型代码；② 一次 attention 调用里 `slot_mapping` 和 `block_table` 分别解决什么问题；③ **如果给新硬件（比如昇腾）写后端，要实现哪些接口**——这是你 W6 项目 A 的直接前置。

### 抽象分层（自上而下）

```
模型代码                attn = Attention(num_heads, head_size, num_kv_heads, ...)
                          attn.forward(q, k, v, attn_metadata)     ← 模型只见这一层
                              │
platform 选择          vllm/platforms/cuda.py 声明支持的后端集合
                       （env VLLM_ATTENTION_BACKEND 可强制指定）
                              │
registry              vllm/attention/registry.py: enum → Builder 的注册表
                              │
AttentionBackend      静态能力描述：get_kv_cache_shape()（物理布局！）、
                       支持特性（prefix caching / MLA / sliding window…）
                              │
AttentionImpl         每层一个实例：forward = "写 KV + 算 attention"
                              │
MetadataBuilder       batch 级：SchedulerOutput → 该后端的 AttentionMetadata
```

要点：
- `Attention` 层（`layer.py`）是模型实现（如 `llama.py`）唯一 import 的东西——**后端对模型代码完全透明**，这就是"插拔"的实现方式。
- `AttentionType` 区分 DECODER / ENCODER / ENCODER_DECODER / PREFILL_ONLY（MTP draft 层用，W4 Day 25 呼应）。

### 一次 forward 里两个关键映射

**`slot_mapping`（写入侧）**：batch 中第 i 个 token 的 KV 要写到哪个物理槽位。

```
slot = block_id × block_size + 块内偏移
```

prefill 阶段模型计算出的 k/v，由 attention 后端按 slot_mapping scatter 进 kv_caches tensor。这个映射在 `gpu_model_runner.py` 构建 input 时算好。

**`block_table`（读取侧）**：attention 计算时，每个序列去哪读历史 KV。decode kernel 沿序列的 block 列表逐块 gather——**KV 永不搬移，靠间接寻址读非连续内存**，这是 PagedAttention 论文（W1 Day 4）在代码层的落点。

`CommonAttentionMetadata`（各后端共享部分）核心字段：`query_start_loc`（batch 内每序列 query 的起止，varlen 场景代替 padding）、`seq_lens`、`num_reqs`。

**读代码任务**：跳进 `flash_attn.py` 的 `AttentionImpl.forward`，分辨出：哪段把 k/v 写进 cache（slot_mapping 用法）、哪段分派到 prefill（varlen）/decode（paged）kernel、`unified_attention` 这类"一步式"接口怎么同时处理 prefill+decode 混合 batch（V1 chunked prefill 决定了混合 batch 是常态——W2 Day 11 的呼应）。

### 写作业：新硬件 backend 接入清单（对接你的昇腾经验）

如果让你给一块新 NPU 写 attention 后端（vllm-ascend 的 `AscendAttentionBackend` 就是这么来的），最小工作集：

1. **`AttentionBackend` 子类**：
   - `get_kv_cache_shape(...)`：决定这块硬件上 KV cache 的物理形状/布局——**这是你最有发言权的地方**：哪个维度连续能让你 NPU 的搬运合并成大 burst？`block_size` 要不要为了对齐约束调整？
2. **`AttentionImpl` 子类**：
   - `forward`：写 KV + 调你的 paged attention 算子（prefill 走 varlen、decode 走 paged gather；或统一 kernel）
3. **`AttentionMetadataBuilder` 子类**：把 `SchedulerOutput` + block table 翻译成你算子要的参数格式（如昇腾侧的 tiling 参数、block 索引数组）
4. **平台注册**：platform 类里声明支持 + 默认选择逻辑
5. **特性声明**：支持不支持 prefix caching / MLA / 特定精度，决定上层功能开关

把这份清单写进笔记——W6 选 vllm-ascend 贡献点时（Day 36），直接拿它当 checklist 扫 issue。

### 实验

```bash
# 对比不同后端（你的卡支持哪几个就跑哪几个）：
VLLM_ATTENTION_BACKEND=FLASH_ATTN  vllm serve ... 
VLLM_ATTENTION_BACKEND=FLASHINFER  vllm serve ...
VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve ...
# 各跑一次 vllm bench serve，记录 TPOT/TTFT；再跑一次长上下文（如 32K）看差距是否拉大
```

```python
# 代码级确认选择逻辑（比背文档可靠）：
from vllm.platforms import current_platform
# 起服务时打日志/断点，观察最终实例化的 AttentionImpl 类名
```

### 思考题

1. `block_size` 从 16 改成 8/32，分别影响哪四样东西？（hash 粒度、碎片、block table 大小、**decode kernel 每次 gather 的连续长度**）
2. 为什么 prefill 用 varlen（`query_start_loc`）而不是 padding 到最长序列？
3. backend 报"不支持某模型"时，常见原因是什么？（head_size / 注意力类型 / hybrid 结构不在支持列表）

### 产出

- [ ] 后端抽象分层图 + forward 调用链笔记（带源码行号）
- [ ] 《新硬件 backend 接入清单》一页（W6 直接复用）
- [ ] 后端对比实验数据表

---

## Day 18：CUDA Graph

### 目标

讲清：① decode 为什么必须用 CUDA Graph（launch 开销量化）；② capture 怎么做到"一次录制、反复换状态重放"；③ full vs piecewise 的取舍；④ 亲手测出 `-O` 各档位差距。

### 问题：decode 一步要 launch 多少 kernel

Llama/Qwen 类模型一个 decode step：每层约 5-10 个 kernel（rmsnorm / qkv GEMM / rope / attention / o_proj GEMM / residual add…）× 36 层 + 采样 ≈ **200-400 次 launch**。每次 launch CPU 侧开销约 3-10μs → **单步 CPU 开销 1-4ms**。而 batch=1 时 GPU 每步实际计算可能只有几 ms → **GPU 在等 CPU 发指令**。这就是 decode 阶段"kernel launch bound"的定量解释（也是 W1 Day 3 roofline 图上没有的一类瓶颈：**执行开销不在算也不在读，在指挥**）。

解法：CUDA Graph 把整步的所有 kernel 录成一张图，之后每步**一次 launch**。前提：图内一切静态。

### 三个静态化技巧（V1 的工程核心）

1. **Shape 静态 → bucket + padding**：不能为每个 batch size 录一张图。按尺寸集合（如 1,2,4,8,…,max_num_seqs）capture，实际 batch 向上取整 pad 到桶值。padding 的 token 走 dummy 路径，结果丢弃。
2. **数据可变 → 持久 buffer + 状态写入**：capture 期间用固定地址的输入 buffer（`state_dict`：token ids、positions、seq_lens、block_table、slot_mapping…）；**replay 前把新状态 memcpy 进这些 buffer，再 `graph.replay()`**——图引用的是地址，不是值。
3. **显存爆炸 → 共享 memory pool**：几十张图各自 capture 会各自占 workspace；V1 用 `torch.cuda.graph_pool_handle()` 让所有图共享一个 pool，capture 尺寸集合时显存增量大幅缩小。

**读代码任务**：`gpu_model_runner.py` 里找 CUDA Graph Runner：capture 的尺寸集合来自哪（`compilation_config.cudagraph_capture_sizes`）、replay 前写入了哪些状态、图捕获时用了什么 dummy 输入。

### full CG vs piecewise CG

| | full CUDA Graph | piecewise CUDA Graph |
|---|---|---|
| 覆盖范围 | 整个模型 forward 一张图 | 以 attention 为切分点，attention eager、其余子图捕获 |
| attention 处理 | 必须静态（bucket 化的 decode 可以） | eager 执行，天然支持动态 shape（varlen prefill） |
| 适用阶段 | decode（batch bucket 后形状稳定） | prefill + decode 通吃 |
| 与 torch.compile | 可独立使用 | 与 torch.compile 配合（在 `splitting_ops` 处切图） |
| 代价 | 桶多则 capture 慢/显存涨；桶少则 padding 浪费 | 切分点处仍有 eager 开销，优化上限低于 full |

**为什么 attention 是天然切分点**：它是 forward 中唯一 shape 随序列集合剧烈变化、且有复杂 metadata（block table）的算子；其余算子（GEMM/rmsnorm/rope）bucket 化容易。

**为什么 prefill 不用 full CG**：序列长度方差大 → 桶要么爆炸要么 padding 浪费大量计算；且 prefill 是 compute-bound，launch 开销占比本来就小——**优化要打在瓶颈上**（回到 W1 第一性原理）。

### 实验：`-O` 档位对比

```bash
# 同一模型、同一负载，四组（compilation level 语义随版本演进，先查你版本的官方文档确认映射）：
vllm serve Qwen/Qwen3-8B --enforce-eager        # 纯 eager，无图
vllm serve Qwen/Qwen3-8B -O0                    # 无编译
vllm serve Qwen/Qwen3-8B -O1                    # 默认：torch.compile + piecewise
vllm serve Qwen/Qwen3-8B -O3                    # 最激进：尝试 full graph（不支持则回退）
# 压测记录 TPOT p50/p99（重点看 decode），并记录启动时间（capture 是有成本的）
```

预期：`--enforce-eager` 与 `-O1`+ 的 TPOT 差距在小 batch 时最大（launch 占比高）；batch 大到一定程度后差距收窄（单 kernel 变大，launch 占比下降）——**画出"batch size × CG 加速比"曲线，就是一张完美的一性原理配图**。

进阶（可选）：nsys 抓 `--enforce-eager` vs `-O1` 各 10 秒，肉眼看 kernel 间隙密度变化。

### 与昇腾经验的连接

- NPU 上同样存在 Host 下发开销（算子 launch/acl 调用），昇腾侧的等价物是**整图模式下发/计算图下沉**；你的 Fixpipe 无 Queue 手工流水（绕过 TPipe/TQue 抽象、直接管硬事件）本质也是**消除框架与下发开销**——跨平台同一个问题：控制面慢于数据面时，合并控制流。
- bucket + padding ≈ 你 tiling 里的"对齐到整块，尾块单独处理"：用少量浪费换路径统一。

### 思考题（面试高频）

1. 为什么 decode 用 CUDA Graph 而 prefill 通常不用？（上面两段就是答案，练成 60 秒版本）
2. capture 为什么要一个 dummy/warmup forward？（static shape 锁定 + 显存分配稳定，避免 capture 期间真实分配）
3. bucket 怎么选？1,2,4,8…指数好还是 1,2,3,4…线性好？（显存/capture 时间 vs padding 计算浪费的权衡）
4. CUDA Graph 下 block_table 怎么变？（固定大小的持久 buffer，replay 前整表刷新）

### 产出

- [ ] 《decode 单步 launch 数 × 单 launch 开销》的估算笔记（用自己的模型算一遍）
- [ ] `-O` 档位 × batch size 的 TPOT 矩阵实验数据
- [ ] full/piecewise 对照表（可直接搬上面表格 + 你版本的补充）

---

## Day 19：Async scheduling 与 CPU 开销隐藏

### 目标

讲清：① 不重叠时 GPU bubble 从哪来（画时间线）；② V1 overlap/async scheduling 的机制；③ 用 nsys 亲手量出 bubble 与剩余瓶颈。

### 时间线：无重叠 vs 重叠

```
无重叠（同步调度）：
CPU:  [调度N][输入N]        [调度N+1][输入N+1]
GPU:                 [执行N]                  [执行N+1]
                            ↑ GPU 空转 = CPU 调度时间 = bubble

重叠（V1 目标）：
CPU:  [调度N][输入N] [调度N+1][输入N+1] [调度N+2]...
GPU:            [执行N........]   [执行N+1......]
                    ↑ CPU 准备 N+1 与 GPU 执行 N 并行，bubble 被吃掉
```

成立条件：**CPU 单步准备时间 < GPU 单步执行时间**。decode 步长典型 5-20ms（batch 大时）→ CPU 有几 ms 余量，可行；但极端场景（极小 batch、极快 kernel、复杂调度决策）CPU 重新变成瓶颈——这就是为什么"把 scheduler 本身做快"（数据结构、增量计算）仍是主题。

V1 的实现路线（读代码时抓这三件事）：
1. `EngineCore` 主循环（`v1/engine/core.py`）：调度与执行解耦，executor 的结果通过 **callback 异步回传**，scheduler 不必阻塞等待。
2. Async scheduling（vLLM 官方博客 *Zero-Overhead Scheduling in vLLM V1*，搜标题精读）：调度完全与上一步 GPU 执行并行；**代价是正确性约束**——调度器此刻看到的是"第 N 步之前"的状态，因此 preemption 这类"必须基于最新状态"的决策要特殊处理（典型做法：需要抢占时先让当前步执行完再处理）。
3. 输出侧：detokenize/流式返回也在独立环节异步化，不占调度线程。

**读代码验证点**：async scheduling 开/关时 `Scheduler.schedule()` 的输入状态差一个 step——找到那个"上一步输出回填状态"的调用（`update_from_output` 一族）。

### nsys 实验：找 bubble

```bash
# 1. 常规压测跑起来（比如并发 32）
# 2. 抓 profile：
nsys profile -o decode_step --duration 10 -t cuda,nvtx,osrt \
    vllm serve ...   # 或对已跑进程 attach；也可用 vllm 自带 --profile 开关
nsys stats decode_step.nsys-rep        # 看 kernel 时间线
nsys ui decode_step.nsys-rep           # GUI 里看 CPU 线程与 GPU 流的对齐
```

看什么：
- GPU 流上相邻 kernel 的 gap：集中出现在每步的**开头**（调度+输入构建+launch）还是**步中间**（单个 kernel 排队/同步）？
- CPU 线程时间线：主调度线程、Python 解释器执行 `schedule()` 的时段有多长？是否与 GPU 执行段对齐（重叠成功）？
- 采样/日志/请求处理是否在关键路径上抢 CPU（GIL 竞争是 vLLM 把 EngineCore 拆成独立进程的原因之一——W2 Day 8 伏笔回收）。

产出一张标注过的截图：`[bubble 长度] [CPU 段] [GPU 段]` 三段对照。

### 与昇腾经验的连接

这就是你做了三年的**生产者-消费者流水线**：CPU 调度 = MTE2 搬数据，GPU 执行 = Cube 计算，async scheduling = 你的 SetFlag/WaitFlag 事件驱动多 buffer 乒乓。区别只在粒度：你做的是指令级（μs），这里是系统级（ms），**方法论完全同构**——面试讲这条线是你最强的差异化叙事。

### 思考题

1. async scheduling 下为什么 preemption 变难？如果允许"基于过期状态抢占"会发生什么？
2. GPU step 变得很短（大 batch + CUDA Graph + 小模型）时，谁先成为新瓶颈？（CPU 调度 → 引擎全链路，此时"调度器算法复杂度"重新重要）
3. detokenization 放在哪个进程做，为什么？（W2 Day 8 的进程架构图上标出来）

### 产出

- [ ] 重叠/不重叠时间线图（手画，含 bubble 标注）
- [ ] nsys 截图 + bubble 量化结论（你的部署里 bubble 占步长的百分比）
- [ ] Zero-Overhead Scheduling 博文精读笔记

---

## Day 20：mini 引擎开工（项目 B · 第 1 天）——KV 池与块管理

> 项目 B 持续到 Day 21（本周）+ Day 27（W4 收尾）。**这是面试作品集里"我懂调度与 KV 管理"的实证**——读过源码和亲手实现是两个段位。

### 目标（今天）

纯 Python 实现：固定大小 block 的 KV 池 + block table + 引用计数，配单元测试。不追求性能，追求**语义正确**——正确性不变式就是 vLLM 那 三层结构的最小复刻。

### 建议工程结构

```
mini_vllm/
├── block.py         # KVCacheBlock / FreeBlockQueue / BlockPool
├── kv_cache.py      # BlockTable（request → blocks）+ 池容量记账
├── prefix_cache.py  # hash 链 + cached_blocks + COW（明日可选，Day 27 前完成即可）
├── scheduler.py     # 明天
├── executor.py      # 明天
└── tests/
    ├── test_block.py
    ├── test_prefix.py
    └── test_scheduler.py
```

### 核心骨架（照此实现，语义对齐 vLLM）

```python
@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    token_ids: list[int] = field(default_factory=list)
    block_hash: Optional[int] = None

class FreeBlockQueue:
    """双向链表；pop_head 分配 / push_tail 归还；队列序 = LRU 驱逐序"""

class BlockPool:
    def __init__(self, num_blocks: int, block_size: int): ...
    def allocate(self) -> KVCacheBlock: ...        # 从队头取；若队头是缓存块则驱逐（Day 16 语义）
    def free(self, block: KVCacheBlock) -> None:   # ref_cnt-1；归零才回队尾
        ...
    def fork(self, block: KVCacheBlock) -> KVCacheBlock:  # COW：共享块写入前分裂
        ...

class BlockTable:
    def append_slots(self, request_id: str, num_tokens: int) -> None: ...
    def get_block_ids(self, request_id: str) -> list[int]: ...
    def free_request(self, request_id: str) -> None: ...
```

### 必须通过的不变式（测试就是照这个写）

```
I1（守恒）: 任意时刻 free_queue 中块数 + Σ ref 计数占用块数 == num_blocks
I2（无重叠）: free_queue 中不存在 ref_cnt > 0 的块
I3（容量）: 任何时刻 Σ(每请求已分配 token 数) ≤ num_blocks × block_size
I4（COW 安全）: 两请求共享前缀后各自追加 token，互不覆盖
     —— 构造场景：A、B 共享 3 块后各自 decode，检查各自最后块内容独立
```

今天验收标准：`pytest tests/test_block.py tests/test_prefix.py` 全绿，且 I1-I4 各有对应测试用例。**别小看这步**——W4 Day 27 加 preemption 时，这些不变式就是你的安全网。

### 与昇腾/后端经验的连接（写进项目 README 的"设计说明"）

明确写出你的 BlockPool 与 vLLM V1 `BlockPool` 的字段级对照表（哪个方法对应哪个方法），以及"如果底层换成真实设备，slot 映射怎么变成 `block_id × block_size + offset`"——这段话在面试里就是"我读源码读进去了"的证据。

---

## Day 21（复盘日 + 项目）——continuous batching 调度器 + 全链路大图

### 目标（今天）

1. mini 引擎补上 **iteration 级 continuous batching 调度器**（waiting/running + token budget），跑通端到端 demo；
2. 复盘：画出 vLLM V1 完整数据流大图。

### 调度器骨架

```python
class Scheduler:
    def __init__(self, block_table, token_budget: int, max_seqs: int): ...
    def schedule(self) -> SchedulerOutput:
        # 1. running 优先：为每个 running 请求预留 1 token（decode）
        #    KV 不够 → 抛 NoFreeBlocks →（W4 加 preemption，本周先拒绝/排队）
        # 2. 剩余 token budget 从 waiting 按序收新请求（FCFS）
        # 3. 预算耗尽或队列空 → 返回本步调度结果
```

executor 今天可以是**模拟执行器**（每 token 延迟用 `time.sleep` 或纯计数模拟），重点是调度正确：

```
端到端 demo（打印每步）：
step 3: running=[r1,r2]  new_tokens=[1,1]  waiting=5  free_blocks=120
step 4: running=[r1,r2,r3] ...            # r3 完成 prefill 加入 running
```

验收场景（写成集成测试）：
1. 不同长度请求混合到达，running 集合动态进出（对比 static batching 的"等最长的"）；
2. KV 耗尽时新请求停在 waiting（本周版本），free 后自动放入；
3. token budget 限制单步 prefill 量（为 W4 chunked prefill 留口子）。

### 复盘产出 1：V1 完整数据流大图（本周最重要的一张图）

要素清单（画的时候逐个检查，缺一个回去补读）：

```
请求 → Processor(tokenize/构造 Request) → waiting queue
     → Scheduler.schedule()【token budget / max_seqs / KV 检查】
     → KVCacheManager.allocate_slots()【prefix 命中 → 新块分配 → block table】
     → SchedulerOutput{token_ids, block 增量, positions}
     → GPUModelRunner【AttentionMetadata: slot_mapping/block_table】
     → Attention Backend【写 KV(slot_mapping) + attention(block_table)】
     → 采样 → output processor（detokenize/流式）
     → （异步）下一轮调度与本次执行重叠
```

画完自测三连（不看笔记）：
1. 指着图讲 60 秒"一个 token 的旅程"；
2. 在图上标出 prefix caching、COW、LRU 驱逐分别发生在哪个环节；
3. 在图上标出 CUDA Graph 覆盖的区段与 async scheduling 重叠的两段。

### 复盘产出 2：本周知识点自测清单

不看书回答；答不上的标记为 W4 前补课项：

1. `FreeBlockQueue` 的顺序为什么天然就是 LRU 驱逐序？
2. block hash 为什么必须链式包含父块 hash？extra_keys 里有哪些？各防什么问题？
3. COW 的触发条件与拷贝时机（你版本源码里的准确答案）？
4. `slot_mapping` 与 `block_table` 分别在 attention 的哪一侧使用？
5. 给新硬件写 attention backend 的最小接口集（5 项清单背下来）？
6. decode 单步 launch 开销怎么估算？CUDA Graph 三个静态化技巧？
7. full vs piecewise CG 的取舍？attention 为什么是切分点？
8. async scheduling 成立的条件？对 preemption 的约束？
9. block_size 增大/减小各影响什么（至少 4 项）？
10. Qwen3-8B BF16 的每 block 显存？3 秒内答出 2.25MB 量级。

### 本周产出物清单（对齐总计划）

| 产出物 | 对应天 | 状态 |
|---|---|---|
| 《KV cache 三层数据结构图》+ 手算对照 | D15 | [ ] |
| hash 链/COW/驱逐三图 + prefix caching 实验数据表 | D16 | [ ] |
| attention 后端分层图 + 新硬件接入清单 + 后端对比数据 | D17 | [ ] |
| `-O` 档位 × batch size 实验矩阵 + launch 开销估算 | D18 | [ ] |
| nsys bubble 分析截图 + Zero-Overhead 博文笔记 | D19 | [ ] |
| mini 引擎：block/pool/COW + 不变式测试全绿 | D20 | [ ] |
| mini 引擎：continuous batching 端到端 demo | D21 | [ ] |
| V1 完整数据流大图（本周最重要） | D21 | [ ] |

### 常见坑

1. **读成 V0**：搜到的博客/教程若讲 `BlockSpaceManager`、`SequenceGroup`、swap 模式——那是 V0，立即关掉。V1 无 swap，preemption 只重算。
2. **版本漂移**：`Scheduler` 已从 `v1/core/scheduler.py` 向 `v1/core/sched/` 迁移、KVCacheManager 职责向 coordinator/manager 族拆分——**以职责定位代码，不要死记路径**。
3. prefix caching 实验忘 warmup，把冷启动当基线，结论反向。
4. CUDA Graph 实验混入 prefill 负载导致 decode 提升被稀释——分开测。
5. mini 引擎跳过测试直接写调度器，W4 加 preemption 时不变式崩了无从查起。

### 下周预告（衔接 W4）

量化专题会回到 Day 15 的 `kv_cache_dtype`（FP8 KV cache = 每 block 显存减半 → 可用 block 数翻倍的连锁反应）；投机解码会回到 Day 17 的 `AttentionType.PREFILL_ONLY` 与 Day 18 的"decode 步变短后 launch 开销占比上升"。**本周这三处伏笔先在笔记里标出来。**
