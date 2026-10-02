# Day 2 · vLLM V1 架构 + 源码精读 + 首次压测

> **总时长**：6-8 小时（上午 2.5h + 下午 3h + 晚上 2h）
> **今日目标**：能画出 V1 引擎数据流图；说清 scheduler 每个 step 做什么；手上有第一组真实压测数据
> **产出物**：一张手绘 V1 数据流图 + 压测数据记录表

---

## 作息建议

| 时间 | 内容 | 时长 |
|---|---|---|
| 09:00-11:30 | 模块一：V1 架构总览 + 请求生命周期 | 2.5h |
| 14:00-15:30 | 模块二：scheduler.py 源码导读 | 1.5h |
| 15:45-17:00 | 模块三：kv_cache_manager.py 源码导读 | 1.25h |
| 19:30-21:30 | 模块四：起服务 + 压测实操 | 2h |
| 21:30-21:45 | 整理数据流图 + 自测 | 0.25h |

---

## 模块一：V1 架构总览（上午，2.5h）

### 1.1 为什么 V1 是多进程架构

V0 时代 Python 单进程跑所有事，GIL 导致 API 处理、调度、采样互相阻塞。V1 的核心改动：**把开销大的部分拆到独立进程，用 ZMQ 通信**。

```
客户端
  │
  ▼
┌─────────────────────────┐     ┌──────────────────────────┐
│  API Server 进程         │     │  EngineCore 进程（0..N）  │
│  - FastAPI / OpenAI 接口 │ ZMQ │  - Processor：tokenize    │
│  - 请求接收/流式返回      │◄───►│  - Scheduler：调度决策     │
│  - detokenize（输出处理） │     │  - ModelExecutor/Worker   │
└─────────────────────────┘     │  - ModelRunner：前向执行   │
                                └──────────────────────────┘
```

要点（面试常问"为什么拆进程"）：
1. **绕过 GIL**：tokenize/detokenize、HTTP 处理与调度/执行真并行
2. **隔离故障**：引擎崩溃不拖死 API 层
3. **为 data parallel 和 P/D 分离铺路**：多个 EngineCore 可以挂在一个入口后

### 1.2 一个请求的完整生命周期（必背，7 步）

1. **接入**：API server 收到 OpenAI 格式请求，校验参数
2. **预处理**：Processor 做 tokenize，构造 `Request` 对象，送入 EngineCore
3. **准入**：Scheduler 把请求放入 `waiting` 队列
4. **调度**：每个 engine step，scheduler 决定本 step 哪些请求跑、各跑多少 token，向 KV cache manager 申请 block
5. **执行**：ModelRunner 组 batch（prefill 块 + decode token 混排）→ 前向 → 采样
6. **输出**：新 token 经 ZMQ 回 API 进程 → detokenize → 流式推给客户端；请求状态更新（未完成的回 running）
7. **释放**：请求完成（EOS/长度上限/停止词）→ KV block 引用计数归零 → block 回收进 free 池

### 1.3 关键参数速查表（压测和面试都要用）

| 参数 | 作用 | 调整方向 |
|---|---|---|
| `max_num_batched_tokens` | 每 step 的 token 总预算 | 调大→prefill 快但 TPOT 抖；调小→相反 |
| `max_num_seqs` | running 队列容量上限 | 受 KV 显存约束，超了会触发抢占 |
| `max_model_len` | 单请求最大长度 | 直接决定 block table 大小 |
| `gpu_memory_utilization` | KV 池占显存比例 | 默认 0.9，OOM 时调低 |
| `enable_prefix_caching` | 前缀缓存 | V1 默认开 |
| `enable_chunked_prefill` | 长 prompt 切块 | V1 默认开 |

### ✅ 模块一检验

- 不看资料画出上面的进程结构图，并讲出拆进程的三个理由
- 口头复述请求 7 步生命周期（30 秒版）

---

## 模块二：scheduler.py 源码导读（下午，1.5h）

**文件**：`vllm/v1/core/scheduler.py`（不同版本行数有差异，按函数名找，不按行号）

### 阅读路线（按此顺序，别的先跳过）

**① `schedule()` 主方法（40 min，今天唯一要逐行读的部分）**

它回答一个问题：本 step 跑哪些请求、各跑多少 token。逻辑骨架：

```
1. 先遍历 running 队列（decode 优先保证）：
   - 每个 running 请求需要 1 个 token 的预算（+投机解码的草稿 token）
   - 调 kv_cache_manager.allocate_slots() 申请新 block
   - 申请失败 → 触发抢占：从队尾挑优先级最低的请求 preempt
     （把它的 block 释放，状态打回 waiting）
   - token budget 用尽 → 停止

2. 再遍历 waiting 队列（准入 prefill）：
   - 检查：KV 够吗？max_num_seqs 满了吗？LoRA/其他约束？
   - 计算本次给多少 prefill token：
     剩余 budget 够全量 → 全量 prefill
     不够 → chunked prefill：切块，本 step 只跑 budget 允许的一段
   - 命中 prefix cache 的部分直接从 prefill 长度中扣除

3. 输出 SchedulerOutput：本 step 的调度决策清单
```

**要边读边标注的三个关键问题**（面试会考）：
- 为什么先排 running 再排 waiting？（decode 的 SLO 是连续的 TPOT，饿死 decode 代价高于推迟新请求准入）
- 抢占挑谁？（running 队尾、最后进来的——类 LRU）
- chunked prefill 的块大小由谁决定？（剩余的 `max_num_batched_tokens` 预算）

**② `update_from_output()`（20 min）**：执行结果回流——追加新 token、判完成、释放 block、更新统计。抓住"请求完成后 block 怎么回收"这条线即可。

**③ `Request` 状态机（10 min）**：WAITING → RUNNING →（PREEMPTED → WAITING）→ FINISHED_*。

### 略读清单（知道存在即可，不展开）

`kv_cache_manager_interface`、`encoder_cache_manager`、结构化输出相关逻辑、async scheduling 的 lookahead 部分。

---

## 模块三：kv_cache_manager.py 源码导读（下午，1.25h）

**文件**：`vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/block_pool.py` + `vllm/v1/core/kv_cache_utils.py`

### 阅读路线

**① 数据结构三件套（30 min）**：
- `KVCacheBlock`：物理块——block_id、ref_cnt、block_hash
- `BlockPool`：free block 队列（`free_block_queue`），allocate/free 的 O(1) 操作
- `BlockTable`：每个请求的逻辑块 → 物理块映射（就是 Day 1 论文里的那张表的工程实现）

**② prefix caching 命中路径（30 min）**——重点：
- hash 怎么算：**链式 hash = f(父 block hash, 本块 token ids, 额外键)**——额外键包含多模态标识、LoRA id、cache_salt 等
- `hash_request_tokens()` → `find_longest_cache_hit()`：新请求进来时沿 block 边界找最长命中前缀
- 命中后做什么：ref_cnt++，prefill 长度扣除命中部分
- **为什么 hash 必须链式**？（防止"中间块相同但前缀不同"的误命中——前缀不同则注意力结果不同，KV 不可共享）

**③ `allocate_slots()`（15 min）**：为请求追加 n 个 token 需要的 block——算需要几个新块（考虑当前尾部块剩余空间）→ 从 free 池取 → 不够返回失败（触发 scheduler 抢占）。

### ⚡ 昇腾翻译提示

block pool 的 free 队列管理 ≈ 你在昇腾做的内存池/buffer 管理；链式 hash ≈ Merkle tree 的父链防篡改思想。读这段代码时你会非常快——这就是标准系统编程，自信一点。

---

## 模块四：起服务 + 压测实操（晚上，2h）

### 4.1 起服务（15 min）

```bash
# 推荐租卡：AutoDL / 各大云，单卡 A100-80G 或 H100 均可，镜像选 PyTorch 2.x + CUDA 12.x
pip install -U vllm

vllm serve Qwen/Qwen3-8B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

观察启动日志里两行关键信息：**KV cache 池能容纳多少 token**（`GPU KV cache size: xxx tokens`）和 **最大并发**（`Maximum concurrency for xxx tokens per request: xx`）——把这两行抄进笔记，这就是 Day 1 手算公式的实测验证，**算一遍和日志对数**（对不上就是学习机会）。

### 4.2 并发扫描压测（60 min）

```bash
# 并发 = 1 / 4 / 16 / 64 各跑一轮
vllm bench serve \
  --backend openai-chat --base-url http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --dataset-name sharegpt \
  --num-prompts 200 \
  --max-concurrency <1|4|16|64>
```

（老版本 vLLM 用 `python benchmarks/benchmark_serving.py`，参数相同）

### 4.3 数据记录表（照抄格式填写）

| 并发 | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | 总吞吐 (tok/s) | 单请求吞吐 |
|---|---|---|---|---|
| 1 | | | | |
| 4 | | | | |
| 16 | | | | |
| 64 | | | | |

**必须看出的三个规律**（对着数据自己解释，解释不了就回模块二查）：
1. 并发 1→16：总吞吐近似线性涨，TPOT 几乎不变（权重读取被摊销——Day 1 的 decode 公式）
2. 并发 64：TPOT 开始变差（预算竞争/算力饱和），TTFT 因排队显著恶化
3. 存在某个"甜点并发"：goodput（满足 SLO 的请求占比）最高

### 4.4 指标对照（30 min）

`vllm serve` 加 `--disable-log-stats false`，观察日志周期性输出的：`Running: x reqs, Waiting: y reqs, GPU KV cache usage: z%`——这就是 scheduler 状态的实时窗口，**把一组压测时刻的日志和曲线对应起来**。

---

## 今日产出

1. **V1 数据流图**（手绘拍照即可，要素齐全：两个进程、ZMQ、7 步生命周期、waiting/running 队列、block pool）
2. **压测数据表** + 三条规律的解释（写在本子上，Day 6 诊断树要用）
3. **源码笔记**：schedule() 的伪代码骨架（不超过 20 行）

## Day 2 收工自测清单

- [ ] 能画出 V1 双进程结构并讲清拆分理由
- [ ] 能默写 schedule() 的两段式逻辑（先 running 后 waiting）
- [ ] 能回答：抢占发生在什么函数里、挑谁、被抢占请求的 KV 去哪了
- [ ] 能回答：prefix caching 的 hash 为什么必须链式
- [ ] 手上有真实压测曲线，且能用 Day 1 的公式解释曲线形状
- [ ] 启动日志的 KV cache size 和你手算的对上了
