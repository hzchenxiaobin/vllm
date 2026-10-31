# Week 2：vLLM V1 源码精读（上）—— 调度链路

> **本周目标**：能完整讲清"一个请求从进入引擎到吐出 token"的全生命周期，落到源码级细节。
> **前置**：Week 1 已建立 prefill/decode 计算访存模型、KV cache 显存手算、SLO 指标体系。
> **版本基线**：本文以 2025 年主线版本为准（v0.9–0.11+，V1 已是唯一架构，V0 代码已删除）。标注"注："的细节在不同版本间有差异，阅读时以你 checkout 的代码为准。
> **源码路径约定**：仓库根指 vLLM 主仓（github.com/vllm-project/vllm）；历史版本中 `v1/core/scheduler.py` 已迁移为 `v1/core/sched/scheduler.py`，本文统一用新路径。

---

## 0. 环境准备（Day 8 前完成）

```bash
# 1) 安装运行时
pip install -U vllm

# 2) 拉源码用于阅读（与安装版本对齐，记录 commit）
git clone https://github.com/vllm-project/vllm
cd vllm && git log -1   # 把 commit hash 记进笔记

# 3) 定位已安装的 vllm 源码（改代码做实验用）
python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"

# 4) 起服务 + 确认指标端点（贯穿本周的实验载体）
vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.85 --max-model-len 32768
curl -s http://localhost:8000/metrics | head -n 20
```

建议把 vllm 以 editable 方式安装（`pip install -e .`），Day 13 要在 `schedule()` 里临时加打印。

---

## Day 8：V1 架构总览

### 8.1 为什么要重构出 V1（先看动机，再看结构）

V0 的问题（面试可作为开场叙事）：

1. **前后端耦合**：API server 与引擎逻辑在同一异步框架里，detokenize、请求管理等 CPU 工作与 GPU step 循环互相干扰，Python GIL 抖动直接体现在 ITL 上。
2. **双引擎实现**：`LLMEngine`（同步）与 `AsyncLLMEngine`（异步）两套代码长期分叉，维护成本高。
3. **调度器分阶段**：prefill/decode 交替的调度逻辑复杂，chunked prefill 是补丁式加入。
4. **prefix caching 非默认**：存在开销顾虑，默认关闭。

V1 的答案：**前后端分离 + 单一 EngineCore + 统一调度循环 + 默认 prefix caching**。官方博客给出的成绩：DeepSeek-R1 在 H100 上端到端 ~1.7x 提速。

### 8.2 进程架构图（自己照着画一遍，标注通信方式）

```
┌────────────────── 进程 P0：API Server（前端）──────────────────┐
│  FastAPI  (vllm/entrypoints/openai/api_server.py)              │
│    /v1/chat/completions   /v1/completions   /metrics           │
│        │                                                       │
│        ▼                                                       │
│  AsyncLLM  (vllm/v1/engine/__init__.py)                        │
│    ├── Processor        tokenize → 构建 Request                │
│    ├── Detokenizer      增量反 tokenize（新 token → 文本增量）  │
│    ├── OutputProcessor  finish 判定 / 流式输出 / 统计上报        │
│    └── EngineCoreClient (AsyncZMQ)                              │
└──────┬──────────────────────────────▲─────────────────────────┘
       │ add_request(Request)        │ EngineCoreOutputs（新 token ids）
       ▼                             │
┌────────────────── 进程 P1：EngineCore ──────────────────────────┐
│  EngineCore  (vllm/v1/engine/core.py)                           │
│    主循环：schedule() → execute_model() → update_from_output()  │
│    ├── Scheduler  (vllm/v1/core/sched/scheduler.py)  ← 本周主角 │
│    │     waiting: deque[Request] / running: list[Request]       │
│    │     KVCacheManager (v1/core/kv_cache_manager.py) ← W3 主角 │
│    └── Executor  (v1/executor/)                                 │
│          uniproc（单进程）/ multiproc（TP 多进程）/ ray 等        │
└──────┬──────────────────────────────▲─────────────────────────┘
       │ SchedulerOutput              │ ModelRunnerOutput
       ▼                              │（sampled ids 留在 GPU）
┌──────────── 进程 P2..Pn：Worker（TP 时每 rank 一个进程）─────────┐
│  Worker  (vllm/v1/worker/gpu_worker.py)                         │
│    └── ModelRunner  (vllm/v1/worker/gpu_model_runner.py)        │
│          ├── InputBatch：持久化输入 buffer，组 batch / 增删请求  │
│          ├── CUDA Graph capture & replay（Day 18 详解）          │
│          └── Attention Backend（FlashAttention/FlashInfer/Triton）│
└─────────────────────────────────────────────────────────────────┘
```

三个关键点（面试必答）：

1. **Processor 在前端进程（P0）**：tokenize 在发往 EngineCore 之前完成，跨进程传的是 token ids 而非文本，序列化开销小。
2. **Detokenizer/OutputProcessor 也在 P0**：重 CPU 的输出处理与引擎核心循环隔离，EngineCore 只管"算"。
3. **uniproc 模式下 Worker 是 P1 内的线程**；TP ≥ 2 默认 multiproc，每个 rank 一个 Worker 进程，由 Executor 管理消息广播。

### 8.3 跨进程通信：ZMQ + msgspec

- 前后端两条 IPC 通道：请求输入（add_request / abort / health）与输出回传（EngineCoreOutputs / stats）。
- 序列化用 msgspec（结构化 + msgpack，比 pickle/json 快且安全）。
- 设计意图：**EngineCore 的 step 循环不被前端的 Python 噪声干扰**；同时为"引擎核心独立部署"（P/D 分离、engine-only 进程）铺路——这是 Week 5 的伏笔。

### 8.4 V0 → V1 关键差异表（背下来）

| 维度 | V0 | V1 |
|---|---|---|
| 引擎抽象 | LLMEngine / AsyncLLMEngine 双实现 | 单一 EngineCore，前后端分离 |
| 进程模型 | API server 与引擎同进程 | AsyncLLM(P0) / EngineCore(P1) / Worker(P2..) |
| tokenize / detokenize | 引擎内 | 均在前端进程 |
| prefix caching | 默认关 | **默认开**，零开销设计 |
| 调度器 | prefill/decode 交替 + 补丁 | 统一单循环，chunked prefill 内建 |
| preemption | recompute / swap 可选 | **仅 recompute** |
| 默认 max_num_batched_tokens | 2048 | 8192（注：以 SchedulerConfig 为准） |
| 默认 max_num_seqs | 256 | 1024 |
| CUDA Graph | 有限 batch | 默认开、覆盖更大 batch |
| 调度与执行 | 串行 | async scheduling（CPU/GPU 重叠，Day 19） |

### 8.5 阅读材料与顺序

1. 官方博客 v1 alpha / v1.0 发布公告（blog.vllm.ai）——建立大图景。
2. 仓库 `docs/design/v1/` 目录下的设计文档（multiproc_executor、kv_cache_interface、spec_decode 等，以 main 分支实际文件为准）。
3. 对照上图通读三个入口文件（只看骨架，不逐行）：
   - `vllm/v1/engine/__init__.py`（AsyncLLM）
   - `vllm/v1/engine/core.py`（EngineCore 主循环）
   - `vllm/v1/worker/gpu_model_runner.py`（只看 `execute_model` 的骨架）

### 8.6 自检问题

- V1 为什么把 EngineCore 拆成独立进程？（至少说出 3 条）
- Processor 为什么放在前端而不是 EngineCore？
- 如果让你给新硬件接入 V1（对接你的昇腾背景），替换的是图中哪一层？（Worker/ModelRunner/Attention Backend——为 Day 17 预热）

### 8.7 产出物

- 手绘（或 draw.io）V1 进程架构图一张，标出：进程边界、ZMQ 通道、每步数据流方向。

---

## Day 9：请求入口链路

### 9.1 全链路调用栈（从 HTTP 到第一个 token）

```
POST /v1/chat/completions
 → OpenAIServingChat.chat_completion()          entrypoints/openai/serving_chat.py
   → AsyncLLM.generate()                        v1/engine/__init__.py
     → Processor.process_inputs()               v1/engine/processor.py
         - chat template 渲染 + tokenize（TokenizerGroup）
         - 构造 Request 对象（见 9.2）
     → EngineCoreClient.add_request(request)    ZMQ IPC + msgspec
 ════════════ 跨进程 ════════════
 → EngineCore.add_request()
     → Scheduler.add_request()
         req.status = WAITING；waiting.append(req)
 → EngineCore 主循环 step()
     → Scheduler.schedule() → SchedulerOutput
     → Executor.execute_model(scheduler_output)
         → Worker → ModelRunner（组 batch / CUDA Graph replay / forward / sample）
     → Scheduler.update_from_output() → [EngineCoreOutput]
 ════════════ 跨进程回传 ════════════
 → OutputProcessor.process_outputs()
     → Detokenizer.update()：增量 decode 出新增文本
     → finish 判定（stop string 在前端！见 9.4）
     → 写入该请求的 asyncio.Queue → SSE 流推给客户端
```

走读时拿笔标出：**哪些工作在 P0、哪些在 P1、什么数据跨了进程边界**。

### 9.2 Request 对象解剖（vllm/v1/request.py）

```python
# 节选示意，字段有删减
class RequestStatus(Enum):        # 实际是 IntEnum 位标志
    WAITING, RUNNING,
    FINISHED_STOPPED, FINISHED_LENGTH_CAPPED,
    FINISHED_ABORTED, FINISHED_IGNORED,   # prompt 超过 max_model_len
    PREEMPTED ...

@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]    # tokenize 后的输入
    params: SamplingParams         # temperature/top_p/max_tokens/stop...
    output_token_ids: list[int]    # 已生成 token，增量追加
    status: RequestStatus
    num_computed_tokens: int       # 关键：已完成 forward 的 token 数（含 cache 命中）
    _block_ids: list[int]          # 该请求的 KV block 表（Week 3 详解）
    # 多模态：mm_positions/mm_hashes；LoRA：lora_request；投机：spec_token_ids
```

`num_computed_tokens` 是调度器最重要的账本：prefill 是否完成、chunk 推进到哪、抢占后是否清零，全看它。

### 9.3 EngineCore 主循环（vllm/v1/engine/core.py，示意）

```python
async def run_busy_loop(self):
    while True:
        # 1) 非阻塞收取 ZMQ 新输入（add_request / abort）
        await self._maybe_poll_inputs()
        # 2) 没有未完成请求则挂起等待新输入（idle 路径）
        if not self.scheduler.has_unfinished_requests():
            ...  # await 输入事件
            continue
        # 3) 核心 step
        outputs = await self.step()      # schedule → execute → update
        # 4) 回传输出
        if outputs:
            await self.send_outputs(outputs)
```

`step()` 的三段式（**本周后面每天都围绕它展开**）：

```python
async def step(self):
    scheduler_output = self.scheduler.schedule()          # Day 10-12
    model_output = await self.model_executor.execute_model(scheduler_output)
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)                   # 记账 + finish 判定
    return engine_core_outputs
```

### 9.4 输出侧：为什么 stop string 在前端判

- **EOS / max_tokens / stop token id**：只需要 token id，EngineCore 在 `update_from_output` 里判，最早能判。
- **stop string**：需要 detokenize 后的**文本**，而 detokenize 被搬到了前端进程 → 只能在 OutputProcessor 判。
- 这带来一个隐蔽行为：stop string 命中时，EngineCore 可能已经多算了几步（多生成的 token 被丢弃）。面试聊数据流设计时这是很好的细节。

### 9.5 动手实验：日志跟踪一个请求（贯穿本周）

```bash
# 终端 1
vllm serve Qwen/Qwen3-8B --log-level info

# 终端 2：单请求
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","prompt":"用一句话介绍 vLLM","max_tokens":32}'
```

观察：启动日志里的进程/组件初始化顺序；请求进来后的 periodic 统计行（`Avg prompt throughput / Avg generation throughput / Running: n / Waiting: m`）。

进阶：`py-spy dump --pid <EngineCore 进程号>` 抓引擎进程栈，验证多进程结构（`ps aux | grep -i engine` 找 pid）。

### 9.6 自检问题

- 一个请求跨了几次进程边界？各传什么数据？
- `num_computed_tokens` 与 `output_token_ids` 分别由谁更新、何时更新？
- 请求在哪个进程被判定"结束"？stop string 呢？

### 9.7 产出物

- 一页请求生命周期走读笔记（含 9.1 调用栈 + 你的日志观察截图/摘录）。

---

## Day 10：Scheduler（一）—— 队列与 budget

### 10.1 数据结构（vllm/v1/core/sched/scheduler.py）

```python
class Scheduler:
    waiting: deque[Request]     # FCFS；优先级请求插队（注：priority 支持版本相关）
    running: list[Request]      # 已进入批的请求（decode + 未完成的 chunked prefill）
    max_num_batched_tokens: int # 单步 token 预算（V1 默认 8192）
    max_num_seqs: int           # 并发序列数上限（V1 默认 1024）
```

两个限流旋钮的分工（面试必考）：

| 参数 | 限制什么 | 影响 |
|---|---|---|
| `max_num_batched_tokens` | **单步 forward 的总 token 数**（prefill chunk + 全部 decode） | 单步耗时上限 → ITL 上限；prefill 吞吐 |
| `max_num_seqs` | 同时 in-flight 的请求数 | 并发上限 → KV 占用水位 → 抢占概率 |

### 10.2 schedule() 主流程（示意伪代码，主干保真、分支从简）

```python
def schedule(self) -> SchedulerOutput:
    # 0) waiting 队首检查：prompt 超 max_model_len → FINISHED_IGNORED

    # 1) 预算初始化
    rem_token_budget = self.max_num_batched_tokens

    # 2) 先调度 running（优先级高）
    for req in self.running:
        # decode 请求：1 token；未完成的 chunked prefill：剩余 prompt tokens
        num_new_tokens = 1 if req.is_decode else req.remaining_prompt
        num_new_tokens = min(num_new_tokens, rem_token_budget)   # → 切块
        new_blocks = self.kv_cache_manager.allocate_slots(req, num_new_tokens)
        if new_blocks is None:               # KV 不足 → 抢占（Day 12）
            self._preempt(req)               # free 全部 KV，回 waiting 队头
            continue
        rem_token_budget -= num_new_tokens
        # 记入本步调度：num_scheduled_tokens / 新 block ids

    # 3) 再调度 waiting（新 prefill，用剩余预算）
    while self.waiting and len(self.running) < self.max_num_seqs:
        req = self.waiting[0]
        num_new_tokens = min(req.prompt_len, rem_token_budget)   # → chunked prefill
        if num_new_tokens == 0 or allocate_slots(...) is None:
            break                             # 没预算 / 没 KV → 本步不接纳
        ...
        self.waiting.popleft()

    # 4) 汇总 SchedulerOutput：
    #    scheduled_new_reqs / scheduled_cached_reqs / num_scheduled_tokens
    #    / block_table 增量 / preempted_reqs
```

**读码顺序建议**：`schedule()` → `_get_new_token_count`（每个请求本步算几个 token）→ `SchedulerOutput` 数据类（v1/core/sched/output.py）→ `update_from_output`。

### 10.3 token budget 的记账规则

- 每个 running decode：本步消耗 1 token 预算。
- 每个 chunked prefill：本步消耗 min(剩余 prompt, 当前剩余预算)。
- budget 的本质：**把单步 forward 的算力/时长钉在一个上界**，从而钉住 ITL 上界（用 Week 1 的 FLOPs 手算验证：budget=8192 在 7B 模型 A100 上单步 ≈ 2×7e9×8192 ÷ ~150 TFLOPs ≈ 0.77s——这就是最坏 ITL）。
- 注：投机解码开启时 decode 一步消耗 draft_tokens+1；`long_prefill_token_factor` 控制空闲队列下长 prompt 是否整段放行，语义随版本演进，读你版本的 budget 分支代码确认。

### 10.4 KV 池大小如何确定（启动期）

`Worker.determine_num_available_blocks`：用 dummy 最大 batch 跑一次 profiling，量出 activation 峰值 →
`num_gpu_blocks = (gpu_memory_utilization × 总显存 − 权重 − activation 峰值) ÷ 每 token KV 字节`。
对照 Week 1 的手算公式，这里就是它的工程化落地。`--kv-cache-dtype fp8` 可让块数翻倍（Day 23 实验伏笔）。

### 10.5 自检问题

- 一步里可以同时有 prefill 和 decode 吗？谁优先？（可以；running 优先，decode 是 running 的主体）
- budget 剩 100、队首 prompt 8000，会发生什么？（切块 100 或不接纳，版本分支相关）
- `max_num_seqs` 和 KV 池哪个先成为瓶颈？怎么从指标分辨？

### 10.6 产出物

- `schedule()` 流程图一张（含 budget 记账与两个限流旋钮的位置）。

---

## Day 11：Scheduler（二）—— chunked prefill

### 11.1 问题：混跑干扰（用数字说话）

设 Qwen3-8B 在 A100，8B 参数 FP16 ≈ 15GB，HBM 2TB/s：

- 一个 decode step 的时延下界 ≈ 15GB ÷ 2TB/s ≈ **7.5ms**
- 一个 8192-token prefill ≈ 2×8e9×8192 ÷ 150TFLOPs ≈ **0.9s**

不开 chunked prefill：长 prompt 到来时，同批 decode 请求的下一个 token 要等 ~0.9s → **ITL 从 7.5ms 尖峰到 900ms（120 倍）**，流式体验崩坏。

开 chunked prefill（budget=2048）：单步 prefill 部分被钉在 ~0.22s 以内 → ITL 有界。

### 11.2 机制：切块与混排

- 长 prompt 按 token 预算切成多个 chunk，**跨多个 step 完成**；已算部分的 KV 正常写入 cache（append，不重复计算）。
- 同一步的组成 = 若干 decode（每个 1 token）+ 0~1 个 prefill chunk（新请求或续算）。
- 续算请求保持在 `running` 中，`num_computed_tokens` 逐 chunk 推进，直到等于 prompt 长度后转入 decode 阶段。
- KV 分配按 chunk 长度增量进行（partial 分配），block table 逐步 append。

### 11.3 计算量是否重复？（高频误区，务必想透）

不重复。chunk i 的 attention 只计算"chunk i 的 Q × 已有全部 KV"，新增 KV 只是追加。**总 FLOPs 与一次性 prefill 相同**，多付的只是每步固定开销（组 batch、kernel launch、调度）× chunk 数。这正好对应你在昇腾上"大矩阵分块流式搬运，L1 放不下就切"的经验：切块不增加计算量，增加的是控制开销。

### 11.4 一页总结《chunked prefill 的收益与代价》（直接产出物）

**收益**

1. ITL 有界：单步 token 数被 budget 钉死 → decode 不会被长 prefill 饿死（P99 ITL 大幅改善）。
2. 吞吐提升：prefill 与 decode 混排，单步 token 利用率满（无 prefill-only 空转段）；对 goodput（SLO 内吞吐）提升尤其明显。
3. 排队更公平：长 prompt 不霸占引擎，短请求 TTFT 可预测。

**代价**

1. TTFT 变长：长 prompt 要跨多步完成，负载高时明显。
2. 固定开销 ×chunk 数：每步组 batch/launch/调度开销重复支付，纯 prefill 吞吐略降。
3. 状态复杂度：partial prefill 的请求管理、KV 增量分配、与 prefix caching 的交互。

**budget 怎么设（调参速查）**

| 场景 | 建议 budget | 理由 |
|---|---|---|
| 流式交互（ITL 敏感） | 2048–4096 | 单步短，ITL 平滑 |
| 默认均衡 | 8192 | V1 默认 |
| 离线吞吐 | 16384+ | step 摊薄固定开销 |

### 11.5 自检问题

- 为什么 chunked prefill 降 TPOT 抖动而不降平均 TPOT？代价转嫁到了哪个指标？（TTFT）
- budget 设成 256 会怎样？（ITL 极平滑；prefill 极慢、TTFT 爆炸、吞吐崩）

---

## Day 12：Scheduler（三）—— preemption

### 12.1 触发条件

KV block 耗尽：某 running 请求本步需要新 block 而 `allocate_slots` 返回 None（free 队列空、可驱逐的 cached block 也用尽）。
典型诱因：`max_num_seqs` 过大 + 长输出请求堆积 → KV 增长超预期。这是 **admission control 失当**的信号，不是正常态。

### 12.2 V1 的处理流程（recompute 模式）

```
allocate_slots 返回 None
   → 选 victim：优先抢占最晚到达的 running 请求（保留老请求已投入的计算）
   → kv_cache_manager.free(该请求全部 blocks)   # prefix caching 下引用计数-1，
   →  状态复位：num_computed_tokens = 0          # block 可能转为 cached 保留
   →  output_token_ids 保留（作为"重算的 prompt"）
   → waiting.appendleft(req)                    # 队头！优先恢复，防饥饿
   → SchedulerOutput.preempted_reqs 记录，ModelRunner 下一步把它从 batch 摘除
重调度时：prompt + 已生成 tokens 全量重算（prefix cache 命中则近似免费）
```

注：victim 选择顺序、watermark（为 running 预留增长空间的分配水位线）细节随版本演进，读你 checkout 的 `schedule()` 抢占分支并记进笔记。

### 12.3 recompute vs swap（对照 V0，面试标准答案）

| | recompute | swap |
|---|---|---|
| 恢复成本 | 重算全部 token 的 FLOPs | KV 经 PCIe 搬回 |
| 额外资源 | 无 | CPU RAM + PCIe 带宽 |
| 实现复杂度 | 低（复用 prefill 路径） | 高（异步拷贝管理、双池） |
| prefix caching 下 | 命中缓存则近似免费 | 与缓存机制纠缠 |

手算对比（Qwen3-8B，1k token 上下文）：KV ≈ 56KB/token → 57MB，PCIe Gen4 ~20GB/s → swap 恢复 ~3ms；recompute ≈ 2×8e9×1024 ÷ 150TFLOPs ≈ **110ms**。swap 更快，但 V1 仍砍掉它：**默认 prefix caching 让被抢占请求的 block 大概率还在缓存里，recompute 代价大幅趋近 swap，而状态机和管理面简单一个量级**。

### 12.4 观测与调优

指标：`vllm:num_preemptions`（counter，持续增长 = KV 超配）。

| 症状 | 原因 | 调整 |
|---|---|---|
| preemptions 增长 + ITL 毛刺 | KV 不足 | `--max-num-seqs` 调小 / `--max-model-len` 调小 / `--gpu-memory-utilization` 提高 / KV FP8 |
| preemptions 增长 + queue time 高 | 并发超配 | 同上；必要时前置限流 |
| 无抢占但 TTFT 高 | prefill 拥塞 | budget 调大 / P/D 分离（W5） |

### 12.5 自检问题

- 抢占对客户端是什么表现？（请求变慢/流暂停，不失败）
- 被抢占请求为什么放 waiting 队头而不是队尾？（已投入计算最多 + 防饥饿震荡）
- 什么负载下抢占会反复发生（thrashing）？怎么破？

---

## Day 13：动手验证调度行为

### 13.1 实验一：长 prompt 洪峰 → 观察 chunked prefill

```bash
vllm serve Qwen/Qwen3-8B --max-num-batched-tokens 2048 --max-num-seqs 32

vllm bench serve --backend openai --model Qwen/Qwen3-8B \
  --dataset-name random --random-input-len 8192 --random-output-len 128 \
  --num-prompts 32 --request-rate 2 --percentile-metrics ttft,tpot,itl
```

对照组：`--max-num-batched-tokens 8192` / `16384` 各跑一遍。
预期：budget 越小 → ITL p99 越平滑、TTFT 越差。同时观察 INFO 日志的 periodic 统计行（Running/Waiting/prompt throughput）。

### 13.2 实验二：挤爆 KV → 观察 preemption

```bash
# 故意压缩 KV 池 + 放开并发
vllm serve Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.5 --max-model-len 16384 --max-num-seqs 128

vllm bench serve --backend openai --model Qwen/Qwen3-8B \
  --dataset-name random --random-input-len 12000 --random-output-len 512 \
  --num-prompts 64 --request-rate 4

# 另开终端持续观测
watch -n1 'curl -s localhost:8000/metrics | grep -E "num_preemptions|gpu_cache_usage|num_requests_waiting|num_requests_running"'
```

预期：`gpu_cache_usage_perc` 冲到 ~1.0 → `num_preemptions` 增长 → ITL 出现长毛刺（被抢占请求的输出暂停）。

进阶：在 `schedule()` 抢占分支加一行 `print`，把 preempt 的 request_id 与当时的 free block 数打出来，与 metrics 对时序。

### 13.3 指标清单（/metrics，名字以实际输出为准）

| 指标 | 含义 | 调度关联 |
|---|---|---|
| `vllm:num_requests_running` / `waiting` | 当前批/队列规模 | budget 与 max_num_seqs 的直接体现 |
| `vllm:request_queue_time_seconds` | 入队 → 首次被调度 | prefill 拥塞度 |
| `vllm:time_to_first_token_seconds` | TTFT 分布 | chunked prefill 代价面 |
| `vllm:time_per_output_token_seconds` | ITL/TPOT 分布 | budget 上界 + 抢占毛刺 |
| `vllm:num_preemptions` | 抢占累计次数 | KV 超配报警 |
| `vllm:gpu_cache_usage_perc` | KV 池占用率 | 抢占前兆 |

### 13.4 实验记录模板（产出物）

| # | 现象（数据/截图） | 源码机制（文件:行为） | 指标表现 |
|---|---|---|---|
| 1 | budget=2048 时长 prompt TTFT ↑ | scheduler.py: waiting 切块 | TTFT p50/p99 对比表 |
| 2 | KV 占满后请求输出暂停 N 秒 | 抢占 → recompute → 队头恢复 | num_preemptions + ITL 毛刺图 |

---

## Day 14：复盘日

### 14.1 请求在 scheduler 中的状态机（产出物，手画）

```
                     add_request()
                          │
                          ▼
                   ┌─────────────┐
      schedule():  │   WAITING   │  waiting: deque（FCFS）
    预算 & KV 满足 │  (排队中)   │
                   └─────────────┘
                     ▲         │
      被抢占          │         │ 本步被调度：分配 KV block、
   （释放全部 KV，    │         │ 记 num_scheduled_tokens，
    num_computed_    │         │ 新请求建 block table
    tokens 清零，    │         │
    回队头优先恢复）  │         ▼
                   ┌─────────────────────┐
                   │       RUNNING       │
                   │  prefill 阶段：      │  num_computed_tokens <
                   │  chunk 逐段推进 ────▶│  prompt_len（含 cache 命中抵扣）
                   │  decode 阶段：       │
                   │  每步 +1 token       │
                   └─────────────────────┘
                          │ finish 判定（EngineCore: EOS/长度/stop token；
                          │            前端: stop string）
                          ▼
     FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / FINISHED_ABORTED / FINISHED_IGNORED
                          │
                          ▼
                free 全部 KV blocks（引用计数-1）
```

### 14.2 自测推演：10 个请求、KV 只够 6 个（不看答案先推一遍）

设定：R1..R10 同时到达（编号即到达序），每个 prompt 8192 tokens、要求输出 512；
`max_num_batched_tokens = 16384`，`max_num_seqs = 64`；KV 池恰好容纳 6 个请求的满上下文（6×(8192+512)）。

| Step | 动作 | running | waiting | 说明 |
|---|---|---|---|---|
| 1 | 调度 R1、R2 完整 prefill（2×8192=16384 用满预算） | R1,R2 | R3..R10 | budget 钉死单步 token 数 |
| 2 | 调度 R3、R4 prefill | R1-R4 | R5..R10 | 注：真实代码中 running 的 decode 先扣预算（每个 1 token），剩余全给 prefill——两种理解都要会讲 |
| 3 | 调度 R5、R6 prefill | R1-R6 | R7..R10 | KV 池被 6 个请求的上下文占满 |
| 4..N | R1..R6 每步各 +1 token（6 tokens/步，预算大量闲置） | R1-R6 | R7..R10 | R7 想进：allocate_slots 失败（KV 不足）→ 留在 waiting。**预算够 ≠ KV 够** |
| N+k | R1 生成满 512 → FINISHED_LENGTH_CAPPED → free 其 KV | R2-R6 | R7..R10 | |
| N+k+1 | R7 prefill 进批 | R2-R7 | R8..R10 | 队头 FCFS 接替 |
| ... | 依次完成、依次补位 | ... | ... | 全部完成 |

抢占变体：若某步某 running 请求需要新 block 而池空 → 抢占最晚到达者（如 R6）→ 释放其全部 KV → R6 回 waiting **队头**（排在 R7 前）→ 下个空位先恢复 R6（重算 8192 + 已生成 tokens；prefix cache 命中则近似免费）。

**推演检查点**（口头能讲清才算过）：

- 每一步 batch 的组成（几个 decode / 几个 chunk / 各多少 token）；
- 为什么第 3 步后 R7 进不来（KV 约束 vs budget 约束的区别）；
- 抢占选谁、为什么、恢复时排哪、重算多少；
- 全程哪些指标会怎么动（running/waiting 数、gpu_cache_usage、num_preemptions、queue time）。

### 14.3 本周一图流（复盘产出）

请求 → tokenize(P0) → ZMQ → waiting → schedule(budget/KV/抢占) → SchedulerOutput → Executor → ModelRunner 组 batch → forward → sample → update_from_output(记账/finish) → EngineCoreOutputs → ZMQ → detokenize/stop-string(P0) → SSE。
每个箭头旁标注：所在进程、关键数据结构、失败分支（IGNORED/PREEMPTED/ABORTED）。

### 14.4 速查卡片（面试前 10 分钟看这个）

- step 三段式：`schedule() → execute_model() → update_from_output()`
- 两个旋钮：`max_num_batched_tokens`（单步 token 上界 → ITL 上界）、`max_num_seqs`（并发 → KV 水位）
- 调度优先级：running（decode/续算）> waiting（新 prefill）
- chunked prefill：FLOPs 不重复，多付固定开销；收益 ITL/goodput，代价 TTFT
- 抢占：KV 不足触发、victim 最晚到达、recompute-only、回 waiting 队头、prefix cache 兜底
- V1 三进程：P0 AsyncLLM（Processor/Detokenizer/OutputProcessor）、P1 EngineCore（Scheduler/KVManager）、P2+ Worker（ModelRunner/AttnBackend），ZMQ+msgspec 连接

---

## 附录 A：本周源码文件地图

| 文件 | 职责 | 优先级 |
|---|---|---|
| `vllm/v1/engine/__init__.py` | AsyncLLM 前端：generate 入口、请求状态管理 | 必读 |
| `vllm/v1/engine/processor.py` | tokenize、构造 Request | 必读 |
| `vllm/v1/engine/core.py` | EngineCore 主循环、step 三段式、async scheduling | 必读 |
| `vllm/v1/engine/output_processor.py` | finish 判定、流式输出、统计 | 必读 |
| `vllm/v1/engine/detokenizer.py` | 增量 detokenize | 浏览 |
| `vllm/v1/core/sched/scheduler.py` | 调度器（本周核心） | 精读 |
| `vllm/v1/core/sched/output.py` | SchedulerOutput / CachedRequestState | 精读 |
| `vllm/v1/request.py` | Request / RequestStatus | 精读 |
| `vllm/v1/core/kv_cache_manager.py` | KV 分配/释放接口（allocate_slots/free） | 只看接口 |
| `vllm/v1/worker/gpu_worker.py` | Worker：初始化、KV 池测量 | 浏览 |
| `vllm/v1/worker/gpu_model_runner.py` | 组 batch、执行 forward（W3 展开） | 只看骨架 |
| `vllm/v1/executor/multiproc_executor.py` | TP 多进程执行器 | 浏览 |
| `vllm/v1/metrics.py` | Prometheus 指标定义 | 配合 Day 13 |
| `vllm/config/scheduler.py`（新版拆分） | max_num_batched_tokens / max_num_seqs 默认值 | 查证 |

## 附录 B：面试高频问题速答（调度链路 10 问）

1. **V1 为什么拆进程？** GIL/CPU 噪声隔离（detokenize 等移出核心循环）、崩溃隔离、结构上支持引擎独立部署（P/D 分离铺路）。
2. **一个请求从 HTTP 到第一个 token 经过哪些组件？** FastAPI → AsyncLLM → Processor(tokenize) → ZMQ → EngineCore.add_request → Scheduler.schedule → ModelRunner.execute_model → update_from_output → ZMQ 回传 → Detokenizer/OutputProcessor → SSE。
3. **max_num_batched_tokens 与 max_num_seqs 的区别？** 前者限单步总 token 数（决定 step 时长与 ITL 上界），后者限并发序列数（决定 KV 水位与抢占风险）。
4. **chunked prefill 的收益与代价？** 收益：ITL 有界、吞吐/goodput 升；代价：TTFT 升、固定开销 ×chunk 数。FLOPs 不重复。
5. **为什么 V1 只保留 recompute 抢占？** 实现简单；prefix caching 默认开，被抢占请求重算大概率命中缓存，代价趋近 swap。
6. **抢占选谁做 victim？** 最晚到达的 running 请求——保护已投入计算最多的老请求。
7. **被抢占请求如何恢复？为什么放队头？** num_computed_tokens 清零、全部重算（含已生成 tokens）；队头优先恢复，防饥饿震荡。
8. **stop string 为什么在前端判？** 需要 detokenize 后的文本，而 detokenize 在 P0；代价是可能多算几步。
9. **async scheduling 怎么重叠 CPU/GPU？** 第 N 步 GPU 执行时，CPU 并行算第 N+1 步的 schedule 与组 batch；采样的 token ids 留在 GPU 上由下一步直接消费，CPU 只需知道长度。要求 CUDA Graph（避免需要采样结果的 CPU 回退路径）。
10. **如何从指标判断调度问题？** TTFT 升/ITL 稳 → prefill 拥塞（调 budget 或 P/D 分离）；ITL 毛刺 → budget 过大或抢占；num_preemptions 增长 → KV 超配，降 max_num_seqs/max_model-len 或 KV 量化。

## 附录 C：常见误区清单

- chunked prefill 会重复计算 attention → **错**，总 FLOPs 相同，多的只是每步固定开销。
- budget 越大吞吐越高 → 不一定，ITL p99 恶化后 goodput 反降。
- 抢占 = 请求失败 → **错**，对客户端只是变慢，不报错。
- V1 默认关闭 prefix caching → **错**，V1 默认开启且零开销设计。
- prefill 和 decode 是两个交替的引擎阶段 → V1 是统一单循环混排。
- max_num_seqs 只影响吞吐不影响延迟 → **错**，过大会推高 KV 水位触发抢占，ITL 毛刺。

## 附录 D：昇腾经验 → vLLM 调度链路 概念映射表

| 昇腾经验 | vLLM 对应 | 共同本质 |
|---|---|---|
| WeightQuantBatchMatmul 的 baseM/baseN 分块搜优 | chunked prefill 的 token budget 切块 | 控制单次执行 workload 规模，平衡吞吐与延迟 |
| ASW 蛇形滑窗提升 L2 命中率 | prefix caching 的 block 复用 | 局部性挖掘，降低重复访存 |
| 无 Queue 手工流水（SetFlag/WaitFlag 事件驱动） | async scheduling 的 CPU/GPU 重叠 | 用异步/事件隐藏搬运与控制开销 |
| Fixpipe 流水掩盖 Cube/Vector 延迟 | CUDA Graph 消除 launch 开销 | 固化控制流，让硬件连续执行 |
| CalRebalanceBlock 三道剪枝条件 | 调度器 budget/KV/watermark 可行性检查 | 资源约束下的可行性剪枝 |

---

## 打卡记录

| Day | 完成打勾 | 一句话收获 |
|---|---|---|
| Day 8 | [ ] | |
| Day 9 | [ ] | |
| Day 10 | [ ] | |
| Day 11 | [ ] | |
| Day 12 | [ ] | |
| Day 13 | [ ] | |
| Day 14 | [ ] | |
