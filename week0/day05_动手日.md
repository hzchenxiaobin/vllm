# Day 5 · 动手日——mini 调度器（选项 1）/ vllm-ascend 导读（选项 2）

> **总时长**：6-7 小时，只做一个选项
> **今日目标**：把前三天"读来的知识"变成"写过的代码"——白板环节最大的底气来源
> **默认选选项 1**；只有昨天/今天状态差、或环境实在不通时选选项 2

---

## 选项 1：mini 推理调度器（推荐，约 250 行）

### 作息建议

| 时间 | 内容 | 时长 |
|---|---|---|
| 09:00-10:30 | 数据结构与 KV 池 | 1.5h |
| 10:45-12:30 | 调度器主循环 | 1.75h |
| 14:00-16:00 | 演示场景 + 抢占 + 日志可视化 | 2h |
| 16:15-17:00 | 整理"我与 vLLM 的差距"清单 + 白板演练 | 0.75h |

### 架构设计（照这个写，别自己重新设计）

```
┌────────────────────────────────────────────┐
│  Scheduler                                   │
│   waiting: deque[Request]                    │
│   running: list[Request]                     │
│   + schedule() -> dict[req_id, n_tokens]     │
├────────────────────────────────────────────┤
│  BlockPool                                   │
│   blocks: list[Block]  (Block: id, ref_cnt)  │
│   free_ids: deque[int]                       │
│   + alloc() / free() / can_alloc(n)          │
├────────────────────────────────────────────┤
│  Request                                     │
│   prompt_len, output_len, generated          │
│   block_table: list[int]  (逻辑块→物理块)     │
│   status: WAITING/RUNNING/FINISHED           │
└────────────────────────────────────────────┘
```

### 核心代码骨架（关键部分直接给，填空的自己写）

```python
BLOCK_SIZE = 16          # 每块 token 数，和 vLLM 默认一致
NUM_BLOCKS = 96          # 故意调小，才能演示抢占

class BlockPool:
    def __init__(self, num_blocks):
        self.ref_cnt = [0] * num_blocks
        self.free_ids = deque(range(num_blocks))

    def alloc(self):
        bid = self.free_ids.popleft()
        self.ref_cnt[bid] = 1
        return bid

    def free(self, bid):
        self.ref_cnt[bid] -= 1
        if self.ref_cnt[bid] == 0:
            self.free_ids.append(bid)

    def can_alloc(self, n):
        return len(self.free_ids) >= n

class Request:
    def __init__(self, rid, prompt_len, output_len):
        self.rid, self.prompt_len, self.output_len = rid, prompt_len, output_len
        self.generated = 0
        self.block_table = []          # 每个元素是物理 block_id
        self.status = "WAITING"

    @property
    def total_tokens(self):
        return self.prompt_len + self.generated

    def blocks_needed(self):
        """当前 token 总量需要多少个 block"""
        return -(-self.total_tokens // BLOCK_SIZE)   # 向上取整

class Scheduler:
    def __init__(self, pool, max_batched_tokens=512):
        self.pool, self.budget = pool, max_batched_tokens
        self.waiting, self.running = deque(), []

    def schedule(self):
        plan = {}
        budget = self.budget
        # ---- 第一段：保 running（decode 优先）----
        for req in list(self.running):
            need_blocks = req.blocks_needed() - len(req.block_table)
            if not self.pool.can_alloc(need_blocks):
                self.__preempt()        # 你写：抢占队尾请求
                if not self.pool.can_alloc(need_blocks):
                    plan = {k: v for k, v in plan.items()}  # 本 step 放弃它
                    continue
            # 分配新 block、追加 block_table、记 1 个 decode token
            ...
            plan[req.rid] = 1
            budget -= 1
            if budget <= 0:
                return plan
        # ---- 第二段：准入 waiting（prefill，支持切块）----
        while self.waiting and budget > 0:
            req = self.waiting[0]
            n = min(req.prompt_len, budget)      # chunked prefill 的雏形
            blocks = -(-n // BLOCK_SIZE)
            if not self.pool.can_alloc(blocks):
                break                            # 显存不够，停止准入
            ...
            plan[req.rid] = n
            budget -= n
            self.running.append(self.waiting.popleft())
        return plan
```

**留给你自己写的三个空**（这是今天的思考量所在）：
1. `__preempt()`：从 running 队尾挑一个，释放它的 block、清 block_table、状态打回 waiting **队首还是队尾？**（想清楚各自后果——vLLM 放回 waiting 队首附近以保证公平，想想为什么）
2. 请求完成时的 block 释放路径
3. 每步执行后的状态推进（generated += plan[rid]，判完成）

### 演示场景（必须跑出来的三个）

```python
# 场景 1：continuous batching 的价值
# 4 个请求输出长度 10/100/200/500 同时到达
# 打印每个 step 的 running 集合 → 观察短请求提前退出、新请求补位

# 场景 2：抢占
# NUM_BLOCKS 调到只够 6 条并发，灌入 10 个请求
# 打印抢占事件发生时刻、被抢请求 id、它后续何时恢复

# 场景 3：对照实验
# 同一负载跑 static batching（凑齐 4 个才跑、全完成才放下一批）
# vs continuous batching，对比总耗时和平均完成时间
```

日志格式建议：`step=37 | running=[r1:decode, r3:prefill 256/4096, r7:decode] | free_blocks=12 | preempted=1`

### 白板讲法（写完代码后对着镜子练两遍）

> "我手写过一个 250 行的简化 serving 引擎：block 池 + 引用计数 + iteration 级调度。vLLM 比它复杂主要在五处——①真实模型的前向与采样；②chunked prefill 的切块和混排；③prefix caching 的链式 hash 与 COW；④多进程拆分 + async scheduling 隐藏 CPU 开销；⑤多 GPU worker 的分布式执行。但调度决策的核心两段式——先保 decode 再准入 prefill、显存不足触发抢占——我的实现和 scheduler.py 是同构的。"

---

## 选项 2：vllm-ascend attention backend 导读（备选）

### 阅读路线（按序）

1. **插件注册入口**：vllm-ascend 如何通过 `vllm.platform_plugins` 注册 AscendPlatform——理解 vLLM 的硬件抽象层设计
2. **Attention 三件套接口**（vLLM 侧 `vllm/attention/backends/abstract.py`）：
   - `AttentionBackend`：内核选择/配置
   - `AttentionMetadata`：本 step 的调度元数据（seq_lens、block_tables、slot_mapping）
   - `AttentionImpl`：真正的 forward——paged KV 读写 + attention 计算
3. **vllm-ascend 的实现**：找到它的 attention backend 文件，看 metadata 如何构造、Ascend kernel 如何被调用
4. **对照思考**：如果让你优化它的 decode attention kernel，你会从哪几个维度下手？（结合你的 tiling/bound 方法论写出 3 条）

### 产出：《vLLM 硬件后端接入指南》一页

结构：插件机制 → 必须实现的接口清单 → metadata 数据流 → 一个优化切入点清单。

> 面试差异化话术："我不只会用 vLLM，我研究过怎么把新硬件接进 vLLM——attention backend 需要实现 X/Y/Z 三个抽象，Ascend 侧的现状是……可优化点是……"

---

## 今日产出

- 选项 1：可运行的 mini 调度器 + 三个演示场景的日志截图/文本 + "我与 vLLM 的差距"五条清单
- 选项 2：《vLLM 硬件后端接入指南》一页

## Day 5 收工自测清单

- [ ] 不看代码，白板能画出 mini 引擎架构图并讲 5 分钟
- [ ] 能解释抢占时请求放回 waiting 的位置选择及理由
- [ ] 能说出自己的实现和 vLLM 的五处差距（每处都能展开一句）
- [ ] （选项 2）能列出 attention backend 三件套接口及各自职责
