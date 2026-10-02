# Day 3 · 核心机制四连（面试主战场）

> **总时长**：7-8 小时
> **今日目标**：四个机制全部达到"四段式"表达水平（原理 → 解决什么问题 → trade-off → 什么时候失效），这是专家答案和背诵答案的分水岭
> **产出物**：四张机制卡片（每张 ≤ 半页 A4）+ 一组对比实验数据

---

## 作息建议

| 时间 | 内容 | 时长 |
|---|---|---|
| 09:00-10:45 | 机制一：Continuous Batching | 1.75h |
| 11:00-12:30 | 机制二：Chunked Prefill | 1.5h |
| 14:00-15:45 | 机制三：Prefix Caching | 1.75h |
| 16:00-17:30 | 机制四：CUDA Graph | 1.5h |
| 19:30-21:00 | 实验：chunked prefill 开关对比 | 1.5h |
| 21:00-21:30 | 写四张卡片 | 0.5h |

---

## 机制一：Continuous Batching（上午）

### 原理

传统 **static batching**：一批请求凑齐 → 一起跑 → **全部完成才放下一批**。问题：同一批里输出长度差异巨大，短请求早早结束，它占的算力/显存空转到批次里最长的请求结束。

**Continuous batching（iteration-level scheduling）**：把调度粒度从"批次"降到"迭代步"——**每生成一个 token 后重新决定 batch 组成**：完成的请求立即退出释放 KV，waiting 里的新请求立即插入。

### 解决什么问题

一个数值例子（背下来，面试可直接讲）：

> 4 个请求，输出长度 10/100/200/500，单步 20ms。
> Static：batch 跑 500 步 = 10s，其中请求 1 有 490 步在空等 → 有效利用率 = (10+100+200+500)/(4×500) = **40.5%**
> Continuous：请求 1 第 10 步退出，立刻补新请求 → 理想情况下 GPU 始终满载，吞吐接近 static 的 2-3 倍，且短请求时延大幅下降

### Trade-off

- 调度器本身成为每步开销（CPU）→ 所以 V1 做 async scheduling、多进程拆分
- batch 动态变化 → 需要 PagedAttention 支持非连续 KV（两个机制是配套的）
- 准入太激进 → KV 显存耗尽 → 抢占（preemption），被抢占请求 TTFT 暴涨

### 什么时候失效/不适用

- 离线批量推理（吞吐量唯一目标、无 SLO）：static batching 配合排序装箱反而更优
- 单请求场景：无 batch 可调

---

## 机制二：Chunked Prefill（上午）

### 原理

一个 32K token 的 prompt 如果整段 prefill，这个 step 独占 budget 几百毫秒，**同 batch 的 decode 请求全部卡顿**（TPOT 尖峰）。Chunked prefill 把长 prompt 切成块（块大小 = 本 step 剩余 token budget），和 decode 请求**混排在同一个 step**：每个 step 先保证 running 请求的 decode token，剩余 budget 分给 prefill 切块。

### 解决什么问题

- TPOT 平滑：decode 不再被长 prompt "顶住"
- 预算利用率：prefill 的 compute-bound 特性和 decode 的 memory-bound 特性**互补**——同一个 step 里两者混合，算力和带宽同时被用上（这是根因，务必能讲：混合 batch 的算术强度介于两者之间，比纯 decode 更接近 roofline 拐点）

### Trade-off

- 长 prompt 的 TTFT 略微变长（被切了多块，每块都要等调度）
- `max_num_batched_tokens` 成为关键调参：调大 → TTFT 好、TPOT 抖；调小 → 相反。经验起点 2048-8192，按 SLO 调
- 调度逻辑复杂度上升

### 什么时候失效

- prompt 普遍很短（如分类、embedding 类负载）：切块无收益
- 极致 TTFT 场景（首 token 就是 SLO 核心）：可能需要反向调参，优先 prefill

---

## 机制三：Prefix Caching（下午）

### 原理

不同请求共享相同前缀（系统提示词、多轮对话历史、few-shot 模板）→ 前缀的 KV 只需算一次。实现依赖 Day 2 读的**链式 block hash**：`hash = f(父块hash, 本块token, cache_salt等)`，命中后 ref_cnt++，prefill 跳过命中段。

### 解决什么问题

- 共享系统提示词场景：TTFT 中 prefill 部分近似归零（直接命中）
- 多轮对话：每轮只需 prefill 增量部分
- 显存：共享前缀物理块只存一份（配合 COW，分叉时只复制尾块）

### 关键细节（面试追问点）

1. **命中粒度是 block**（默认 16 token）：前缀必须对齐 block 边界才命中——差一个 token 错位，后面全不命中
2. **多租户隔离**：`cache_salt` 参与 hash，不同租户的相同前缀不共享——防止侧信道探测"别人有没有问过这句话"
3. **命中率的收益是乘法**：命中率 50% ≠ TTFT 减半，因为省掉的是 prefill 计算部分，排队和 decode 不变

### 什么时候失效

- 前缀高度个性化（每个请求 prompt 完全不同）：命中率≈0，纯开销（hash 计算很小，可忽略）
- 前缀长度 < 一个 block

---

## 机制四：CUDA Graph（下午）

### 原理

GPU kernel 每次 launch 有固定 CPU 开销（~几 µs/个）。decode 一步涉及几十上百个小 kernel（每个 layer 的 QKV/attention/FFN/norm...），单步总时延可能只有几 ms → **launch 开销占比可达两位数百分比**，且 CPU 提交速度跟不上 GPU 执行速度（CPU-bound）。

CUDA Graph：把一串 kernel 的调用序列**录制**成图，之后一次 launch 重放整图 → CPU 开销从 O(kernel数) 降到 O(1)。

**难点**：图要求形状固定。decode 的 batch size 是变化的 → 解决方案：**按 batch size 分桶抓图**（1/2/4/8/16/...），运行时 padding 到最近的桶。

**Piecewise CUDA Graph**：attention 部分的 KV 长度是动态的、不好入图 → 把模型切成"attention 之外的部分抓图 + attention 动态执行"的拼接。这是 V1 的默认形态（对应 compilation level 的 PIECEWISE）。

### 解决什么问题

- decode 的 CPU launch 开销 → TPOT 下降，尤其小 batch 时（GPU 等 CPU 的间隙被消除）
- 配合 async scheduling，CPU 调度时间与 GPU 执行重叠

### Trade-off

- 抓图占显存（每个桶一份中间激活 buffer）→ `gpu_memory_utilization` 里的一部分
- 启动时 capture 耗时（分钟级）
- 动态形状场景（prefill 变长）不适用 → 所以 **prefill 不抓图**（这就是高频题第 3 题的标准答案核心）

### 什么时候失效

- prefill 主导的服务（形状多变、单步时延长，launch 开销占比可忽略）
- 超长上下文 attention 主导时，图外部分占比大，收益打折

---

## 晚上实验：chunked prefill 开关对比（1.5h）

```bash
# 基线：默认（chunked prefill 开）
vllm serve Qwen/Qwen3-8B --max-num-batched-tokens 8192

# 对照：关闭 chunked prefill
vllm serve Qwen/Qwen3-8B --no-enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

压测负载要**混合长短 prompt**（ShareGPT 即可，或自造：一半 4K 长 prompt + 一半 200 token 短 prompt，并发 16）。

**记录与预期**：

| 指标 | chunked prefill 开 | 关 | 你的解释 |
|---|---|---|---|
| TPOT p99 | 平稳 | 出现尖峰 | 长 prompt 独占 step |
| 长 prompt TTFT | 略高 | 略低 | 被切块排队 |
| 总吞吐 | 高 | 低 | 混合 batch 的算术强度互补 |

**如果数据不符合预期**：检查负载是否真的有长 prompt——这是比数据本身更好的面试素材（"我设计实验时踩过的坑"）。

---

## 今日产出：四张机制卡片模板

每张卡严格四段式，半页以内：

```
【机制名】Continuous Batching
① 原理一句话：调度粒度从批次降到迭代步，每步重组 batch
② 解决什么：static batching 的空转（给 40.5% 利用率那个例子）
③ Trade-off：调度开销/抢占风险/依赖 PagedAttention
④ 失效场景：离线批量、单请求
⑤ 一句话昇腾挂钩（可选）：相当于把"整批等齐再跑"改成"流水线滚动补位"
```

## Day 3 收工自测清单

- [ ] 四个机制各有至少一个**数字例子**可以讲
- [ ] 能解释"为什么混合 prefill+decode 的 batch 比纯 decode 更接近 roofline 拐点"
- [ ] 能回答：prefix caching 命中粒度为什么是 block、错位会发生什么
- [ ] 能回答：为什么 prefill 不用 CUDA Graph（形状动态 + launch 开销占比低，两条都要说）
- [ ] 实验数据能讲出"预期 vs 实际"，差异能解释
- [ ] 四张卡片写完，每张默讲 90 秒不卡壳
