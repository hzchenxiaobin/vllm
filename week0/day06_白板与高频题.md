# Day 6 · 白板四件套 + 高频题过堂

> **总时长**：7-8 小时，全天只练"输出"
> **今日目标**：四件套全部计时达标；8 道高频题录音自答并修正——今天的每一分钟都直接兑换成面试分数
> **方法原则**：**开口说、动手写、计时**。看懂了 ≠ 讲得出，讲得出 ≠ 3 分钟内讲得完

---

## 作息建议

| 时间 | 内容 | 时长 |
|---|---|---|
| 09:00-11:00 | 白板四件套第 1-2 件（各练 3 遍） | 2h |
| 11:15-12:30 | 白板四件套第 3-4 件（各练 3 遍） | 1.25h |
| 14:00-17:00 | 高频题 1-8 录音自答 + 回听修正 | 3h |
| 19:30-20:30 | 薄弱项补练 + 修正稿定稿 | 1h |

---

## 白板四件套（每件练 3 遍，计时）

### 第 1 件：显存估算（目标：3 分钟）

**练习流程**：自己随机出题（模型、精度、上下文长度、卡型各换一个），完整推三遍。

**评分点**（面试官在看什么）：
1. 公式写对：KV/token = 2 × layers × **kv_heads** × head_dim × dtype（写出 kv_heads 而非 heads，直接+10 分）
2. 分步清晰：先权重 → 再每 token KV → 再可用显存 → 最后并发数
3. 有结论讨论：算完 12 条并发后主动说"所以长上下文场景显存经济性差，旋钮有 FP8 KV / P-D 分离 / offload"

**随机题 3 道（今天做掉）**：
- Qwen3-32B（64 层、8 KV 头、128 维）FP8，4×H100，32K 上下文
- Llama-3-8B（32 层、8 KV 头、128 维）FP16，1×4090 24G，8K 上下文
- DeepSeek-V3 类 MLA 模型：如果 KV 被压缩到 1/16，同样硬件并发翻几倍？（直觉题：≈16 倍，但受 compute 限制达不到，答出这句就是专家）

### 第 2 件：手绘 block table（目标：5 分钟）

**场景**：block_size=4；请求 A prompt "abcdefgh"（8 token）；请求 B 与 A 共享前缀 "abcd" 后分叉。

**要画出的状态演进**：
1. A 跑完 prefill：block0[a,b,c,d] ref=1，block1[e,f,g,h] ref=1
2. B 进来命中前缀：block0 ref=2，B 的 block_table=[block0, block2]
3. B 生成新 token 与 A 分叉：COW——若共享块要写入则复制；前缀只读则共享保持
4. A 结束：block0 ref=1，block1 释放回 free 池

**评分点**：引用计数每一步写对；主动提"链式 hash 防误命中"和"cache_salt 租户隔离"。

### 第 3 件：调度推演（目标：5 分钟）

**题目**：KV 容量只够 6 条并发（每条 4K token），同时到达 10 个请求（输出长度各异），max_num_batched_tokens=4096。

**推演模板**（按 step 叙述）：
- Step 0-1：准入 6 条（预算 4096 → 每条 4K prompt 要 chunked？说明切块过程）
- 第 7 条：KV 不足，滞留 waiting
- 某 step 有请求完成：block 释放 → waiting 队首准入
- 极端情况：running 中某请求 KV 申请失败 → 抢占队尾 → 打回 waiting、KV 释放、之后恢复（recompute：重算 KV）

**评分点**：先 running 后 waiting 的顺序；抢占选择逻辑；恢复方式 recompute vs swap 的取舍（recompute 省显存带宽、swap 省算力——vLLM V1 主要用 recompute）。

### 第 4 件：性能诊断树（目标：3 分钟讲完）

**背这张树，达到默写**：

```
症状 1：TTFT 高，TPOT 正常
 └─ 排队问题：waiting 队列长 → 准入太保守/容量不足
    ├─ 查 GPU KV cache usage：长期 <50% → 可调大 max_num_seqs
    └─ 长期 >90% + preemption 涨 → KV 超配 → 加卡/缩 max_model_len/FP8 KV
 └─ prefill 拥塞：长 prompt 占比高
    ├─ chunked prefill 块太小 → 调大 max_num_batched_tokens
    ├─ 重复前缀多 → 确认 prefix caching 开启、查命中率
    └─ 结构性 → P/D 分离

症状 2：TPOT 高（decode 慢）
 ├─ batch 太大 → 算力饱和 → 限 max_num_seqs 或加卡 TP
 ├─ 通信占比高（TP 场景）→ nsys 看 NCCL 时间 → 降 TP 或换拓扑
 ├─ CPU bound → 看 GPU 空闲间隙 → CUDA Graph / async scheduling 是否生效
 └─ 长上下文 → attention 占比涨 → KV 量化 / FlashInfer 后端

症状 3：preemption 持续增长
 └─ KV 池小于工作集 → 调低 max_num_seqs（宁可排队不可抢占）
    → 抢占的代价比排队大得多（重算 prefill）
```

---

## 高频题录音自答（每题：开口 3 分钟 → 回听 → 对照要点修正）

### 答题要点（修正用，不是背诵稿）

**Q1. PagedAttention 解决了什么问题？碎片率怎么算？**
- 三类浪费：最大长度预留、过度预留、内部碎片（60-80% → <4%）
- 碎片率 =（已分配 block 中未用 token 槽位）/（总槽位）；PagedAttention 只剩每请求最后一个 block 的内部碎片，期望浪费 = block_size/2 个 token
- 加分：OS 虚拟内存类比 + block table 间接寻址

**Q2. continuous batching vs static batching？**
- 调度粒度：批次 vs 迭代步
- 数字例子：40.5% 利用率那个（Day 3）
- 配套依赖：PagedAttention（动态显存）+ 每步重组 batch 的调度器

**Q3. 为什么 decode 用 CUDA Graph 而 prefill 不用？**
- decode：单步几 ms、kernel 小且多 → launch 开销占比高；形状可枚举（batch 分桶）
- prefill：形状动态（seq_len 任意）→ 图没法抓；单步时延长 → launch 占比可忽略
- 补充 piecewise CG：attention 动态部分排除在外

**Q4. chunked prefill 的 trade-off？token budget 怎么设？**
- 收益：TPOT 平滑 + 算力/带宽互补（算术强度混合）
- 代价：长 prompt TTFT 略增
- budget 调参：TTFT 导向调大、TPOT 导向调小；起点 2048-8192，按 p99 SLO 实测迭代

**Q5. 量化对 TTFT 和 TPOT 分别什么影响？**
- TPOT：访存 bound → 权重字节数减半 ≈ decode 时延减半，收益最大
- TTFT：算力 bound → 只有 W8A8/FP8 受益；W4A16 可能变慢（dequant）
- 加分：你的昇腾 WeightQuantBatchMatmul 实战经验一句

**Q6. P/D 分离什么时候不值得做？**
- 单实例能扛的流量：传输开销纯亏
- 短 prompt 负载：prefill 占比小
- 网络差的环境；加一句"分离的本质是用传输成本换 SLO 独立性"

**Q7. TP 什么时候是负收益？**
- 单卡放得下时：all-reduce 是纯开销
- 跨节点 TP（无 NVLink）：通信吃掉收益
- 大 batch：本来就 compute-bound，聚合带宽用不上
- 正收益场景：显存不够 / decode 带宽聚合 / 小 batch

**Q8. TTFT 高怎么排查？TPOT 高怎么排查？**
- 直接展开第 4 件的诊断树——这题就是诊断树的口语版

### 回听检查表（每条录音逐项打勾）

- [ ] 前 20 秒有结论句（先给答案，再展开）
- [ ] 至少一个数字或公式
- [ ] 至少一个 trade-off / 失效场景
- [ ] 没有"呃""然后""就是"超过 3 次
- [ ] 3 分钟内收住（超时的题重录）

---

## 今日产出

1. 四件套各 3 遍的计时记录（写在卡片角落：如"显存估算 2'40'' ✓"）
2. 8 条录音 + 每条的修正稿（修正稿 = 作战包第 5 件）

## Day 6 收工自测清单

- [ ] 四件套全部计时达标
- [ ] 8 题全部录过、听过、修正过
- [ ] 随机抽一题（让家人/朋友随机点），能 20 秒内开口给结论句
