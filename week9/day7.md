# Day 7：性能复现、精度评测与本周复盘

> **Week 9 · DeepInfer 源码精读 · Day 7（收官日）**
> 前置知识：Day 1–6 全部（今天的复盘要把六天的结论收口成最终交付物）。
> 源码基线：commit `9bd9cf2`（2026-10-21）。今日主角：`examples/benchmark.py`（588 行）、`examples/sweep/`、`eval/`、`tests/models/test_offline_inference.py`、`examples/refresh_offline_decode_ppl.py`、`engine/profile_server.py`、`csrc/graph_timer/`。
> 今日用时：4~5 小时（阅读 2h + 复盘产出 2h + 自测 0.5h）

---

## 今日学习目标

| # | 目标 | 验收标准 |
|---|---|---|
| 1 | 讲清 benchmark 的阶段解耦 | 解释 N 与 K 为什么独立、fork 机制、满接受口径、LB flush 免疫 |
| 2 | 讲清 PPL 基线回归机制 | 回答 README 两问：为什么按 platform/workload/DP rank 分 key？多节点刷新为什么文件锁？ |
| 3 | 产出本周五件交付物 | 架构图终稿 / 差异清单终稿（28 条带路径）/ kernel 地图 + 并行矩阵索引 / 四段口述稿索引 / 15 条面试 Q&A |
| 4 | 白板收官 | decode step 完整调用链；四大核心差异；prefix caching 路由层接口设计 |

---

## 一、benchmark：阶段解耦的测量学（`examples/benchmark.py`）

### 1.1 阶段常量与 N/K 独立性（README 指定问题）

```python
prefill_warmup_steps = 5;   prefill_measure_steps = 10;  prefill_profile_steps = 2
decode_warmup_steps  = 50;  decode_measure_steps  = 60;  decode_profile_steps  = 3
# prefill_total = 5 + 2×(10+1) + 2×2 = 31 步；decode_total = 50 + 2×61 + 2×5 = 182 步
max_tokens = decode_total_steps × (nextn + 1)              # 输出长度由步数常数反推，不需要指定

N = prefill_num_requests = min(ceil(31 × full_prefill_tokens / non_cached_tokens), 512)
K = args.decode_num_requests（默认 64）
batch_num_requests = max(N, K)                              # 唯一的耦合点：decode 复用 prefill 产出的共享前缀 KV
```

**为什么 N 和 K 独立**：prefill 的吞吐测量只取决于"每步塞满 `--prefill_batch_size` 个 token"，需要多少请求是**从步数倒推**的（N 上限 512，触顶时打 WARNING 说 prefill 供给不足以填满稳态计划）；decode 的测量只取决于"running 队列里恰好 K 个请求"。两阶段通过**一个共享前缀**衔接：

- `N > K`：下标 ≥ K 的请求给 `max_tokens=1`——最后一个 prefill chunk 产出首 token 后立即结束、在第一次 decode 调度前被逐出 running，**永远不进 decode batch**；
- `N ≤ K`：batch 取 K，多出来的 prefill chunk 变成更长的 warmup。

### 1.2 fork 机制：一条源请求孵出整个 batch（`benchmark.py:331-353`）

```python
def add_benchmark_batch():
    if cached_tokens > 0:
        source_seq = add_requests(1)[0]
        prefill_cached_prefix(source_seq, cached_tokens)    # chunk_checkpoints=[cached] 精确算到前缀末
        fork_request_batch(source_seq)                      # fork 成 max(N,K) 条，各自算 non-cached 后缀
    else:
        add_requests(batch_num_requests)

def fork_request_batch(source_seq):
    for fork_index in range(1, batch_num_requests):
        fork_seq = scheduler.fork_seq(source_seq, ...)      # ★ Day 2 的 clone_states 机制！
```

**Day 2 埋的线在这里收口**：`SchedulerOutputs.clone_states/clone_blocks` 的消费者就是 `fork_seq`——把源请求的 SWA state 与 KV 块克隆给 fork 出的请求（`assert fork_seq.num_cached_tokens == source_seq.num_cached_tokens`）。这就是"benchmark 复用共享前缀"的物理实现：**1 次前缀 prefill + N−1 次 state 克隆**，而不是 N 次重复 prefill。`chunk_checkpoints`（Day 2 add_request 的参数）在这里被用来"精确 prefill 到指定 token 数"——它同时也是 PD 传输的检查点机制证据。

### 1.3 测量窗口的工程学：免疫一切非稳态

benchmark 对测量窗口的保护是多层递进的（对应 examples/README 的 "Stage decoupling" 一节）：

| 干扰源 | 对策（代码位置） |
|---|---|
| 冷启动 / JIT / graph 捕获 | warmup 段（5/50 步）不测量（`run_stable_prefill_step(measure=False)`） |
| **LB 统计 flush**（EP all_gather，抖动几十 ms） | ① 测量区间若撞上 flush **整段剔除**（`flushed` 标志，`stable_prefill/decode` 的 deque）；② `align_lb_stat_flush_phase` 在 profile 窗前**烧步**把 flush 相位顶过去（断言 countdown 逐 step 递减） |
| 引擎周期日志 | `set_engine_log_interval(0/5/20)` 精确分段：批构造、warmup、profile 窗、teardown 全程静默；**测量段加倍——前半开日志**（每 5/20 步，moe_lb 从这里读）**后半静默出数** |
| profiler 本身的扰动 | ① sweep 扫点一律 `PROFILE=0`；② prefill profile 窗**断言恰好满 chunk**（`processed == full_prefill_tokens`）；③ decode 多采 2 步付 profiler 启动成本、导出只取最后 3 步；④ **decode profiler 之后不允许任何测量**（"退出后残余减速 7-11% @bsz=1"）；⑤ trace 内 step 时间膨胀近 50%（bsz=1）——**只看 kernel 组成，不许拿来算 SPS** |
| 请求供给耗尽 | `remaining_prefill_tokens()` 守门 + 断言"至少够 warmup+profile 的满 chunk" |

**指标口径**（`measure_prefill / measure_decode`）：

```python
prefill_tokens_per_s_per_gpu = 窗口 token 数 / 窗口秒数 / cp_size
decode_avg_sps  = 窗口步数 / 窗口秒数                       # VARLEN=0 下每 step 恒提交 1+nextn 个 token
decode_tokens_per_s_per_gpu = avg_sps × K × (nextn+1) / cp  # ★ 满接受口径（full-acceptance）
decode_max_step_ms = 窗口内最慢单步                          # 应接近 1000/avg_SPS
```

满接受口径的注释（`benchmark.py:539`）与 plot_pareto 的断言是本周数字纪律的缩影："**VARLEN=1 时每 step 提交的 token 数随接受率变，avg_SPS 已含接受率，不可再乘 τ**"——`plot_pareto.decode_tau`（`τ = 1 + nextn × a`，a 为外部给的 draft 接受率）对 VARLEN 数据直接 assert 拒绝。

### 1.4 benchmark 的环境定制（`setup_llm`）

```python
DEEPINFER_MAX_BACKTRACKING_TOKENS = "1"      # 最小回退（测量口径稳定）
DEEPINFER_FORCE_LOAD_BALANCE = "1"           # 固定伪随机路由（可复现的 expert 负载；
                                             #  worker 的 set_random_seed(seed+rank) 保证各 rank 种子不同）
DEEPINFER_MOE_LB_STAT_INTERVAL = "10"        # 加密 flush：prefill logged 段只有 ~11 步，保证段内有真 moe_lb
DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL = "32"  # benchmark 用更粗的图集合
gpu_memory_utilization = 0.95; max_model_len = 2^20
temperature=0.3, top_p=0.95, top_logprobs=5, ignore_eos=True
```

`--yoco_clip`（裸给即 140）映射到 mode/keep 两个正交变量，截断开启时强制 `PREFILL_STORE_EXTRA_STATE=1`（Day 6 的 ring 扩容）。`--profile_memory` 走三阶段显存快照（最大 batch 图捕获 / prefill / eager decode），dump torch memory snapshot pickle。

### 1.5 测量流程全景（今日 SVG 之一）

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 620" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect width="1100" height="620" fill="#fafbfc"/>
  <text x="550" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">benchmark.py 一次运行的阶段流水（PROFILE=1）</text>

  <!-- phase 0: setup -->
  <rect x="40" y="54" width="200" height="120" rx="10" fill="#f1f5f9" stroke="#94a3b8"/>
  <text x="140" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="#334155">引擎构造</text>
  <text x="140" y="100" text-anchor="middle" font-size="10.5" fill="#475569">profile_init 预热 kineto</text>
  <text x="140" y="116" text-anchor="middle" font-size="10.5" fill="#475569">图捕获 / decode_perf 曲线</text>
  <text x="140" y="132" text-anchor="middle" font-size="10.5" fill="#475569">日志关闭（interval=0）</text>
  <text x="140" y="156" text-anchor="middle" font-size="10.5" fill="#64748b">add_benchmark_batch：</text>
  <text x="140" y="170" text-anchor="middle" font-size="10.5" fill="#64748b">1 源请求 → fork max(N,K)</text>

  <!-- phase 1: prefill -->
  <rect x="270" y="54" width="380" height="120" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/>
  <text x="460" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a8a">Prefill 阶段（吞吐测量）</text>
  <rect x="284" y="92" width="105" height="26" rx="5" fill="#dbeafe" stroke="#2563eb"/>
  <text x="336" y="109" text-anchor="middle" font-size="10" fill="#1e3a8a">warmup ×5</text>
  <rect x="399" y="92" width="105" height="26" rx="5" fill="#93c5fd" stroke="#2563eb"/>
  <text x="451" y="109" text-anchor="middle" font-size="10" fill="#1e3a8a">burn（供给对齐）</text>
  <rect x="514" y="92" width="120" height="26" rx="5" fill="#bfdbfe" stroke="#2563eb"/>
  <text x="574" y="109" text-anchor="middle" font-size="10" fill="#1e3a8a">logged 测量段</text>
  <rect x="284" y="126" width="160" height="26" rx="5" fill="#dbeafe" stroke="#2563eb"/>
  <text x="364" y="143" text-anchor="middle" font-size="10" fill="#1e3a8a">silent 测量段（interval=0）</text>
  <rect x="454" y="126" width="180" height="26" rx="5" fill="#fde68a" stroke="#d97706"/>
  <text x="544" y="143" text-anchor="middle" font-size="10" fill="#78350f">profile 窗 ×2 满 chunk（先对齐 LB 相位）</text>
  <text x="460" y="166" text-anchor="middle" font-size="10" fill="#1e40af">指标：tokens/s/GPU（剔除 flush 区间；撞 flush 的 interval 丢弃）</text>

  <!-- phase 2: decode -->
  <rect x="680" y="54" width="380" height="120" rx="10" fill="#fef2f2" stroke="#dc2626" stroke-width="1.8"/>
  <text x="870" y="78" text-anchor="middle" font-size="12.5" font-weight="700" fill="#7f1d1d">Decode 阶段（SPS 测量）</text>
  <rect x="694" y="92" width="100" height="26" rx="5" fill="#fee2e2" stroke="#dc2626"/>
  <text x="744" y="109" text-anchor="middle" font-size="10" fill="#7f1d1d">warmup ×50</text>
  <rect x="804" y="92" width="100" height="26" rx="5" fill="#fecaca" stroke="#dc2626"/>
  <text x="854" y="109" text-anchor="middle" font-size="10" fill="#7f1d1d">logged ×60</text>
  <rect x="914" y="92" width="132" height="26" rx="5" fill="#fee2e2" stroke="#dc2626"/>
  <text x="980" y="109" text-anchor="middle" font-size="10" fill="#7f1d1d">silent ×61 → avg_SPS</text>
  <rect x="694" y="126" width="180" height="26" rx="5" fill="#fde68a" stroke="#d97706"/>
  <text x="784" y="143" text-anchor="middle" font-size="10" fill="#78350f">profile 窗：多采 2 步付启动成本，导出取 3 步</text>
  <rect x="884" y="126" width="162" height="26" rx="5" fill="#e2e8f0" stroke="#94a3b8"/>
  <text x="965" y="143" text-anchor="middle" font-size="10" fill="#475569">之后禁测（残余减速 7-11%）</text>
  <text x="870" y="166" text-anchor="middle" font-size="10" fill="#991b1b">指标：avg_SPS、max_step、满接受吞吐 = avg_SPS×K×(1+nextn)/cp</text>

  <!-- output -->
  <rect x="40" y="200" width="1020" height="70" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="550" y="224" text-anchor="middle" font-size="12.5" font-weight="700" fill="#14532d">产出（dp_rank==0）</text>
  <text x="550" y="246" text-anchor="middle" font-size="11" fill="#166534">stdout：完整配置 + 三指标（含引擎取整后的真实冗余 expert 数、dflash 的 expert 数另算）</text>
  <text x="550" y="264" text-anchor="middle" font-size="11" fill="#166534">--result_jsonl：配置 + 窗口原始量（步数/token 数/秒/avg_SPS）追加一行 —— "token 口径换算留给消费方"</text>
  <text x="550" y="282" text-anchor="middle" font-size="11" fill="#166534">trace：prefill(2 步) 与 decode(3 步) 两段头尾相接合成一个 json（_trace.export_profile_trace）</text>

  <!-- sweep -->
  <rect x="40" y="296" width="1020" height="150" rx="10" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
  <text x="550" y="320" text-anchor="middle" font-size="12.5" font-weight="700" fill="#7c2d12">sweep：一维扫描 → 一条 Pareto 曲线（sweeps/&lt;tag&gt;/）</text>
  <g font-size="10.5" text-anchor="middle">
    <rect x="60" y="336" width="300" height="88" rx="8" fill="#ffedd5" stroke="#ea580c"/>
    <text x="210" y="356" font-weight="700" fill="#7c2d12">decode 模式</text>
    <text x="210" y="374" fill="#9a3412">扫 decode_num_requests：8/16/32/64/128/256</text>
    <text x="210" y="390" fill="#9a3412">nextn 由参数指定；出图给 draft_acceptance a</text>
    <text x="210" y="406" fill="#9a3412">τ = 1 + nextn×a，两轴同乘 τ</text>
    <text x="210" y="418" fill="#9a3412">（VARLEN=1 的数据被 assert 拒绝）</text>
    <rect x="390" y="336" width="300" height="88" rx="8" fill="#ffedd5" stroke="#ea580c"/>
    <text x="540" y="356" font-weight="700" fill="#7c2d12">prefill 模式</text>
    <text x="540" y="374" fill="#9a3412">扫 prefill_batch_size：4096…65536</text>
    <text x="540" y="390" fill="#9a3412">--yoco_clip 固定 140（不作为扫描维度）</text>
    <text x="540" y="406" fill="#9a3412">横轴用实测窗口均值 ms</text>
    <text x="540" y="418" fill="#9a3412">（不用 batch/吞吐倒推——它假定每步恰好装满）</text>
    <rect x="720" y="336" width="320" height="88" rx="8" fill="#ffedd5" stroke="#ea580c"/>
    <text x="880" y="356" font-weight="700" fill="#7c2d12">encoder 模式（ve_sweep.py）</text>
    <text x="880" y="374" fill="#9a3412">只载 VE 权重，同进程逐点计时</text>
    <text x="880" y="390" fill="#9a3412">扫每 batch 图片数 1…32</text>
    <text x="880" y="406" fill="#9a3412">纵轴按每图 LLM token 数折算（与 prefill 同单位）</text>
  </g>
  <text x="550" y="438" text-anchor="middle" font-size="10.5" fill="#9a3412">工程细节：每扫点换端口（TIME_WAIT 复用会卡 rendezvous）；PROFILE=0；FIXED_FIELDS 一致性断言（同 tag 混配置直接报错）；仅 dp_rank 0 落 jsonl</text>

  <!-- discipline band -->
  <rect x="40" y="466" width="1020" height="140" rx="10" fill="#f8fafc" stroke="#94a3b8"/>
  <text x="550" y="492" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">测量纪律速记（面试可引用）</text>
  <text x="550" y="518" text-anchor="middle" font-size="11.5" fill="#334155">① N 从步数倒推（上限 512）、K 独立；耦合仅"decode 复用 prefill 的共享前缀 KV"（fork + clone_states）</text>
  <text x="550" y="540" text-anchor="middle" font-size="11.5" fill="#334155">② 满接受口径：吞吐 = avg_SPS × K × (1+nextn) / cp；VARLEN 数据已含接受率不可再乘 τ</text>
  <text x="550" y="562" text-anchor="middle" font-size="11.5" fill="#334155">③ 测量窗对 LB flush / 日志 / profiler 三类干扰全部免疫（剔除 + 烧步 + 分段静默）</text>
  <text x="550" y="584" text-anchor="middle" font-size="11.5" fill="#334155">④ trace 只看 kernel 组成（bsz=1 膨胀近 50%）；官方 Pareto：prefill 4 卡 / decode 64 卡（各在一个 scale-up 域内）</text>
</svg>
```

---

## 二、精度评测与 PPL 基线回归

### 2.1 `eval/`：AIME / GPQA / HLE 链路

| Benchmark | 任务数 | 每任务样本 | Pass@1 口径 | 默认采样 |
|---|---|---|---|---|
| AIME 2026 | 30 | 32 | 32 样本平均准确率 | temp 1.0 / top_p 1.0 |
| HLE | 2500（2158 文本 + **342 图片**） | 1 | 单样本准确率 | 同上 |
| GPQA Diamond | 198 | 8 | 8 样本平均准确率 | 同上 |

架构要点（`eval/README.md` + `run_eval.py`）：

1. **生成与评分解耦**：generation 把全部请求**单批**喂引擎、原始输出落盘；scoring 读盘算指标——**换 judge 不用重新生成**。每 benchmark 一个目录（`run.sh` 两段驱动 + `bench.py` 负责 prompt 构造/答案抽取/评分），`common/` 共享路径、JSONL I/O、Pass@k、judge、encoder；
2. **prompt 由 checkpoint 自带的 encoder 渲染**（`<checkpoint>/encoding/encoding.py`，运行时加载）——**编码逻辑随模型分发**，保证评测与官方 API 的 token 序列一致（deepseek-recipe 思想的评测侧延伸）；
3. **reasoning effort** 参数（1–100 或 low/high/max，默认 100）控制思考长度；`--max-len` 默认 144K；`--decode-blacklist "0,2,129262..129264"` 屏蔽特定 token；`--batch-invariant` 显式开启（注释："slower"——**评测要为可复现付性能税**）；
4. 数据集不自动下载（HLE/GPQA gated），record 格式与图片三种引用方式（本地路径 / base64 / multi_modal_data）在 README 有精确契约。

### 2.2 PPL 基线回归（README 两问的答案）

**机制**（`tests/models/test_offline_inference.py`）：跑 `offline_inference.py` → 正则解析每个 DP rank 的 `decode_ppl=` 行 → 与 `offline_decode_ppl.json` 的 golden **逐样本精确相等**（`assert actual == expected`——浮点 bitwise 相等！）。

**为什么按 platform / workload / DP rank 分 key？**

| key | 原因 |
|---|---|
| platform（nvidia/ascend） | 两平台的 kernel 实现、数值路径不同（cuda/ vs asc/ 双份 TileLang、FlashMLA vs NPU 实现、fp8_sf_type int32 vs int16），PPL 必然不同——甚至**并行配置都不同**：NVIDIA 4 卡（EP4/CP2），Ascend 2 节点×8 卡（EP16/CP8，"4 卡放不下模型"），两者 dp_size 都是 2 |
| workload（short/long） | short 与 long（16389 输入 / 4096 输出 / nextn=5）走的代码路径不同（长上下文的 KV 布局、chunk 边界），基线分开维护 |
| dp_rank | **每个 DP rank 的请求子集是固定的**（rank r 恰好 5×(r+1) 个样本，_prompts 的确定性分配）——attention DP 下各 rank 请求不相交、采样流独立（Day 5），PPL 序列天然按 rank 确定。NVIDIA 有 batch-invariant 兜底；Ascend 不支持 batch-invariant，**固定 per-rank 请求集是它确定性的唯一来源**——按 rank 分 key 是把"确定性条件"显式编码进基线结构 |

golden 的 rank 集必须与当前并行配置精确匹配（多了是 stale、少了是未测），并断言"从 0 开始连续"——**基线结构与并行拓扑强绑定**。

**为什么多节点刷新要文件锁？**（`refresh_offline_decode_ppl.py`）`_launch.run_dp` 把 DP rank 切到各节点，**每个节点只产出自己那几个 rank 的测量值**，刷新是"读整份 JSON → 只改自己的 rank → 写回"的读-改-写。多节点并发刷新时无锁会发生丢更新（后写者覆盖前写者的 rank 段）。解法：`fcntl.flock(LOCK_EX)` 跨节点文件锁（共享文件系统上的锁语义）+ 写回用 tmp + `os.replace` 原子替换 + 收缩配置时清掉 `rank ≥ DP_SIZE` 的陈旧条目。配套限制：`--output`（候选文件）**只允许单节点**——多节点下本进程只有部分 rank，"写出来必然缺 rank"（直接 error 并提示用默认路径）。

这套机制回答了一个深层问题：**"一个数值变了，怎么知道是回归还是预期改进？"**——跑 refresh 出候选文件（单节点）、人审 diff、确认后合并进 golden。PPL 基线 = DeepInfer 的数值正确性 CI 门禁，与 `test_batch_invariant`（结构不变性）、eval（任务精度）构成三层精度保障。

### 2.3 profiling 工具链（`profile_server.py` + `csrc/graph_timer/`）

**运行期 HTTP profiling**（`engine/profile_server.py`）：每个 engine 进程一个 HTTP 服务（默认关，`DEEPINFER_PROFILE_HTTP_ENABLE=1`），`POST /profile` 采 num_steps 步并落盘——**阻塞到文件写完才返回路径，所以没有停止接口**。监听 `PORT_BASE(6070) + driver_gpu_id`；只采 driver worker 那张卡（torch profiler 是进程内的，CP slave 在别的进程）。与 benchmark 的离线 profiling 互补：**生产/长跑过程中按需抓一段**。

**graph_timer**（`csrc/graph_timer/`，cuda/asc 双实现）：解决"**CUDA graph 内部的分段计时**"——profiler 看到的图 replay 是一个整体。机制：

```python
# cuda_graph_runner.capture（Day 3 见过的 keep_graph=True 在这里兑现）：
graph = torch.cuda.CUDAGraph(keep_graph=True)          # 保留原始 cudaGraph_t
graph_handle = graph_timer.GraphHandle(graph)          # C++ 拿到原始图
graph_timer.register_graph(graph_handle)               # 匹配图内的事件记录节点（event-record nodes）
# replay 前：
cuda_graph_timer.flip_event_buffer(graph_handle)       # ping-pong 事件缓冲翻面
graph.replay()                                         # 图内 kernel 触发事件节点记录时间戳
# post_process 时：
cuda_graph_timer.flush_out()                           # 汇成 per-name 耗时字典
```

`EventHandle.record(stream, capture_external)` 的 `capture_external` 标志让 **capture 期间的事件记录变成图的事件节点**——Day 4 kernel 地图里的 `cuda_graph_timer.record("attn-part-N")` 计时就是这么嵌进图里的。这是"图内黑盒"问题的标准解法之一（另一个是 Day 3 的 keep_graph + 事件节点匹配）。

---

## 三、本周复盘一：白板收官

### 3.1 一次 decode step 的完整调用链（终版，默写级）

```
LLM._run_engine → engine.step()
  ① scheduler.schedule(pool)                     [C++，释放 GIL；DP 投票定步型（ALL/ANY 点）]
  ② clone/zero_states（抢占/fork）→ default stream 上的 GPU 克隆；unpin 延迟
  ③ run_workers("execute_model_step")            [CP>1 时 shm 广播 (method, kwargs)]
     worker.execute_model_step（四段式）：
     1. _correct_seqlens_qk                      [C++ prepare_zos_inputs + TileLang confidence/seqlens kernel]
     2. model_runner.execute_main_model           [禁止 CUDA 同步]
        _prepare_input → prepare_decode_v4_zos    [GPU 上写图 buffer：gather_input_ids_zos D2D 拼 token、
                                                   prepare_v4_decode_metadata 一次 kernel 写 12 类字段]
        graph: select(total_q) → replay           [紧凑字段图内展开（compute_attn_info）]
        eager: model() → N × DecoderLayer
          ├ compute_attn_info：pivots → 索引      [TileLang get_kv_*_index / expand_swa_index]
          ├ Q/KV 投影                              [DeepGEMM FP8；融合路径 fused_norm_rope_attn_rope_cast]
          ├ store_kv_cache：量化写 cache           [TileLang，ue8m0 位拼接]
          ├ indexer：paged-mqa-logits              [DeepGEMM] + topk [DeepSelect] + convert_kv_index [TileLang]
          ├ flash_mla_with_kvcache                 [FlashMLA：paged FP8 cache + 物理索引]
          ├ MoE：mega_moe（NVL 域内）              [DeepGEMM SymmBuffer] 或 dispatch/combine [DeepEP]
          │        （gate 经 EPLB 的 logical_to_physical 映射；expert_count 顺手累计）
          └ dflash.update_main_cache              [目标层特征 → draft 的主 token K]
     3. _post_process_step(prev)                  [同步上一步 sample_end_evt → process_MTPInfo →
                                                    scheduler.process_outputs → build_request_outputs]
     4. execute_sample                            [d2h_evt 同步 → prepare_sampler_inputs(C++) →
                                                    SO mask（per draft 步）→ fused_sample（TileLang：
                                                    温度/top-k/top-p/logsoftmax/SplitMix64 采样/reject
                                                    sampling）→ dflash graph replay（draft + confidence）]
  ④ pre_sampler_sync                              [DeferredRelease + do_save_kvcache（虚）+ SO pump]
  ⑤ record sample_end_evt；prev_step_state = curr；return 上一步的输出（uint64 句柄）
```

### 3.2 四大核心差异（README 指定的对比口径）

| 维度 | vLLM V1 | DeepInfer | 一句话本质 |
|---|---|---|---|
| **调度** | Python 调度器 + async output processor 线程 | **C++ 调度器（GIL 释放）+ worker 四段式穿插 + GPU 侧 seqlens 修正 + 对象池化** | vLLM 用并发线程藏开销，DeepInfer 用语言与数据结构消灭开销 |
| **并行** | TP 为主（每层 2 次 all-reduce），PP/DP/EP 可选 | **Attention DP（零数据通信）+ balanced CP（zigzag）+ MoE Wide-EP + EPLB**；无 TP/PP | vLLM 切张量，DeepInfer 切请求——MLA 的 latent KV 让 TP 无收益 |
| **投机解码** | 固定长度 draft，全验证 | **confidence-scheduled varlen**：GPU 上按累积接受率分配验证预算，省 30% 验证计算 | DeepInfer 把"验证多少"也变成了调度问题 |
| **确定性** | 无系统保证 | **六处 batch-invariant 纵向保证 + 确定性采样配方**（SplitMix64、固定归约序、PPL bitwise 回归） | 确定性是设计目标不是副产品——为此宁可禁用个别融合路径 |

---

## 四、本周复盘二：开放题——prefix caching 路由层接口设计

> "如果让你给 DeepInfer 补一个 prefix caching 路由层（未开源部分），接口该怎么设计？"（Day 1 预告、Day 3 边界盘点的收口）

**先盘点已有的钩子**（全部来自本周源码）：

| 已有设施 | 位置 | 提供的能力 |
|---|---|---|
| `select_state_slice_to_store(TO_CE=...)` | cache_engine.py | 从 state 槽切出落盘段（128 对齐）或完整 PE→CE 段 |
| `load_state_from_store_to_prefill_cache` | cache_engine.py | 落盘 state 装回 prefill cache（含 ring 重排） |
| `load_segments_{swa,ca}` + `BlockLoadPlan` | metadata + csrc | 块粒度装载描述符（paged KV → gathered buffer，-1 零填） |
| `do_save_kvcache / pe_*_save_kvcache / init_pd_split` | llm_engine.py 虚函数 | 引擎侧落盘/传输钩子（开源树无覆盖者） |
| `chunk_checkpoints` | Sequence | "从指定位置续算"的检查点（benchmark 用它精确 prefill 到 N） |
| KVSTORE_GRAIN=128 / 命中策略 L−134 / SWA 回退一块 | Q&A + cache_engine | 存储粒度与命中语义 |

**接口设计**（对齐 Q&A 的三段约束：匹配在路由前、fetch 异步、swap-in 与前向重叠）：

```python
# ===== 层 1：Entrypoint 侧（CHECK 角色）=====
class PrefixRouter:
    def match(self, token_ids: np.ndarray) -> PrefixHit:
        """hash 链前缀匹配（仅 metadata，块哈希 = block 内 token 哈希的滚动链——vLLM 同款）。
        返回 PrefixHit(hit_blocks: int, swa_hit: bool, store_host: str)。
        ★ 约束（Q&A 原文）：命中长度必须在路由决策之前算出，这步无法 overlap。"""

# ===== 层 2：存储客户端（kvcache-store 的本地代理）=====
class KVStoreClient:
    def fetch_async(self, prefix_key: PrefixKey, hit_blocks: int) -> FetchHandle:
        """独立进程把 block KV 从 3FS 拉进本机共享内存（对应 Q&A 的 Fetching 段）；
        SWA KV 走内存目录：hit 但不在本机 → 400G VLAN 回拉（store_host 字段）。"""

# ===== 层 3：引擎侧装载（喂给已有的 load/swap 接口）=====
class PrefixLoader:
    def swap_in(self, handle: FetchHandle, seq: Sequence) -> SwapInPlan:
        """共享内存 → HBM：逐层 H2D 与 forward 重叠（GDR，RDMA NIC 打满 PCIe 不占 SM）。
        输出直接变成 seq.chunk_checkpoints + metadata 的 load_segments —— 复用 BlockLoadPlan。"""
```

**五个关键决策**：

1. **块哈希链放在 SequenceData**（C++ 侧，与 token 追加同步滚动）——Python 侧不能有 per-block 计算（ZOS 纪律）；
2. **hit_blocks 的粒度 = KVSTORE_GRAIN=128**，SWA 命中判定用"最后一块区间内有无 SWA KV"（Q&A 的回退一块规则），接口上表现为 `swa_hit: bool` 而非长度；
3. **fetch 与 swap_in 全异步**，引擎在 `add_request` 后、首步调度前拿到 `SwapInPlan`；未就绪的请求留在 waiting（复用 SO 的"编完才入队"模式）；
4. **写路径**挂在已有的 `do_save_kvcache` 虚函数上：decode 的 128 边界跨越 → state ring 的 replica 区快照（Day 1 QA）→ `select_state_slice_to_store` 切段 → 异步写 3FS；
5. **失败语义**对齐 SO 的哲学：fetch 超时/校验失败 → 降级为无命中 prefill（杀请求是最坏情况，不崩引擎）。

**与 Q&A 的对账**：匹配不可隐藏 ✓（match 同步）；Fetching 异步进程 ✓；Swap-in GDR 重叠 ✓；本机 SWA 目录 + 400G VLAN ✓（store_host）；TTL/GC 在存储侧不进引擎 ✓。

---

## 五、本周交付物（终稿）

### 5.1 一页纸架构图（Day 1 版本的迭代终稿）

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 980" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect width="1140" height="980" fill="#fafbfc"/>
  <text x="570" y="30" text-anchor="middle" font-size="21" font-weight="700" fill="#0f172a">DeepInfer 一页纸终稿：生产系统 × 开源引擎 × 本周六天的新知（commit 9bd9cf2）</text>

  <!-- production band -->
  <rect x="30" y="48" width="1080" height="120" rx="10" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="570" y="70" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e1b4b">生产层（未开源，Q&amp;A 描述）：E/P/D 分离 + 3FS + 四负载均衡 + deepseek-recipe</text>
  <text x="570" y="92" text-anchor="middle" font-size="10.5" fill="#4338ca">prefill 普通 8 卡机（EP2–8/DP2–8）· decode 64–128 卡超节点（EP64–128 Wide-EP，moe_lb 1.090）· VE 单卡 · P:D=1:1 · KV TTL 3/7 天 · 14.4TB/GPU · 155T tok/天 · $38/GPU-hour</text>
  <text x="570" y="112" text-anchor="middle" font-size="10.5" fill="#4338ca">Prefill LB（4 档桶 + 二次成本模型 cost(q,c) + 时间预算 all-reduce MIN）· Decode LB（请求数 + KV 总长双维）· VE LB（patch 数）· EPLB（prefill 4 冗余/1000 步 · decode 1 冗余/10000 步）</text>
  <text x="570" y="134" text-anchor="middle" font-size="10.5" fill="#4338ca">KV 三层：in-flight（GPU）· SWA（本机内存 1,415,040B/seq = 20 层×134×528B，400G VLAN 回拉）· block KV（3FS，10+2 EC）· 命中 ≤ L−134，SWA miss 回退一块</text>

  <!-- open source engine -->
  <rect x="30" y="184" width="1080" height="420" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>
  <text x="570" y="208" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1e3a8a">开源引擎（deepinfer/ + csrc/）</text>

  <!-- left: engine core -->
  <rect x="50" y="224" width="330" height="360" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="215" y="246" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a8a">引擎核心（Day 2）</text>
  <text x="66" y="270" font-size="10.5" fill="#1e40af">LLMEngine（离线唯一入口，无 API server）</text>
  <text x="66" y="288" font-size="10.5" fill="#1e40af">C++ Scheduler（1329 行，GIL 释放，DP 投票）</text>
  <text x="66" y="306" font-size="10.5" fill="#1e40af">ZOS 四段式：fwd(N)→post(N−1)→sample(N)→defer</text>
  <text x="66" y="324" font-size="10.5" fill="#1e40af">  唯一同步点等已完成的 event；输出晚一步</text>
  <text x="66" y="342" font-size="10.5" fill="#1e40af">seqlens 修正：CPU 定 total_q，GPU 分配</text>
  <text x="66" y="360" font-size="10.5" fill="#1e40af">  （cumprod + rank counting）</text>
  <text x="66" y="382" font-size="10.5" fill="#1e40af">RequestOutputPool：uint64 句柄 + DeferredRelease</text>
  <text x="66" y="400" font-size="10.5" fill="#1e40af">forbid_cuda_sync：launch 路径零同步纪律</text>
  <text x="66" y="422" font-size="10.5" fill="#1e40af">EngineType：NORMAL / PROMPT / COMPLETION</text>
  <text x="66" y="440" font-size="10.5" fill="#1e40af">  （PD 钩子：init_pd_split / send_processed_prompts</text>
  <text x="66" y="456" font-size="10.5" fill="#1e40af">   / pe_*_save_kvcache —— 部署层未开源）</text>
  <text x="66" y="478" font-size="10.5" fill="#1e40af">shm_broadcast：无锁环形缓冲（1MB×10）</text>
  <text x="66" y="496" font-size="10.5" fill="#1e40af">  driver→CP slaves 方法广播；只广播 kwargs</text>
  <text x="66" y="518" font-size="10.5" fill="#1e40af">profile_server：POST /profile（6070+gpu_id）</text>
  <text x="66" y="536" font-size="10.5" fill="#1e40af">stats：avg_SPS / moe_lb / TTFT / MFU（FP8=2×BF16 峰值）</text>
  <text x="66" y="560" font-size="10.5" fill="#1e40af">DecodePerfCache：静态曲线 + update_mtp_runtime 动态校正</text>

  <!-- middle: worker -->
  <rect x="400" y="224" width="350" height="360" rx="8" fill="#fee2e2" stroke="#dc2626"/>
  <text x="575" y="246" text-anchor="middle" font-size="12" font-weight="700" fill="#7f1d1d">Worker 层（Day 3）</text>
  <text x="416" y="270" font-size="10.5" fill="#991b1b">model_runner：四路分派（driver/follower×prompt/</text>
  <text x="416" y="286" font-size="10.5" fill="#991b1b">  decode×ZOS×graph）；禁止 CUDA 同步</text>
  <text x="416" y="308" font-size="10.5" fill="#991b1b">metadata 四套：base/V4（C++）· yoco（截断视图）·</text>
  <text x="416" y="324" font-size="10.5" fill="#991b1b">  dflash（fixed-stride）· zos（GPU 侧构建）</text>
  <text x="416" y="342" font-size="10.5" fill="#991b1b">紧凑字段 + 图内展开（get_clean_list）</text>
  <text x="416" y="364" font-size="10.5" fill="#991b1b">cache_engine：单 uint8 池 + 层偏移 + state 槽</text>
  <text x="416" y="382" font-size="10.5" fill="#991b1b">  （cuMemAlloc 直分；share-kv = CED）</text>
  <text x="416" y="400" font-size="10.5" fill="#991b1b">  state_size_to_disk = 20×134×528B；落盘 128 粒度</text>
  <text x="416" y="418" font-size="10.5" fill="#991b1b">  开源边界：select_state_slice_to_store / load_* 就绪</text>
  <text x="416" y="440" font-size="10.5" fill="#991b1b">CUDA graph：total_q 键 + 2 幂/间隔 8 集合</text>
  <text x="416" y="458" font-size="10.5" fill="#991b1b">  从大到小共享 pool；input_on_device 图内零 H2D</text>
  <text x="416" y="476" font-size="10.5" fill="#991b1b">dflash runner：req_cnt 键，共享主图 pool/stream</text>
  <text x="416" y="498" font-size="10.5" fill="#991b1b">graph_timer：事件节点 ping-pong，图内分段计时</text>
  <text x="520" y="524" font-size="10.5" fill="#7f1d1d" font-weight="600">kernel 地图（Day 4）：</text>
  <text x="416" y="542" font-size="10" fill="#991b1b">FlashMLA（attn）· DeepGEMM（全部 GEMM+Mega MoE）· DeepEP（all-to-all）</text>
  <text x="416" y="558" font-size="10" fill="#991b1b">TileKernels（gate/swiglu/量化）· DeepSelect（topk）· TileLang 33 自研（四位一体）</text>

  <!-- right: model & features -->
  <rect x="770" y="224" width="320" height="360" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="930" y="246" text-anchor="middle" font-size="12" font-weight="700" fill="#14532d">模型与特性（Day 4/6）</text>
  <text x="786" y="270" font-size="10.5" fill="#166534">V4.1：MLA + 压缩 KV(C1/C2) + 稀疏 indexer + CED</text>
  <text x="786" y="288" font-size="10.5" fill="#166534">  + YoCo consumer + HC 残差流 + Engram</text>
  <text x="786" y="306" font-size="10.5" fill="#166534">V4Attention 11 步；Wq/Wkv LoRA 两段投影</text>
  <text x="786" y="330" font-size="10.5" fill="#166534">DSpark：DFlash 与主模型共享器官</text>
  <text x="786" y="348" font-size="10.5" fill="#166534">  （main_kv_proj 顺路写 draft KV）</text>
  <text x="786" y="366" font-size="10.5" fill="#166534">  MTPInfoPool 按调度器槽位常驻</text>
  <text x="786" y="384" font-size="10.5" fill="#166534">  reject sampling 偏差 ≤ 5e-5（有证明）</text>
  <text x="786" y="402" font-size="10.5" fill="#166534">  验证 token −30%，吞吐 +4~7%@60 并发</text>
  <text x="786" y="424" font-size="10.5" fill="#166534">bounded replay：KEEP=140（下界 134=128+6）</text>
  <text x="786" y="442" font-size="10.5" fill="#166534">  decoder prefill 近乎减半；mask/compact 等价</text>
  <text x="786" y="464" font-size="10.5" fill="#166534">batch-invariant 六处保证（FlashMLA/deep_gemm/</text>
  <text x="786" y="482" font-size="10.5" fill="#166534">  tile_kernels/采样 RNG/求和序/EPLB 冲突规避）</text>
  <text x="786" y="500" font-size="10.5" fill="#166534">  确定性 = BI + 关 MTP + 固定 seq_seed</text>
  <text x="786" y="522" font-size="10.5" fill="#166534">SO：xgrammar 三 schema · matcher（KMP trigger/</text>
  <text x="786" y="540" font-size="10.5" fill="#166534">  bound/回滚 nextn+2）· per-draft-step mask</text>
  <text x="786" y="558" font-size="10.5" fill="#166534">  · 故障只杀单请求</text>

  <!-- parallel band -->
  <rect x="30" y="620" width="1080" height="110" rx="10" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/>
  <text x="570" y="644" text-anchor="middle" font-size="13" font-weight="700" fill="#7c2d12">并行策略矩阵（Day 5）：无 TP/PP，一切是拓扑的函数</text>
  <text x="570" y="668" text-anchor="middle" font-size="11" fill="#9a3412">Attention DP：切请求，数据面零通信（调度投票 gloo CPU 组）· MLA latent KV 使 TP 无收益</text>
  <text x="570" y="688" text-anchor="middle" font-size="11" fill="#9a3412">balanced CP：zigzag 镜像 (b+1)+(2P−b)=2P+1，DeepEP BucketBuffer 惰性 all_gather + 双通道广播</text>
  <text x="570" y="708" text-anchor="middle" font-size="11" fill="#9a3412">MoE EP：连续切分横跨 DP/CP · NVL 域内 Mega MoE / 域外 dispatch-combine · EPLB 槽位级逐层重排热切换</text>

  <!-- measurement band -->
  <rect x="30" y="746" width="1080" height="110" rx="10" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.8"/>
  <text x="570" y="770" text-anchor="middle" font-size="13" font-weight="700" fill="#581c87">测量与保障（Day 7）</text>
  <text x="570" y="794" text-anchor="middle" font-size="11" fill="#7e22ce">benchmark：N 从步数倒推/K 独立 · fork+clone_states 共享前缀 · 满接受口径 avg_SPS×K×(1+nextn)/cp</text>
  <text x="570" y="814" text-anchor="middle" font-size="11" fill="#7e22ce">  测量窗免疫 LB flush/日志/profiler · sweep 三模式出 Pareto（τ=1+nextn×a，VARLEN 拒绝）</text>
  <text x="570" y="838" text-anchor="middle" font-size="11" fill="#7e22ce">eval：生成/评分解耦（AIME 32 样本 · HLE 含 342 图 · GPQA 8 样本）· encoder 随 checkpoint 分发</text>
  <text x="570" y="858" text-anchor="middle" font-size="11" fill="#7e22ce">PPL 基线：platform/workload/dp_rank 分 key · bitwise 相等 · fcntl 多节点锁合并 · refresh 出候选人审</text>

  <!-- footer -->
  <text x="570" y="908" text-anchor="middle" font-size="11.5" fill="#64748b">图例：紫=生产（未开源）蓝=引擎 红=Worker 绿=模型特性 橙=并行 紫=测量 —— 深入细节见 day2~day7 各日的专题图</text>
  <text x="570" y="930" text-anchor="middle" font-size="11.5" fill="#64748b">本周方法：文档（Q&amp;A）定系统级结论，代码定机制级结论，测试定契约 —— 三方互证，结论锚定 commit 9bd9cf2</text>
  <text x="570" y="952" text-anchor="middle" font-size="11.5" fill="#64748b">配套交付：差异清单 28 条（5.2）· kernel 地图（day4 §8）· 并行矩阵（day5 §7）· 四段口述稿（day6 §6）· 15 问 Q&amp;A（5.3）</text>
</svg>
```

### 5.2 差异清单终稿：DeepInfer vs vLLM V1（28 条，每条带源码路径）

**引擎与调度（Day 1–2）**

| # | 维度 | vLLM V1 | DeepInfer | 源码 |
|---|---|---|---|---|
| 1 | 服务形态 | API server + EngineCore（ZMQ） | 仅离线 LLM，无 online wrapper | engine/llm_engine.py:140 |
| 2 | 调度器 | Python | C++（pybind，GIL 释放） | csrc/engine/scheduler.hpp |
| 3 | 异步调度 | 独立 output 线程 + 结构化并发 | worker 四段穿插 + GPU 侧 seqlens 修正 | worker/worker.py:88-189 |
| 4 | 上一步结果进本步 | pinned CPU（output processor D2H） | GPU 常驻副本 + TileLang 分配 | sequence.py `*_gpu`、schedule_mtp_varlen.py |
| 5 | DP 调度协作 | 无步型投票 | 两轮 all_reduce(MIN) + gloo CPU 组 | scheduler.hpp `_schedule()` |
| 6 | 输出对象 | Python CompletionOutput | C++ 池 + uint64 句柄（跨 step 须 clone） | RequestOutputPool |
| 7 | 平台 | 十余平台 + check_and_update_config | 仅 GPU/NPU，接口极小（227 行） | platform.py |
| 8 | 模型支持 | registry 数百模型 | 仅 V4.1-Flash 一族 | model_executor/models/ |

**Worker 与 KV（Day 3）**

| # | 维度 | vLLM V1 | DeepInfer | 源码 |
|---|---|---|---|---|
| 9 | Attention 后端 | FlashAttn/FlashInfer/TRTLLM 多后端 | FlashMLA 单后端薄适配 | layers/attention_backend.py |
| 10 | KV cache | kv_cache_manager 完整开源 | 单机层开源；分布式系统（3FS）未开源 | cache_engine.py、README.md:38 |
| 11 | metadata 构建 | Python（按后端） | C++ prompt / GPU 侧 decode（ZOS）+ 图内展开 | csrc/engine/input_metadata/ |
| 12 | KV 物理组织 | per-layer 同构张量 | 单 uint8 池 + 层偏移 + state 槽（cuMemAlloc 直分） | cache_engine.py `process()` |
| 13 | CUDA graph 键 | batch size | total_q 直键 + input_on_device 图内零 H2D | cuda_graph_runner.py:74,95 |

**模型与 kernel（Day 4）**

| # | 维度 | vLLM V1 | DeepInfer | 源码 |
|---|---|---|---|---|
| 14 | 采样 | kernel 序列（softmax→topk→multinomial） | fused_sample 单 kernel + SplitMix64 | fused_sample.py:17-120 |
| 15 | MoE 路径 | fused_moe（按后端） | Mega MoE（NVL 域）/ DeepEP dispatch-combine + EPLB 映射进 gate | v4_modules/mlp.py:184-188 |
| 16 | kernel 生态 | 内置 CUDA/Triton | vendor submodule + pip TileKernels + 33 TileLang（四位一体测试） | vendor/、tilelang_ops/ |
| 17 | C++ 职责 | 引擎/输出部分 | 全部每 step 簿记（scheduler 1329 行等） | csrc/ |

**并行（Day 5）**

| # | 维度 | vLLM V1 | DeepInfer | 源码 |
|---|---|---|---|---|
| 18 | attention 并行 | TP（每层 2 次 all-reduce） | DP（零数据通信 + 投票） | parallel_state.py |
| 19 | 序列并行 | 无独立 CP | balanced CP：zigzag + 惰性 all_gather + 双通道广播 | prepare_cp_mirror.py、cp.py |
| 20 | expert 负载均衡 | 无 | EPLB 全链路（统计→贪心→IPC/P2P→热切换） | eplb/ |
| 21 | 拓扑感知 | 分离的 platform 机制 | deep_cluster 独立包驱动 Mega/IPC/NUMA | deep_cluster/ |

**高级特性（Day 6）**

| # | 维度 | vLLM V1 | DeepInfer | 源码 |
|---|---|---|---|---|
| 22 | 投机解码 | 固定长度 draft + 全验证 | confidence-scheduled varlen + 槽位池 | dflash.py、mtp_info_pool.py |
| 23 | SWA/前缀重算 | prefix caching 全命中 | 有界重放（KEEP=140，常数成本） | prepare_tensor_input_v4.cpp |
| 24 | 确定性 | 无系统保证 | 六处 batch-invariant + 确定性配方 | test_batch_invariant.py |
| 25 | 结构化输出 | xgrammar（采样前 mask） | 同源 + per-draft-step mask + matcher 回滚 | structure_output/ |

**测量与保障（Day 7）**

| # | 维度 | vLLM V1 | DeepInfer | 源码 |
|---|---|---|---|---|
| 26 | benchmark | 每 step 吞吐简单累计 | 阶段解耦 + 满接受口径 + 三类干扰免疫 | examples/benchmark.py |
| 27 | 数值回归 | 无 bitwise 基线 | PPL 按 platform/workload/rank 分 key，bitwise 相等 | test_offline_inference.py |
| 28 | 图内计时 | 无 | graph_timer 事件节点 ping-pong | csrc/graph_timer/ |

**一句话总结**：vLLM 是"通用性优先"的社区框架（多模型多平台多后端的抽象层），DeepInfer 是"单模型深度优先"的生产引擎（把抽象层换成熔合 kernel + C++ 簿记 + 拓扑函数化的并行）——两者在调度语言、KV 组织、并行哲学、确定性保障、测量纪律五个方向上做出了系统性的不同选择。

### 5.3 面试 Q&A 15 条（从本周"关键问题"筛选）

1. **为什么 prefill 和 decode 用不同规格的机器？** prefill compute-bound，卡数只需 HBM 放得下（EP2–8 普通机）；decode memory-bound，Wide-EP 把 expert 权重切到 64–128 卡、每 step 访存降 1/64–1/128 且 batch 聚合（需超节点 + EPLB moe_lb 1.09）。P:D=1:1，最小高效单元 64+64。（day1 §二）
2. **开源边界在哪？** 引擎与 kernel 全开源；KV cache 系统的**调用方**（do_save_kvcache 等虚函数无覆盖者）与三个 DP 侧 LB 未开源，但 `select_state_slice_to_store` / `load_state_from_store_to_prefill_cache` 接口与规格级测试就绪——"字节怎么切开源了，切完发给谁没开源"。（day1 §四、day3 §3.3）
3. **ZOS 怎么实现？与 vLLM 异同？** worker 四段式：launch forward(N) → post-process(N−1，同步的是已完成的 event) → launch sample(N) → defer。vLLM 用独立线程 + 结构化并发、调度器仍 Python；DeepInfer 靠 C++ 调度器 + GPU 侧 seqlens 修正（DSpark varlen 下唯一可行）。（day2 §四）
4. **进程模型？** 三层：DP-rank 顶层进程（各是完整引擎）→ 引擎进程内 C++ 调度器 + driver Worker → CP slave 子进程（shm 广播收方法）。对照 vLLM：砍 API server/EngineCore 分离，保留 multiproc worker + shm_broadcast，新增 EngineCallbacks 回调注入。（day2 §二）
5. **shm_broadcast 的角色？** driver→CP slaves 的方法广播：无锁环形缓冲（每 chunk 1+n_reader 个 flag 字节，写者先清 reader flag 再置 written）+ ZMQ 溢出通道；只广播 (method, kwargs)，driver 独有调度结果不广播——调度只在 driver。（day2 §五）
6. **metadata 为什么多套？** 一个模型的四个正交特性各需一套字段：base/V4（C++，压缩 KV+indexer+CED）、yoco（截断视图，层循环中途变形）、dflash（fixed-stride draft）、zos（GPU 侧构建直写图 buffer）。（day3 §2.2）
7. **capture 集合怎么定？** 2 的幂 + 固定间隔 8，**以 total_q（token 数）为键**（varlen 投机的必然：total_q ∈ [n_req, n_req×(1+nextn)] 任意整数），max = max_num_seqs×(1+nextn)；从大到小捕获共享 memory pool。（day3 §4.1）
8. **dflash 为什么单独的 runner？** batch 键不同（req_cnt vs total_q——draft 是 fixed-stride）、生命周期不同（sample 阶段、但共享主图 pool/stream）、输入输出类型不同（MTPDecodeInput/MTPPredictInfo，do_sample 仅 driver）。（day3 §4.4）
9. **五个 vendor 库各管哪段？** FlashMLA=attention；DeepGEMM=全部 GEMM+Mega MoE+gate+SF 变换；DeepEP=all-to-all 与 engram 通信+NVL 域判定；TileKernels=gate/swiglu/量化/engram 哈希/PDL 开关；DeepSelect=topk（indexer+采样 logprobs 两处）。（day4 §四）
10. **fused_sample 融合了什么？收益？** 温度/top-k/top-p/log-softmax/确定性采样/reject sampling/accept_probs 单 kernel：vLLM 5–8 次 launch、2–3 次 [Q,vocab] 往返 → 1 次 launch、logits 只读一遍；reject 与 confidence 训练信号零额外成本。（day4 §5.6）
11. **Attention DP vs TP 的本质？** 切请求而非切张量：每 rank 全量权重独立 batch，数据面零通信（DP 组只有调度投票）；MLA 的 MQA + latent KV（576 元素/token）让 TP 切 KV 无收益而 all-reduce 纯开销；#DP=#EP 让 MoE all-to-all 完成聚合。（day5 §二）
12. **balanced CP 均衡在哪？** 三层：token 均衡（每 rank 每序列固定 2 块）；工作量均衡（zigzag 配对 (b, 2P−1−b)，(b+1)+(2P−b)=2P+1 恒定，度量是 ke−ks 压缩 key 数）；静态形状均衡（每 rank mirror 缓冲同长，-1 padding，graph 友好）。（day5 §3.2）
13. **EPLB 触发与粒度？** 步数计数达 interval（默认 0=关；examples 30；生产 prefill 1000/decode 10000），窗口 ≤ interval，首次提前 1/4；粒度 = physical expert 槽位级、逐层独立；机内 symm_mem IPC 镜像迁移，权重就位即热切换（gate 读新映射，无 barrier）。（day5 §五）
14. **varlen draft 与固定长度的区别？confidence 省掉什么？** 固定长度全员全验证；DSpark 用 ConfidenceHead 预测接受率，CPU 只定 total_q、GPU 上按累积率排序选前缀——省掉低概率 token 的**验证前向**（−30% 验证 token，接受长度不变 3.29 vs 3.20，+4~7%@60 并发）。（day6 §一）
15. **KEEP_TOKENS=140 怎么来？** 硬下界 = window 128 + backtrack max(·, nextn+1)=6 = 134（load_model assert；draft 全拒时回退后窗口仍完整）；部署值 140 = 134 + 6 余量（对齐 state ring 的 6 槽过冲吸收）；命中上限 L−134。（day6 §2.2）

---

## 六、今日总结（本周收官）

> 七天走完一条完整的认知弧线：Day 1 用 Q&A 建立系统级图景（三大目标、E/P/D、开源边界）；Day 2 钻进引擎（ZOS 四段式、C++ 调度器、shm 广播）；Day 3 落到 Worker（metadata 四套、KV 两池、双 graph runner）；Day 4 打穿模型与 kernel（11 步 attention、Mega MoE、store_kv_cache 逐行、fused_sample）；Day 5 纵览并行（DP/CP/EP/EPLB，拓扑函数化）；Day 6 收拢四个高级特性（DSpark、bounded replay、batch-invariant、SO——它们互相咬合）；Day 7 用测量纪律收口（阶段解耦、满接受口径、PPL bitwise 回归）。**最终沉淀的不是知识点而是方法**：文档定系统结论、代码定机制结论、测试定契约，三方互证且锚定 commit——这套方法下周换任何一个新框架都适用。

---

## 七、今日自测题

1. benchmark 里 N 怎么算？上限多少、触顶时发生什么？N>K 和 N≤K 时 batch 怎么组织？
2. `prefill_cached_prefix` 用了 Sequence 的哪个字段？它与 Day 2 的哪个机制呼应？
3. 满接受吞吐公式？"VARLEN=1 不可再乘 τ"的原因？
4. 测量窗对 LB flush 的两层防御是什么？`align_lb_stat_flush_phase` 为什么只护 profile 窗？
5. 为什么 decode profiler 之后不允许任何测量？
6. PPL golden 的三层 key 各自的原因？Ascend 上按 rank 分 key 为什么是"唯一确定性来源"？
7. 多节点 refresh 的锁保护什么操作？`--output` 为什么单节点限定？
8. profile_server 为什么没有停止接口？为什么只采 driver 一张卡？
9. graph_timer 怎么在不动图的前提下拿到图内分段耗时？
10. （收官题）用自己的话，一分钟讲清 DeepInfer 与 vLLM 的五个方向性差异。

<details>
<summary>参考答案（先自测再展开）</summary>

1. `N = min(ceil(prefill_total_steps × full_prefill_tokens / non_cached_tokens), 512)`——从"跑满 31 个满 chunk 需要多少 token"倒推；触顶打 WARNING（prefill 供给填不满稳态计划，测量不适合 profile）。N>K：下标 ≥K 的请求 max_tokens=1，prefill 完即逐出不进 decode；N≤K：batch 取 K，多余 prefill chunk 变 warmup。
2. `chunk_checkpoints=[cached_tokens]`（精确 prefill 到指定 token 数）；呼应 Day 2 的 `fork_seq → clone_states/clone_blocks`（源请求 state/块克隆给 fork 请求——1 次前缀 prefill + N−1 次克隆）。
3. `avg_SPS × K × (nextn+1) / cp`。VARLEN=1 时每 step 提交的 token 数随接受率变，avg_SPS 本身已含接受率，再乘 τ 会双重计入（plot_pareto 对此直接 assert）。
4. ① 测量区间撞 flush 的整段剔除（deque 的 flushed 标志）；② `align_lb_stat_flush_phase` 在 profile 窗前烧步把 flush 相位顶过去。只护 profile 窗因为：测量段可以用"剔除"兜底（丢至多一个 interval），而 profile 窗必须连续完整（trace 要整段稳定），且窗长 < flush 间隔才塞得进去（有 assert）。
5. profiler 退出后存在残余减速（实测 bsz=1 时 7–11%）——之后测的数系统性偏低。
6. platform：两平台 kernel/数值路径不同（连并行配置都不同：NVIDIA EP4/CP2 vs Ascend EP16/CP8）；workload：short/long 走不同代码路径；dp_rank：每 rank 请求子集固定（5×(r+1) 个）——Ascend 无 batch-invariant，固定请求集是确定性的唯一来源。
7. 读-改-写（读整份 JSON → 只改本节点 rank → 写回）：无锁并发会丢更新。`--output` 候选文件是一次性写出本进程结果，多节点下只有部分 rank，必然缺员。
8. POST /profile 采 N 步落盘，**阻塞到文件写完才返回**——请求本身即完整生命周期，无需停止。torch profiler 是进程内的，CP slave 在别的进程（且 slave 无完整采样路径），只能采 driver 那张卡。
9. capture 时 `keep_graph=True` 保留原始 cudaGraph_t，GraphHandle 匹配图内的**事件记录节点**；计时点用 `EventHandle.record(capture_external=True)` 嵌进 capture 变成图节点；replay 前 flip ping-pong 缓冲、replay 后 flush_out 汇总——图照常整体执行，时间戳由图内节点自己记。
10. 参考 5.2 的一句话总结 + 3.2 的四大差异表（调度语言 / 并行哲学 / 投机解码 / 确定性保障 + 测量纪律）。

</details>

---

## 八、今日产出物（本周最终交付物清单）

| # | 交付物 | 位置 |
|---|---|---|
| 1 | 一页纸架构图终稿 | 本日 5.1 SVG（Day 1 版本的迭代终稿） |
| 2 | 差异清单终稿（28 条带源码路径） | 本日 5.2 |
| 3 | kernel 地图 | day4 §八（decode step 全 kernel × 来源 × 输入输出） |
| 4 | 并行策略矩阵 | day5 §七（DP/CP/EP/ENP × 切分 × 通信 × 代码） |
| 5 | 四段三分钟口述稿 | day6 §六（DSpark / bounded replay / batch-invariant / SO） |
| 6 | 面试 Q&A 15 条 | 本日 5.3 |
| 7 | 请求生命周期时序图 + ZOS 甘特图 | day2 §3.5 / §4.2 |
| 8 | KV 物理布局图 + 调用链 | day3 §3.3 / §2.5 |
| 9 | DSpark 闭环图 | day6 §1.2 |
| 10 | benchmark 测量流程图 | 本日 §1.5 |

**Week 9 完**。下周（Week 10）可选方向：把 Day 3 的 KV 布局与 Day 4 的 kernel 地图对照 vLLM 做一次"如果移植"的纸上推演；或回头补 Week 6 量化日与 DeepInfer FP4 三处的交叉验证；或以本差异清单为骨架整理一篇对外分享。


