# Day 6：高级特性四连 —— DSpark 投机解码、SWA bounded replay、batch-invariant、结构化输出

> **Week 9 · DeepInfer 源码精读 · Day 6**
> 前置知识：Day 2（ZOS 的 seqlens 修正与 confidence kernel）、Day 3（`build_dflash_input`、StateBlock、KV 布局）、Day 4（fused_sample 的 reject 分支、DFlash 特征收集）；vLLM 的投机解码（Week 4）。
> 源码基线：commit `9bd9cf2`（2026-10-21）。今日主角：`models/mtp/dflash.py`（392 行）、`layers/mtp_info_pool.py`、`tilelang_ops/{schedule_mtp_varlen,dflash,structure_output}.py`、`input_metadata/yoco.py`、`structure_output/`（4 文件 595 行）、`examples/yoco/`。
> 今日用时：5~6 小时（本周信息密度最高的一天：四个特性各 1~1.5h）

---

## 今日学习目标

| # | 目标 | 验收标准 |
|---|---|---|
| 1 | DSpark 全流程 | 画出"主模型 forward → 验证 → draft → confidence → 下一步调度"的闭环图；解释 fixed-stride、MTPInfoPool 的槽位设计、varlen 验证的 GPU 调度两阶段 |
| 2 | bounded replay | 手算 140 的构成（下界 134 = 128 + 6）；讲清 encoder/decoder 两个应用点与 mask/compact 等价性 |
| 3 | batch-invariant | 列出六处保证点（FlashMLA / deep_gemm / tile_kernels / random_sampling / sum_smaller_probs / EPLB 冲突），解释确定性采样为什么必须关 MTP |
| 4 | 结构化输出 | 讲清三种 schema 的编译路径、matcher 的 trigger/bound/KMP/rollback、mask 在采样循环的施加点（含与 DSpark 的组合） |
| 5 | 四段三分钟口述稿 | 每个特性："问题是什么、DeepInfer 怎么解、开关是哪个环境变量"（README 指定产出） |

---

## 一、6a：DSpark + confidence-scheduled verification

### 1.1 DFlash 是什么：draft 侧解剖（`models/mtp/dflash.py`）

`DFlash` 从 HF checkpoint 的 MTP 段构造（主模型层数 + 2 之后的模块，全部必须是 `DFlashNextPredictHeadHC`，层数必须等于 `num_nextn_predict_layers`）。它的特殊之处在于**不是独立的小模型，而是与主模型共享器官的预测头**：

```python
# ① main_kv_proj：把主模型目标层特征投影成每层 MTP 的"主 token K"
main_kv_weights = [_clone_linear_weight(module.block.self_attn.Wkv.A.weight) for module in modules]
self.main_kv_proj = ReplicatedLinear(hidden, MTP_num_layers × qk_head_dim, weight=merge_linear_weights(...))
#    "main 和 draft 投影需要不同的连续布局，KV 权重会保留两份"
# ② projection：目标层特征（hidden × num_target_layers）→ hidden
# ③ noise_token_id + dflash_block_size = nextn：draft 是 fixed-stride 布局
#    （每请求固定 nextn 行：第 0 行 = golden token，其余填 noise token）
# ④ window 断言：draft 的 SWA 窗口必须与主模型一致（共用同一套 swa cache 布局）
assert self.window_size == getattr(cfg, "window_size", (-1, -1))[0]
# ⑤ markov（可选）：markov_w1（词表→低秩嵌入）+ markov_w2（低秩→词表 logits 修正）
# ⑥ ConfidenceHead：ReplicatedLinear(hidden + markov_rank → 1)，输出 sigmoid(z / temperature)
#    temperature 按 step 可配（DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE，标量或 per-step 元组）
```

### 1.2 一步 decode 的 DSpark 闭环（今日核心图）

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 760" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="a6" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#475569"/>
    </marker>
    <marker id="a6b" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect width="1120" height="760" fill="#fafbfc"/>
  <text x="560" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">DSpark 一步 decode 的闭环（ZOS + CUDA graph + MTP_VARLEN）</text>

  <!-- Step N: main forward -->
  <rect x="50" y="52" width="500" height="150" rx="10" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.8"/>
  <text x="300" y="76" text-anchor="middle" font-size="13.5" font-weight="700" fill="#581c87">① 主模型 forward（第 N 步，varlen 布局）</text>
  <rect x="66" y="90" width="220" height="34" rx="6" fill="#fae8ff" stroke="#a855f7"/>
  <text x="176" y="105" font-size="10.5" fill="#581c87">目标层收集特征（dflash_input_store，Day 4）</text>
  <text x="176" y="118" font-size="9.5" fill="#7e22ce">dflash_cfg.dspark_target_layer_ids</text>
  <rect x="306" y="90" width="228" height="34" rx="6" fill="#fae8ff" stroke="#a855f7"/>
  <text x="420" y="105" font-size="10.5" fill="#581c87">dflash.update_main_cache</text>
  <text x="420" y="118" font-size="9.5" fill="#7e22ce">projection → main_kv_proj → 写各 MTP 层主 token K</text>
  <text x="300" y="146" font-size="10" fill="#7e22ce">注释原文："全部 token 都按 slot_mapping_store_swa 写入：将被拒绝的 draft 位置正是 forward_drafts</text>
  <text x="300" y="162" font-size="10" fill="#7e22ce">随后写 draft key 的位置，在任何读取之前就会被覆盖，因此无需按接受结果屏蔽"</text>
  <text x="300" y="186" font-size="10" fill="#7e22ce">随主模型 CUDA graph 一起 replay（Day 3）</text>

  <!-- Step N: verification -->
  <rect x="590" y="52" width="480" height="150" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/>
  <text x="830" y="76" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1e3a8a">② sample 阶段：验证 + golden 生成（第 N 步）</text>
  <rect x="606" y="90" width="220" height="34" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="716" y="105" font-size="10.5" fill="#1e3a8a">prepare_verification_metadata</text>
  <text x="716" y="118" font-size="9.5" fill="#1e40af">row_to_req / local_step / is_verify</text>
  <rect x="846" y="90" width="210" height="34" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="951" y="105" font-size="10.5" fill="#1e3a8a">fused_sample（reject sampling）</text>
  <text x="951" y="118" font-size="9.5" fill="#1e40af">读 MTPInfoPool 的 draft logprobs</text>
  <text x="830" y="146" font-size="10" fill="#1e40af">get_accept_result：接受前缀 + 终止行；finalize_verification：</text>
  <text x="830" y="162" font-size="10" fill="#1e40af">last_valid_offset、num_{generated,verified}_tokens（按 nextn 桶）、</text>
  <text x="830" y="178" font-size="10" fill="#1e40af">batch_confidence_error=Σ|accept_prob−confidence|、emitted_stop_token</text>
  <line x1="550" y1="127" x2="590" y2="127" stroke="#475569" stroke-width="2" marker-end="url(#a6)"/>

  <!-- Step N: draft -->
  <rect x="590" y="230" width="480" height="168" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.8"/>
  <text x="830" y="254" text-anchor="middle" font-size="13.5" font-weight="700" fill="#14532d">③ DSpark draft 生成（仍在第 N 步的 sample 内）</text>
  <rect x="606" y="268" width="210" height="40" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="711" y="283" font-size="10.5" fill="#14532d">dflash graph replay（Day 3 双 runner）</text>
  <text x="711" y="298" font-size="9.5" fill="#166534">输入：golden + noise ×(nextn−1)（fixed-stride）</text>
  <rect x="836" y="268" width="220" height="40" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="946" y="283" font-size="10.5" fill="#14532d">N × DFlashDecoderLayer.forward_drafts</text>
  <text x="946" y="298" font-size="9.5" fill="#166534">draft attention 读 main cache + 写 draft K</text>
  <rect x="606" y="318" width="450" height="34" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="831" y="333" font-size="10.5" fill="#14532d">sample_drafts：逐步 logits（markov 修正）+ confidence_head_fused（TileLang sigmoid GEMV）</text>
  <text x="831" y="347" font-size="9.5" fill="#166534">process_sampling(tag="mtp"：独立 RNG 命名空间) → 逐 token 采样 + logprobs</text>
  <text x="830" y="372" font-size="10" fill="#166534">sample_positions_table = arange(1..nextn) + sample_positions + last_valid_offset ← ②的接受结果直接定位</text>
  <text x="830" y="388" font-size="10" fill="#166534">遇 EOS（②的 emitted_stop_token）：confidence_rates 清零 → 下轮 draft 拿不到验证预算</text>
  <line x1="830" y1="202" x2="830" y2="230" stroke="#475569" stroke-width="2" marker-end="url(#a6)"/>

  <!-- pool -->
  <rect x="50" y="230" width="500" height="168" rx="10" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.8"/>
  <text x="300" y="254" text-anchor="middle" font-size="13.5" font-weight="700" fill="#713f12">MTPInfoPool：按调度器槽位的稳定 GPU 存储（mtp_info_pool.py）</text>
  <text x="70" y="280" font-size="10.5" fill="#854d0e">token_ids[rows]、logprobs[rows, force_topk or vocab]、confidence[rows]；rows = (max_num_seqs+1) × nextn</text>
  <text x="70" y="298" font-size="10.5" fill="#854d0e">★ 槽位 0 = padding 保留（logprobs=−inf、pad token=0 → 永不接受）；write_slots=0 的请求跳过写入</text>
  <text x="70" y="316" font-size="10.5" fill="#854d0e">作用：draft 的产物按"调度器槽位"而非 batch 行号存取 —— 验证侧（②）用 mtp_info_read_rows</text>
  <text x="70" y="334" font-size="10.5" fill="#854d0e">跨 step 读上一轮的 draft，batch 组成变化也不换数据；force_topk=4096 把 logprobs 显存压到 1/30</text>
  <text x="70" y="356" font-size="10.5" fill="#854d0e">store_mtp_info（TileLang）按槽位散射写；placeholder 请求的 draft 恒指向 slot 0</text>
  <line x1="550" y1="314" x2="590" y2="314" stroke="#2563eb" stroke-width="2" marker-end="url(#a6b)"/>
  <text x="570" y="308" font-size="9.5" fill="#2563eb">③ 写入</text>
  <line x1="590" y1="360" x2="550" y2="360" stroke="#2563eb" stroke-width="2" marker-end="url(#a6b)"/>
  <text x="570" y="378" font-size="9.5" fill="#2563eb">② 读取</text>

  <!-- step N+1 scheduling -->
  <rect x="50" y="430" width="1020" height="140" rx="10" fill="#fff7ed" stroke="#ea580c" stroke-width="1.8"/>
  <text x="560" y="454" text-anchor="middle" font-size="13.5" font-weight="700" fill="#7c2d12">④ confidence-scheduled verification：下一步的验证预算在 GPU 上分配（第 N+1 步开头，ZOS）</text>
  <rect x="66" y="470" width="480" height="38" rx="6" fill="#ffedd5" stroke="#ea580c"/>
  <text x="306" y="485" font-size="10.5" fill="#7c2d12">Phase 1（cumprod）：acc = Π confidence_rates[remap[i], 0..j]，新请求 acc=0</text>
  <text x="306" y="500" font-size="9.5" fill="#9a3412">remap scatter 融合在读入时完成（cuda/schedule_mtp_varlen.py:26-36）</text>
  <rect x="566" y="470" width="490" height="38" rx="6" fill="#ffedd5" stroke="#ea580c"/>
  <text x="811" y="485" font-size="10.5" fill="#7c2d12">Phase 2（rank counting）：数全 batch 里累积率比自己高的位置数</text>
  <text x="811" y="500" font-size="9.5" fill="#9a3412">"Higher rate wins; tie-break by flat index" —— 即全局排序选 top-k 前缀</text>
  <text x="560" y="532" font-size="10.5" fill="#9a3412">seqlens_q[i] = 预算内 i 拿到的验证行数；Σ seqlens_q = total_q（CPU 只定总额，Day 2 的四段式管线第 ① 段）</text>
  <text x="560" y="550" font-size="10.5" fill="#9a3412">预算本身来自双层硬件感知：启动时 profile 的 decode_perf_curve + 运行期 update_mtp_runtime 实测校正（Day 3/4）</text>
  <line x1="560" y1="398" x2="560" y2="430" stroke="#475569" stroke-width="2" marker-end="url(#a6)"/>
  <text x="575" y="418" font-size="9.5" fill="#475569">confidences 常驻 GPU（Day 2 的 *_gpu 副本）→ 进入下一循环</text>

  <!-- numbers band -->
  <rect x="50" y="590" width="1020" height="150" rx="10" fill="#f8fafc" stroke="#94a3b8"/>
  <text x="560" y="616" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">Q&amp;A 实测数字（EP64/DP64，生产工作点 ~60 并发/卡）</text>
  <text x="560" y="642" text-anchor="middle" font-size="11.5" fill="#334155">每步验证的 draft token 数：fixed-5 = 5.00 → dynamic = 3.2~4.4（砍约 30%）</text>
  <text x="560" y="664" text-anchor="middle" font-size="11.5" fill="#334155">平均接受长度基本不变：3.29（fixed）vs 3.20（dynamic）——省掉的是"低概率 token 的验证计算"</text>
  <text x="560" y="690" text-anchor="middle" font-size="11.5" fill="#334155">吞吐增益随并发上升：16 并发 −1.5% → 32 并发 +1.6% → 60 并发 +5.4% → 96 并发 +11.0%</text>
  <text x="560" y="716" text-anchor="middle" font-size="10.5" fill="#64748b">并发低时 varlen 的额外调度开销吃掉收益；并发高时省下的验证计算换成有效吞吐</text>
</svg>
```

### 1.3 reject sampling 的数学（`tilelang_ops/common.py` 文件头的完整证明）

draft 分布 `p`、target 分布 `q`，先从 `p` 采样，`p[j] ≤ q[j]` 直接接受，否则从 `r[i] = max(q[i] − α·p[i], 0)`（`α = exp(−DRAFT_PROB_PENALTY)`）归一化后重采样。文件头给出逐行证明：令 `eps = 1 − α`，则实际输出分布 `A[j]` 满足

```
q[j] − eps/(1+eps)  ≤  A[j]  ≤  (1+eps) · q[j]      （对所有 token j）
```

`DRAFT_PROB_PENALTY = 5e-5`（注释："防止 reject sampling 采不出东西"）——**A 与 q 的偏差被钳在 5e-5 量级**，这是 DSpark 不掉点（decode PPL 基线可回归）的数学根基。`fused_sample` 顺手产出的 `accept_probs = max(1 − Σ r_i, 0)` 就是 confidence head 的训练信号，`batch_confidence_error = Σ|accept_prob − confidence|` 是它的在线校准误差（每步进 stats，Day 2 见过）。

### 1.4 对照 Week 4 的 vLLM 投机解码（README 关键问题）

| 维度 | vLLM V1（MTP/EAGLE/ngram） | DeepInfer DSpark |
|---|---|---|
| draft 长度 | 固定（`num_speculative_tokens`），全员统一 | **varlen**：confidence-scheduled，per-request 由 GPU 上的预算分配决定 |
| 验证计算 | 固定长度全验证（浪费在低概率 token 上） | 只验证期望收益最高的前缀（省 30% 验证 token） |
| draft 模型 | 独立模块（EAGLE 头/MTP 层） | 与主模型共享器官：目标层特征 → `main_kv_proj` 直写 draft KV（无独立 KV 重算） |
| draft 的 KV | 单独维护 | 复用主模型 SWA cache 布局（window 断言），main K 由主模型 forward 顺路写入 |
| draft 结果存储 | 跟随 batch 张量 | **MTPInfoPool 按调度器槽位常驻**（batch 组成变化不换数据，天然配合 ZOS） |
| 调度时机 | CPU 侧（上一轮结果回来后） | CPU 只定 total_q，**per-request 分配在 GPU 上**（Phase1 cumprod + Phase2 rank counting）——因为 ZOS 下 schedule 时上一步 confidence 还没回来 |
| 与 graph | draft/target 各自捕获 | 同样双 runner，但 draft 图与主图共享 pool/stream（Day 3） |

**"confidence-scheduled verification 省掉了什么"的准确答案**：省掉的是**低接受概率 draft token 的验证前向**（QoC 原文："fixed-length verification wastes verification compute on low-probability tokens"）。代价是多了一个 ConfidenceHead（一次融合 GEMV，TileLang）和 GPU 侧的调度 kernel——两者都被做进了捕获的 graph 里，边际成本近零。

---

## 二、6b：SWA bounded replay 与 140 的手算

### 2.1 两个应用点（Q&A 回顾 + 代码对位）

| 应用点 | 语义 | 计算量 |
|---|---|---|
| **Encoder SWA bounded replay** | prefix cache 命中时，若 SWA KV cache 缺失，重算命中 prefix 的**最后 128 token** 并与新 suffix 一起前向；重算段只重建 SWA KV，block KV 原样复用 | 常数（与命中长度无关） |
| **Decoder SWA bounded replay** | CED 下 decoder 层的 block KV 从最后 encoder 层投影，精确重建需 **2560 token × 20 层**；实际每轮 prefill 只重算最后 128 token，结果仅用于 decode、**不进 prefix cache** | prefill 总计算近乎减半 |

核心机制（Q&A 原文）："replaying from position s, the query at position i is computed only against keys in the range max(s, i−127) to i"——重算成本与上下文长度和层数都无关。

### 2.2 `KEEP_TOKENS=140` 的手算（README 关键问题）

**硬下界（代码 assert，`model_runner.py:110-113`）**：

```
KEEP_TOKENS ≥ window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS
            = 128        + max(显式值, nextn + 1)
            = 128        + 6                      （nextn=5 → 6）
            = 134
```

为什么必须覆盖"窗口 + 回退"：YoCo consumer 层的 SWA 需要末尾 128 个真实 token 的可见域；decode 期 DSpark 的 draft 全拒时状态要**回退最多 nextn+1 = 6 个位置**（回到主 token 之前），回退后窗口左端必须仍有完整 128 token 在保留区内。这正是 Q&A 命中策略的 `L − 134` 上限（"128-token window + 6 tokens of speculative backtracking"）。

**部署值 140 = 134 + 6**：`environ.py` 明说 KEEP_TOKENS "默认 0，启用截断时必须显式设置"；README/QA 的"default V4.1 settings"是 140。多出的 6 与 Q&A 里 state ring 的设计对齐——"copy position can overshoot the boundary by at most 5 tokens, which is absorbed by the 6 extra positions"（decode 侧落盘在跨越 128 边界时的过冲吸收）。**代码只锁死下界 134；140 的 +6 是部署余量**（`get_yoco_store_extra_tokens() = KEEP_TOKENS` 也拿 140 参与落盘 ring 的容量计算）。

配套的窗口参数全家福（`cache_engine.py`）：

```
swa_block_size_inference = window 128 + backtrack 6 = 134        ← state ring 的推理窗口
swa_block_size_kvstore   = 134 + (EXTRA_SIZE 128 + yoco 140)     ← 仅 PREFILL_STORE_EXTRA_STATE=1（默认关）
KVSTORE_GRAIN = 128                                               ← 落盘只在 128 倍数处
```

### 2.3 截断的实现：`yoco_swa_visibility` 的 C++ 语义（`prepare_tensor_input_v4.cpp:149-168`）

```cpp
int32_t abs_lo = std::clamp(std::max({pos - window_size + 1, rect_lo, valid_start}), 0, pos + 1);
// rect_lo = data_len - keep_tokens（截断矩形下界）；valid_start = zero-state 重算的清零下界
// abs_lo == pos + 1 表示空可见域（整行 -1）
sec1（swa state 段）：start_slot 起的 window−start_slot 个 key
sec2（本 chunk 段）：k_lo 起的 new_len 个 key —— TORCH_CHECK_GE(k_lo, clip_prefix) 保证可见 key 不越过保留区
```

每 token 的可见域 = **滑窗 ∩ 保留矩形 ∩ 有效 state** 三者取 max 下界——mask 与 compact 两种模式共用这一函数（yoco.py 文件头："两模式的 SWA 可见域同式（C++ 共用 yoco_swa_visibility），故 attention 数值在保留区间内一致"），差别只在：mask 保留全部 query 行（只把 pivots 换成截断版，`get_mask_view`）；compact 真正删行（`get_consumer_view` 切 compact 布局，Day 3/4 已见层循环边界的双路径）。

### 2.4 等价性验证：`examples/yoco/`（无硬件可跑 L1！）

| 级别 | 命令 | 成本 | 验证内容 |
|---|---|---|---|
| L1 | `python examples/yoco/yoco_equiv.py mask-compact` | **单卡、秒级、不加载权重** | 两模式的 SWA pivots、可见 key 数、store slots、压缩 key 范围逐项相等 |
| L2 | `examples/yoco/run_yoco_equiv_l2.sh --model_name ... [--mtp 5]` | 8 卡 + HF 权重 | 固定 teacher-forcing 轨迹下 logprob/top-K 逐元素相等；`--mtp N` 额外**逐位比较 confidence**（"draft hidden 对 main cache 做 attention 的直接函数"——直接探测 DFlash main cache 的写入等价性） |

两个值得抄的工程细节：L2 要求"除截断模式外所有 `DEEPINFER_*` 环境变量两轮一致"；MTP 的 confidence "滞后一轮挂载，第一轮是 None，最后一轮的 confidence 到不了输出"——**等价性验证连投机解码的内部状态都对齐到比特级**。

---

## 三、6c：batch-invariant 与确定性采样

### 3.1 定义与开关

> **batch-invariant**：batch 的组成方式（同批还有谁、排第几行、batch 多大）不改变每个请求的计算结果。

开关：`DEEPINFER_BATCH_INVARIANT=1`（**仅 NVIDIA**；`runtime_config.py:63` 对 NPU 直接 assert 关闭）。examples 里 NVIDIA 平台 setdefault 开启（Day 2 见过）。

**确定性采样配方**（README）：`BATCH_INVARIANT=1` + `MTP_VARLEN=0` + 固定 `seq_seed`。为什么必须关 MTP？——varlen 验证的预算按 **全 batch** 的 confidence 排序分配（Phase 2 的 rank counting），同一请求换个 batch 邻居，拿到的验证长度就变，接受结果随之漂移。**confidence-scheduled verification 本质是 batch 依赖的**，这是特性之间的真实冲突（如同 EPLB 的多副本归约冲突）。

### 3.2 六处保证点（README 关键问题："在哪些 kernel 上保证"）

| # | 位置 | 保证机制 |
|---|---|---|
| 1 | FlashMLA（`attention_backend.py:141` → `attn.py:666/1002`） | `enable_batch_invariant` 直通 vendor kernel（decode 与 DFlash 两条路径） |
| 2 | deep_gemm / tile_kernels（`runtime_config.py:67`） | `use_deterministic_algorithms(True)`——确定性 GEMM 分割/归约 |
| 3 | TileLang `random_sampling`（Day 4） | SplitMix64 哈希 `(seq_seed, sample_position)`，"independent of batch size and row order"；`_hash_to_uniform_ref` 逐位复现 |
| 4 | TileLang `sum_smaller_probs` | 固定求和顺序：按概率升序、同概率按列序——消除并行归约顺序不确定性 |
| 5 | EPLB 冲突面（`userbuffer.py:79`） | `allow_multiple_reduction = not (BATCH_INVARIANT and REARRANGE>0)`——多副本归约顺序不确定，同开时强制单归约 |
| 6 | 融合路径选择（`attn.py:88-91`） | `use_fused_norm_rope_attn_rope_cast` 在 `(not BATCH_INVARIANT and PROMPT) or sparse or not packed_kv` 才开——**PROMPT 引擎的 dense packed-KV 融合路径在 batch-invariant 下被禁用**（融合 kernel 未做不变性保证，宁可退回慢路径） |

第 6 条是"性能 vs 不变性"取舍的直接证据：**batch-invariant 不是免费的**，个别融合 kernel 需要退路。NPU 侧还有 `rstd_1ulp.py`（norm 做到 1-ULP 舍入一致）但整体不支持该开关。

### 3.3 验证测试（README："tests/ 里找验证测试"）

| 测试 | 验证内容 |
|---|---|
| `tests/models/test_batch_invariant.py` | **主测试**："running generate on the same prompt batched and singly gives identical results"——64 条随机长度 prompt（正态 2500±1000），EP8、nextn=5，比较 (token_ids, logprobs, top_token_ids, top_logprobs) 四元组逐项相等 |
| `tests/models/test_ve_batch_invariant.py` | **视觉编码器**的 batch 不变性（同一图片换 batch） |
| `tests/models/test_sampler.py` | 采样器确定性 |
| `tests/models/test_kl.py` | 数值分布层面的回归 |
| 各 TileLang wrapper 的 `_ref` | kernel 级参考实现逐位对照（`random_sampling_ref` 用 int64 模拟 uint64） |

### 3.4 对照 Thinking Machines 的 batch-invariant 博客

博客的核心论点（Week 6 量化日的延伸阅读）：batch 内 reduction 顺序、split-k 切分、kernel 并行度都随 batch 组成变化，导致"同一请求在 batch=1 和 batch=64 下 logprob 不同"——对评测可复现性和缓存一致性是灾难。解法是把 kernel 写成 batch 无关：固定切分、顺序无关归约、每行独立 RNG。**DeepInfer 是这套哲学的完整工程化**：从 vendor（FlashMLA/deep_gemm）到自研 TileLang 到引擎层（ZOS 的 remap 修正也保持数值一致、EPLB 冲突规避）纵向打通，并用"batched vs singly 全字段相等"的端到端测试锁死。差别在于 Thinking Machines 讨论的是通用 LLM 服务的 invariance，DeepInfer 只为单一模型维护这套保证——**又一处"专用 vs 通用"的取舍**。

---

## 四、6d：结构化输出（SO）

### 4.1 编译：三种约束 → xgrammar（`structure_output/compiler.py`）

```python
if   schema_type == "json":  ctx = compiler.compile_json_schema(schema, strict_mode=True)
elif schema_type == "regex": ctx = compiler.compile_regex(schema)
elif schema_type == "ebnf":  ctx = compiler.compile_grammar(schema)
```

编译在**独立线程池**（`SOMatcherBuilder`，`get_idle_cpus` 绑核、max_workers=空闲 CPU 数）异步进行：`add_request` 时 `submit(seq, so_args, _enqueue)`（Day 2 见过），`pre_sampler_sync` 里 `pump()` 轮询把编好的 matcher 挂到 seq 上入调度——**编译慢不阻塞引擎，编不完不进 batch**。`XGrammarMatcherFuture` 带超时（默认 20s）与 abort 语义。安全细节：`DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES`（默认 10GB）——注释点名 **CVE-2025-32381**（xgrammar 缓存无上限导致 OOM 的 DoS 向量），并保留 kill switch。

### 4.2 matcher：trigger / bound / KMP / rollback（`matcher.py`）

`XGrammarMatcher` 在 xgrammar 的 `GrammarMatcher` 外包了一层**状态机**：

- **`trigger_token_ids`**：触发序列出现前自由生成（内部用 **KMP 增量匹配**跟踪进度，`_kmp_failure` 预处理——与 Day 1 抄录的 deepseek-recipe 流式解析同一技术）；
- **`bound_token_id`**：bound 触发后"不再允许自由 token，先逐 token 强制走完 trigger 序列再进 grammar"；
- **`rollback(steps)`**：**回滚容量 = `MAX_ROLLBACK_TOKENS = nextn + 2`**——为 DSpark 的验证回退预留！draft token 被 trial-accept 后可能整段回退，matcher 状态必须能倒回去（trigger 进度用 `_progress_history` 环形栈恢复，超容量直接 raise）；
- EOS 后的 `mask_after_eos`（全 1 mask）：语法已终结，放行一切（让 EOS 语义自然收尾）。

### 4.3 mask 的准备与施加（README 关键问题："在哪一步施加"）

**准备**（`sampler._get_so_mask:302-408`，每步）：

```python
# ① matcher 前进：接受上一步的 last_accepted_token + last_output_token（真被接受进序列的 token）
# ② 试探 draft：对 last_next_n_token 逐个 accept_token → 为每个 draft 步 k 填 bitmask[k+1]
#    （fill_vocab_mask：trigger 未完 → 单 token mask；已 EOS → 全 1；否则 inner.fill_next_token_bitmask）
# ③ matcher.rollback(num_accepted_steps)   ← ★ 试探完回滚，draft 还没被验证！
# ④ bitmasks（max_MTP_value+1 份）与 indices 在专用 so_copy_stream 上 H2D，
#    so_mask_ready 事件与 logits 计算重叠（Day 4 见过）
```

**施加**（`sampler._verification:433-437`）：

```python
for step, indices in enumerate(so_indices):          # step = 0 是主 token，1..nextn 是各 draft 步
    if len(indices):
        apply_token_bitmask_by_step_inplace(logits, so_bitmasks[step], indices, ftp, step)
```

**施加点 = logits 上、`process_sampling` 之前，且按 draft 步分桶**（`ftp` 定位每个请求的 packed 行、`step` 定位行内偏移）——每个潜在 draft 位置施加它自己的语法约束。kernel 是 TileLang 的 `apply_token_bitmask_by_step_inplace`（按 indices 稀疏行处理，不碰非 SO 请求的行）。prefill 分支在采样行上施加单份 mask（`sampler.py:187-188`）。

**SO × DSpark 的组合是这个设计的精华**：draft 步 k 的 mask 是"假设前 k−1 个 draft 全被接受"的语法状态——验证时 target 在这些 mask 下采样，接受的 token 天然合法；若 draft 被拒，matcher 早已回滚，无副作用。

### 4.4 故障隔离哲学

`update_so_status` 的异常处理注释（sampler.py:377-386）："此处在 CP rank 0 采样主路径，raise 会连坐所有请求 + NCCL barrier 超时；**SO 的 bug 只判死本请求，不带崩 engine**"——`so_status=-1` → `FINISHED_SO_FAILED`。整个 SO 子系统的容错（编译超时、matcher 异常、replay 失败）都收敛到"杀单个请求"。

---

## 五、动手实验

### 实验 1（无硬件，可完整执行）：跑 YoCo L1 等价性

```bash
python examples/yoco/yoco_equiv.py mask-compact          # 秒级、无权重、单卡
python examples/yoco/yoco_equiv.py mask-compact --keep-tokens 140 --window 128
```

对照 2.3 节的 `yoco_swa_visibility` 读输出——这是本周**唯一无硬件也能"跑"的核心特性验证**。

### 实验 2：读测试反推契约

- `tests/models/test_batch_invariant.py`：注意 `use_nextn_decode=5` 的 case——**MTP 开着也测 batch 不变性**（MTP 的验证是 batch 依赖的，但接受判定本身在 batch-invariant kernel 下确定？读 `_get_result` 比较的四元组想清楚边界在哪）；
- `tests/models/test_offline_inference.py` 的 `--check_structured_output` 路径（Day 2 见过）：三个 schema（regex `[A-Z]{2}-[0-9]{3}`、JSON enum schema、EBNF `answer=(yes|no);level=[1-3]`）的断言 = **SO 的行为规格**（`examples/_structured_output.py`）。

### 实验 3：手算闭环

1. **140**（2.2 节）：128 + max(·, 5+1) = 134 下界，+6 部署余量；
2. **reject sampling 偏差**：`eps = 1 − exp(−5e-5) ≈ 5e-5`，输出分布被钳在 `q[j] ± 5e-5` 量级；
3. **MTPInfoPool 显存**：max_num_seqs=256、nextn=5、force_topk=4096 → rows = 257×5 = 1285，logprobs 1285×4096×4B ≈ 21MB（+index 同量级）——对比不开 force_topk：1285×129280×4B ≈ 665MB。**force_topk 的真实动机是池显存**；
4. **draft 的固定形状**：dflash_block_size=5 → draft 图的 batch 键是 req_cnt（Day 3），fixed-stride 每请求恒 5 行。

---

## 六、面试高频问题（四段"三分钟版"口述稿，README 指定产出）

**Q1 · DSpark（问题→解法→开关）**

> 问题：投机解码的固定长度验证把计算浪费在低接受概率的 draft token 上。DeepInfer 的解法：draft 模型 DFlash 与主模型共享器官——主模型 forward 顺路把目标层特征经 `main_kv_proj` 写成 draft 的主 token K，draft 用 fixed-stride 布局（golden + noise × nextn−1）跑 N 层自回归；**ConfidenceHead（融合 sigmoid GEMV）预测每个 draft token 的接受率**，调度时 CPU 只定总预算（total_q），per-request 验证长度在 GPU 上分配：累积接受率 cumprod → 全局 rank counting 选 top-k 前缀（新请求接受率 0）。draft 结果按**调度器槽位**存进 MTPInfoPool（batch 换血不丢数据），验证在 fused_sample 里以 reject sampling 一并完成（数学上输出分布与 target 偏差 ≤ 5e-5），顺手产出 accept_probs 作为 confidence 的训练信号。实测：验证 token 砍 30%（5.00→3.2-4.4）、接受长度不变（3.29 vs 3.20）、生产工作点 +4-7% 吞吐（96 并发 +11%）。开关：`DEEPINFER_ENABLE_MTP_VARLEN=1`、`DEEPINFER_USE_NEXTN_DECODE`（draft 长度，默认 5）。

**Q2 · SWA bounded replay**

> 问题：YoCo/CED 架构下 decoder 层的 SWA KV 精确重建需要 2560 token × 20 层，prefill 成本翻倍。解法：只重放**有界后缀**——从位置 s 重算最近 128 token，query i 只看 `[max(s, i−127), i]` 的 key，重算成本变成与上下文和层数都无关的常数；结果只用于 decode、不进 prefix cache（避免污染）。两个应用点：encoder 侧（SWA miss 时回退一块重算，block KV 原样复用）、decoder 侧（prefill 计算近乎减半）。保留后缀必须 ≥ 窗口 128 + 回退 6（nextn+1，draft 全拒时的状态回退）= 134（代码 assert 的硬下界），部署默认 140 = 134 + 6 余量（对应 state ring 的 6 槽过冲吸收）。命中策略上限 L−134。mask 与 compact 两模式共用 `yoco_swa_visibility`（滑窗 ∩ 保留矩形 ∩ 有效 state），L1/L2 等价性测试做到 logprob 与 confidence 逐位对齐。开关：`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` + `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`。

**Q3 · batch-invariant 与确定性采样**

> 问题：batch 组成改变 reduction 顺序/split-k/RNG 状态，同一请求换 batch 结果就变——评测不可复现。解法：六处纵向保证——FlashMLA 开 `enable_batch_invariant`、deep_gemm/tile_kernels 开确定性算法、`random_sampling` 用 SplitMix64 哈希 (seq_seed, position)（与 batch 和行序无关）、`sum_smaller_probs` 固定求和顺序（概率升序、同概率按列序）、EPLB 多副本归约在冲突时强制单归约、个别未保证的融合路径（PROMPT dense packed-KV）直接禁用退回慢路。端到端测试：同一 prompt batched vs singly 的 (token, logprob, top-k) 四元组逐项相等。确定性采样配方 = batch-invariant + 关 MTP（varlen 验证预算按全 batch confidence 排序分配，本质 batch 依赖）+ 固定 seq_seed。开关：`DEEPINFER_BATCH_INVARIANT=1`（仅 NVIDIA）。

**Q4 · 结构化输出**

> 问题：regex/JSON schema/EBNF 约束生成，且不能拖慢主路径。解法分三层。**编译**：三种 schema 统一走 xgrammar 编译成 grammar 自动机，在独立绑核线程池异步进行（编译慢不阻塞引擎、编不完不入队），缓存带字节上限（防 CVE-2025-32381 的 OOM DoS）。**运行时**：matcher 外包状态机——trigger_token_ids 用 KMP 增量匹配（触发前自由生成）、bound_token_id 强制走完触发序列、**回滚容量 nextn+2 为 DSpark 验证回退预留**；每步先接受上一步真 token，再试探性接受 draft 为每个 draft 步填 mask，然后回滚（draft 未验证）。**施加点**：logits 上、采样前，按 draft 步分桶（`ftp + step` 定位 packed 行）用 TileLang kernel 稀疏施加——每个潜在 draft 位置有自己的语法约束，SO × DSpark 天然组合。容错哲学：SO 的任何 bug 只杀单个请求（so_status=-1），不带崩引擎。开关：`SamplingParams(so_args={"schema_type", "schema"})`。

---

## 七、今日总结（一分钟串联四特性）

> 四个特性其实是**一组互相咬合的齿轮**：DSpark 让 decode 每 step 的 token 数变成 batch 依赖的变量 → ZOS 用 GPU 侧的 seqlens 修正消化它（Day 2）→ CUDA graph 用 total_q 键和 fixed-stride draft 图容纳它（Day 3）→ bounded replay 保证 draft 回退 6 个位置后 SWA 窗口仍然完整（134 下界）→ SO 的 matcher 用 nextn+2 的回滚容量跟着 draft 一起回退 → 而 batch-invariant 与这一切都有张力：varlen 调度是 batch 依赖的、EPLB 多副本归约是顺序不确定的——所以确定性采样必须关 MTP，融合路径宁可退回慢路。**DeepInfer 的工程哲学在这一天最清楚：每个特性都不是孤立的点，是牵动引擎、kernel、调度、测试四个层面的纵向改造。**

---

## 八、今日自测题

1. `main_kv_proj` 的输入是什么？为什么 draft 的主 token K 可以由主模型 forward "顺路"写出？
2. draft 的输入序列长什么样（fixed-stride）？第 0 行和其余行分别是什么？
3. MTPInfoPool 为什么按"调度器槽位"而不是 batch 行号索引？slot 0 的特殊内容是什么？
4. confidence-scheduled verification 的 GPU 调度 kernel 的两个 phase 各做什么？"全局排序"是怎么实现的？
5. reject sampling 的输出分布与 target 分布的偏差上界是多少？由哪个常数控制？
6. `KEEP_TOKENS` 的硬下界公式？140 里的 +6 对应什么设计？
7. mask 模式与 compact 模式的 SWA 可见域为什么相等？哪个函数保证的？
8. 确定性采样为什么要关 MTP？哪两个特性之间存在类似冲突（提示：归约）？
9. SO 的 draft 步 mask 是怎么准备的？为什么准备完要 rollback？
10. `apply_token_bitmask_by_step_inplace` 的 `ftp` 和 `step` 参数各起什么作用？

<details>
<summary>参考答案（先自测再展开）</summary>

1. 主模型指定层（`dspark_target_layer_ids`）的 attention 输入特征（`dflash_input`，Day 4 的 `_store_dflash_input` 收集），经 `projection`（多层拼接 → hidden）+ `projection_norm`。因为 draft 的 KV 布局与主模型 SWA cache 完全一致（window 断言），主 token 的 K 就是这些位置的 KV——由主模型算 hidden 时顺路投影写入（`update_main_cache`，随主图 replay）；被拒 draft 的位置随后被 `forward_drafts` 写的 draft K 覆盖，无需按接受结果屏蔽。
2. 每请求 `dflash_block_size = nextn` 行：第 0 行 = golden token（上一步验证后的采样 token），其余 = `dspark_noise_token_id` 填充（noise 条件生成）。
3. 验证发生在下一步、batch 组成可能已换（ZOS 晚一步）；按调度器槽位（`mtp_info_write_slots`）存储则 draft 数据与 batch 行号解耦，`mtp_info_read_rows` 跨 step 稳定读取。slot 0 = padding 保留：logprobs 全 −inf、pad token（vocab−1）处置 0——placeholder/未产出 draft 的请求读它，永不产生有效接受。
4. Phase 1：按 remap 读上一步 confidence_rates 做 cumprod（新请求累积率 0），remap scatter 融合在读入；Phase 2：每个 block 数全 batch 里累积率比自己高的 (seq, pos) 个数（"Higher rate wins; tie-break by flat index"）——rank counting 即全局排序，rank < 预算的位置构成 top-k 前缀，`seqlens_q` 即每请求分到的验证行数。
5. `q[j] − eps/(1+eps) ≤ A[j] ≤ (1+eps)·q[j]`，`eps = 1 − exp(−DRAFT_PROB_PENALTY) ≈ 5e-5`（common.py 文件头的完整证明）。
6. `KEEP_TOKENS ≥ window_size + MAX_BACKTRACKING = 128 + max(·, nextn+1) = 128+6 = 134`（model_runner.py assert）。+6 是部署余量，与 QA 的 state ring "6 extra positions 吸收 ≤5 token 过冲"对齐。
7. 两种模式都调 `yoco_swa_visibility`（C++ 共用）：可见域 = max(滑窗下界, rect_lo, valid_start)。mask 只换 pivots 不删行；compact 删行但保留区内逐 token 的可见域相同——所以保留区间内 attention 数值一致（yoco.py 文件头 + L1/L2 测试锁死）。
8. varlen 验证预算按全 batch 的 confidence 排序分配（batch 依赖）。类似冲突：batch-invariant × EPLB——多副本归约顺序不确定，`allow_multiple_reduction` 在两者同开时强制 False（单归约路径）。
9. `_get_so_mask` 先接受上一步真 token，再对 `last_next_n_token` 逐个 trial-accept 并为第 k 个 draft 步填 `bitmask[k+1]`，然后 `matcher.rollback(num_accepted_steps)`——因为 draft 尚未被验证，matcher 状态不能真的前进；mask 是"假设前面全被接受"的预制品。
10. `ftp`（first_token_positions）：每个请求在 packed logits（[total_q, vocab]）里的首行位置——定位行；`step`：行内偏移（主 token 为 0，draft 步 k 为 k）——定位该步的 draft 行。两者合起来把 per-request × per-step 的 mask 稀疏地施加到正确的 logits 行。

</details>

---

## 九、今日产出物

1. **四段三分钟口述稿**（第六节 Q1-Q4，README 指定产出）。
2. **DSpark 闭环图**（1.2 节 SVG）+ **140 手算**（2.2 节）。
3. **差异清单 v1.5**（在 Day 5 的 24 条之上补充 4 条）：

| # | 维度 | vLLM V1 | DeepInfer | 源码证据 |
|---|---|---|---|---|
| 25 | 投机解码 | 固定长度 draft + 全验证（ngram/EAGLE/MTP） | confidence-scheduled varlen + GPU 侧预算分配 + 槽位池 | dflash.py、schedule_mtp_varlen.py、mtp_info_pool.py |
| 26 | SWA/前缀重算 | prefix caching 全命中（无 bounded replay） | 有界重放：常数成本重建 SWA KV，KEEP=140 | prepare_tensor_input_v4.cpp、cache_engine.py |
| 27 | 确定性 | 无系统保证（RNG/归约随 batch 漂移） | 六处 batch-invariant 保证 + 确定性采样配方 | test_batch_invariant.py、runtime_config.py |
| 28 | 结构化输出 | xgrammar 集成（mask 在采样前） | 同源 xgrammar + **per-draft-step mask + matcher 回滚**（与投机解码组合） | structure_output/、sampler.py:_get_so_mask |

**明日预告（Day 7）**：性能复现与复盘——`benchmark.py` 的阶段解耦测量（`--prefill_batch_size` / `--decode_num_requests` 的 N/K 独立性）、sweep 与 Pareto 复现、eval 链路、PPL 基线按 platform/workload/DP rank 分 key 与文件锁、`profile_server` / `graph_timer` 工具链；最后对照 Day 1 的差异清单逐项收口，并回答"如果让你给 DeepInfer 补一个 prefix caching 路由层，接口该怎么设计"。


