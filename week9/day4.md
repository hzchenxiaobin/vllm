# Day 4：模型与 Kernel —— MLA、MoE、TileLang ops

> **Week 9 · DeepInfer 源码精读 · Day 4**
> 前置知识：Day 3（metadata 四套、KV 布局、CUDA graph）；vLLM V1 的 attention backends 与采样器（Week 2/3）、投机解码（Week 4）。
> 源码基线：commit `9bd9cf2`（2026-10-21）。今日主角：`models/deepseek.py` + `decoder_layer.py` + `v4_modules/`（23 文件 5197 行）、`layers/attention_backend.py`、`tilelang_ops/`（33 个 wrapper + cuda/asc 各 ~30 个 kernel）、`csrc/`。
> 今日用时：5~6 小时（本周最重的一天：源码走读 3.5h + kernel 地图 1h + 自测 0.5h）

---

## 今日学习目标

| # | 目标 | 验收标准 |
|---|---|---|
| 1 | 讲清 V4.1 层结构 | 白板画出 `DeepseekDecoderLayer.forward` 的数据流（HC 残差三元组、yoco 边界、attn→mlp 两段），说出 `V4Attention.forward` 的 11 步 |
| 2 | 五个 vendor 库接入点地图 | 每个库"管哪一段 + import 在哪个文件 + 关键 API 名"脱口而出 |
| 3 | 精读 store_kv_cache | 解释 tile 划分（`blk = min(max(4096/input_dim,1),16)`）、grid 映射、FP8/FP4 双量化路径、`T.pdl_sync` 的含义 |
| 4 | 讲清 fused_sample 的融合边界 | 相对 vLLM sampler 少了哪些 kernel launch / 显存往返；reject sampling 与 accept_probs 如何一并产出 |
| 5 | 指认 batch-invariant kernel | `random_sampling` / `sum_smaller_probs` 的确定性设计（SplitMix64、列序平局规则） |
| 6 | 产出 kernel 地图 | 一个 decode step 里每个 kernel 的名字、来源（vendor / tilelang / csrc）、输入输出 |

今日阅读清单（对应 week9 README）：

| 文件 | 行数 | 主题 |
|---|---|---|
| `models/deepseek.py`、`decoder_layer.py` | 525+239 | 模型组装与层结构 |
| `v4_modules/`：`attn.py`（1008）、`mla_attn.py`（299）、`mlp.py`（616）、`moe.py`（112）、`yoco.py`（123）等 | 5197 | V4.1 模块全集 |
| `layers/attention_backend.py` | 146 | FlashMLA 接入 |
| `tilelang_ops/`：`flash_mla_utils.py`(107)、`store_kv_cache.py`(380)、`fused_sample.py`(832)、`schedule_mtp_varlen.py`、`swa_index.py`(117) + `cuda/store_kv_cache.py`(195) | — | 五个精读 kernel |
| `layers/linear.py`(185)、`sampler.py`(683)、`sampler_utils.py`(196) | — | 线性层与采样器 |
| `csrc/`（选读） | ~4000 | C++ 侧职责 |

---

## 一、模型结构总览：从 HF 容器到运行时模块

### 1.1 组装链

`models/deepseek.py` 的组装链（与 vLLM 的 registry + `ModelRunner.load_model` 同构，但**只有一个模型**）：

```text
get_model(model_config) ──► build_model_on_meta（meta device 上搭 HF 容器）
                        ──► DeepseekForCausalLM(HFModelParameters, cfg)
                              ├─ DeepseekModel(source_layers, head_module, cfg, load_nextn, shared_topology)
                              │    ├─ embed_tokens: VocabParallelEmbedding（VL 的 MultimodalEmbedding 抽出 tok_emb）
                              │    ├─ layers: N × DeepseekDecoderLayer（Day 3 已见其调用点）
                              │    ├─ hc_head / output_norm（HC 头，见 1.2）
                              │    └─ engram_runtime（ngram 嵌入，见 1.2）
                              ├─ MTP_module: DFlash（load_nextn>0 且有 dflash_cfg 时）——DSpark draft 模型
                              └─ sampler: Sampler(vocab_size)
                        ──► load_hf_checkpoint（权重重打包，见第三节 FP4 路径）
```

两个结构性事实先记下：

- **MTP 是"模块"不是"外挂"**：`DFlash` 挂在主模型上（`self.MTP_module`），其 MoE 层与主模型可能不同构（`DFlashDecoderLayer` 里 `layer_cfg = cfg + {"moe": layer.cfg.moe}`，注释"DFlash MoE 可能跟 main model 不同构"）；prefill 时主模型还会顺路为它收集目标层特征（`dflash_input`，Day 3 已见）。
- **SharedLayerTopology 决定谁真正持有参数**：`mla_attn.py` 文件头写明"a layer with `layer_map[l] == l` is a source layer, and the remaining layers reuse the modules of the source layer"——CED 的 share-kv 在模块层面就是"复用 nn.Module"，不只是 cache 复用。

### 1.2 `DeepseekDecoderLayer.forward`：一层的解剖（`decoder_layer.py:97-175`）

```python
def forward(self, hidden_states, kv_cache, input_metadata, residual, embed_hook, cross_layer_runtime, dflash_input):
    # ① yoco compact 边界：第一个 consumer 层。full 输入走 full-source 路径（为 CED 投影出完整 KV），
    #    compact 输入成为本层及后续的主路径
    if cross_layer_runtime.consumer_full_metadata is not None and cross_layer_runtime.consumer_full_hidden is None:
        compact_inputs, full_inputs = compact_yoco_boundary(hidden_states, residual, input_metadata, cross_layer_runtime, ...)
        cross_layer_runtime.consumer_full_hidden = self.attn_norm_forward(*full_inputs)[1]
        hidden_states, residual = compact_inputs
    # ② HC 归一化：一次产出 hidden_states(fp8) / hidden_states_bf16 / residual 三元组
    hidden_states, hidden_states_bf16, residual = self.attn_norm_forward(hidden_states, residual, hc_post_intervention)
    # ③ DFlash 目标层特征收集（写进 dflash_input 的对应行段，decode 时随主图捕获）
    if self.dflash_input_index is not None:
        self._store_dflash_input(residual.residual, dflash_input)
    # ④ 第一个 consumer 层捕获 shared_hidden（后续 consumer 层的 KV 投影源）
    if self.yoco_captures_shared_hidden and cross_layer_runtime.shared_hidden is None:
        cross_layer_runtime.shared_hidden = hidden_states_bf16.clone()
    # ⑤ attention（含 norm→proj→store→index→core→out proj 全链，见 1.3）
    hidden_states = self.self_attn(hidden_states=..., kv_cache=kv_cache, input_metadata=..., hidden_states_bf16=..., cross_layer_runtime=...)
    # ⑥ FFN 归一化 + MoE
    hidden_states, hidden_states_bf16, residual = self.ffn_norm_forward(hidden_states, residual)
    hidden_states = self.mlp(hidden_states, hidden_states_bf16, input_metadata)
    return hidden_states, residual
```

四个 V4.1 特色概念（vLLM 里都没有对应物）：

| 概念 | 一句话 | 代码位置 |
|---|---|---|
| **HC 残差流** | 层间传递的不是一个 residual，而是 `(fp8 hidden, bf16 hidden, HCLazyPostData residual)` 三元组——fp8 给 GEMM、bf16 给 gate/engram/KV 投影、lazy residual 延迟 HC 后处理 | `v4_modules/hc.py`（`HCLayerMixin.attn_norm_forward`）、`hc_lazy.py` |
| **CrossLayerRuntime** | 跨层持有的运行时值：`shared_hidden`（YoCo consumer 的 KV 投影源）、`pre_rope_kv`、`topk`、`consumer_full_*`——"逐层传同一引用，由各层原地写入" | `cross_layer.py` |
| **Engram** | ngram 嵌入模块：`tile_kernels.engram.engram_hash` 把 ngram token 哈希，`EngramRuntime.fetch(hashes)` 取嵌入，作为 attn-norm 之后的"主干干预钩子"；通信用 `deep_ep.EngramBuffer` | `engram.py`、`engram_metadata.py` |
| **DFlash 特征收集** | 指定层的 attention 输入被抄送到 `dflash_input` 缓冲（`dflash_input_store`，TileLang），主循环结束后 `dflash.update_main_cache` 写入 draft 的 main KV | `decoder_layer.py:92-95`、`deepseek.py:220-223` |

计时也是"层感知"的：`attn-layer-{i}` 只在第二层与倒数第二层记录（`get_enable_timer_layer` 按同构层归组），`scale=len(same_layers)` 把单层耗时放大成组耗时——profile 输出里的一条 `attn-part-N` 其实代表一组同构层。

### 1.3 `V4Attention.forward`：attention 的 11 步（`v4_modules/attn.py:752-880`）

Day 3 的调用链图里 ④ 框的展开。decode/prefill 共用主干：

```python
def forward(self, hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime):
    input_metadata.compute_attn_info(...)                  # ① 紧凑字段→attention 索引（Day 3）
    projected_kv, index_weights = self._project_current_hidden_inputs(hidden_states_bf16)
    if yoco_full_kv_stored:                                # ② yoco compact：用 full hidden 写完整 KV
        self._store_yoco_full_kv(...)                      #    （query 已裁剪，KV 不能裁——CED 要投影）
    elif self.shared_hidden_kv_proj is not None:           # ③ consumer 层：KV 从 shared_hidden 投影
        projected_kv = self.shared_hidden_kv_proj(shared_hidden_bf16)
    with userbuffer.cp_session(enabled=use_cp):
        wait_projected_kv = _make_cp_gather_getter(...)    # ④ CP：先发射 all_gather，与 ⑤ 重叠
        q_a, kv_swa = self._project_q_and_swa(...)         # ⑤ Q 降维投影 + SWA KV 投影
        kv_swa, wait_kv_swa = self._prepare_swa_kv(...)    #    （等 ① 的索引就绪）
        q_b, index_q_b = self._project_q_b(q_a, positions) # ⑥ Q 升维 + RoPE（fused 路径与 ⑩ 合并）
        kv_ca_index = self._compute_kv_ca_index(...)       # ⑦ 压缩：新 token → 压缩 KV，写 ca_cache；
                                                           #    indexer：q·k logits → topk → 物理槽位
    if is_prompt:                                          # ⑧ prefill：组装 gathered KV
        kv = [swa_cache 历史 | 本 chunk 新 SWA | 压缩 cache 历史]
        load_paged_kv_cache(ca_cache, load_segments_ca, ...)   # BlockLoadPlan 装载（Day 3）
        load_paged_kv_cache(swa_cache, load_segments_swa, ...)
        store_kv_cache(kv_swa, swa_cache, slot_mapping_store_swa)  # ★ 先 load 后 store（槽位可能重叠）
    context = self._prefill_attention(...) or self._decode_attention(...)   # ⑨/⑩ FlashMLA
    return self._wv_and_o_proj(context, positions)         # ⑪ Wv einsum（分组吸收）+ 输出投影
```

五个值得咀嚼的细节：

1. **②③ 是 YoCo/CED 的 attention 侧形态**：compact 模式下 query 被裁掉前缀，但 consumer 层的 block KV 还得完整产出（"the KV still has to cover the full context"）——用 full 版 shared hidden 投影写入；后续 consumer 层直接复用（shared_hidden_kv_proj）；
2. **④ 是"通信先行"模式**：CP 的 all_gather 先发射（返回 getter），Q 投影在其飞行期间执行，用时才 `wait`——CPU launch 顺序上的软件流水；
3. **⑦ 是稀疏注意力的心脏**：`index_attn.py` 里 indexer 的 q·k logits 走 **DeepGEMM 的 paged-mqa-logits**（`mlp_relu_schedule_metadata` 就是它的调度元数据，Day 3 见过），top-k 选择走 **DeepSelect**，逻辑→物理槽位换算走 **TileLang `convert_kv_index`**——一个 topk 串起三家库；
4. **⑧ 的顺序约束**（代码注释原文）："The SWA store slots may overlap the ranges loaded above, so prefill must store only after the cache load"——prefill 的 gather 语义要求先读旧值再写新值；
5. **⑪ Wv 是 einsum**：`Wv.weight.einsum_info = (n_wv_group, o_head_dim, wv_in_features)`——吸收了 `o_groups` 分组，MLA 的"吸收式"输出投影。

### 1.4 `v4_modules/` 文件地图

| 组 | 文件 | 职责 |
|---|---|---|
| attention | `attn.py`（1008） | V4Attention / DFlashAttention 运行时（本节主角） |
| | `mla_attn.py` | 参数容器：Wq/Wkv（A→norm→B LoRA 结构）、PackedWkv（A+Gate）、MQAIndexer |
| | `index_attn.py`（575） | indexer 前向（DeepGEMM logits + DeepSelect topk） |
| | `compress.py` | Compressor（C1/C2 压缩窗口边界） |
| | `rope.py` / `norm.py` | RoPE 构造 / RMSNorm 转换 |
| MoE | `mlp.py`（616） | DeepseekMoE 运行时（第三节） |
| | `moe.py` | HF 容器：FusedExperts / TopKGate / ReplicatedSharedExpert |
| 结构 | `cross_layer.py`（383） | CrossLayerRuntime + SharedLayerTopology（share-kv 映射） |
| | `hc.py`（397）/ `hc_blocks.py` / `hc_lazy.py` / `mhc_ops.py` / `mhc_modeling.py` | HC 头与残差流（`tf32_hc_prenorm_gemm`） |
| | `heads.py` / `containers.py` / `builder.py` | 预测头 / HF 容器 / 构建器 |
| 其他 | `engram*.py`（399） | ngram 嵌入 |
| | `yoco.py`（123） | YoCo 谓词与边界操作 |

---

## 二、MLA 参数容器与 FlashMLA 接入

### 2.1 参数容器（`mla_attn.py`）

`LoraRopeCompressFlashMHAAutoRecomp`（"MLA + 压缩 KV + 稀疏 indexer"的参数全集）的关键投影：

| 成员 | 结构 | 说明 |
|---|---|---|
| `Wq` | `ColumnParallelLora`: `A[hidden→q_lora] → norm → B[q_lora→n_head·q_head_dim]` | Q 的低秩两段投影（LoRA 式），FP8 GEMM |
| `q_norm` | RMSNorm（use_qnorm 时） | head 维归一化 |
| `Wkv` | `ColumnParallelLora(out=0)`: `A[hidden→kv_lora+rope]` → norm | KV 侧只有 A 段——latent 直接进 cache |
| `PackedWkv` | `SoftmaxAPEGatedColumnParallelLora`: `A` 产值 + `Gate` 产 logits | **压缩 KV 投影**；C1（ratio=1）单 token softmax 恒为 1 所以只要 A，C2 要 A+Gate（bf16，不参与 FP8） |
| `Wv` | `ReplicatedLinear`（带 `einsum_info`） | 分组吸收式输出投影 |
| `index_attn` | `MQAIndexer` | q_B / weights_proj / k_proj + RoPE（compress_rope_theta），FP4 KV |
| `out_proj` | `ReplicatedLinear` | — |
| `attn_sink_params` | `[n_head]` fp32 参数 | attention sink（FlashMLA 直接吃） |

注意 assert 的密度：`num_key_value_heads == 1`（MQA）、`fp8_attn`（**V4.1 强制 FP8 attention**）、`sparse_stage == 2`、`rope_dim == 64`——单模型框架把 config 当不变量检查。

### 2.2 FlashMLA 的三个入口（`layers/attention_backend.py`，146 行小而精）

```python
from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, fused_norm_rope_attn_rope_cast

def decode_mla_mqa_sparse(query, key_cache, head_dim_v, indices_in_kvcache, sched_meta, softmax_scale,
                          attn_sink=None, batch_invariant=False, index_length=None,
                          key_cache_extra=None, ..., fused_norm_rope_cast_kwargs=None):
    output, lse = flash_mla_with_kvcache(
        q=query, k_cache=key_cache,          # [num_blocks, block_size, 1, bytes_per_token] uint8！
        block_table=None, cache_seqlens=None,     # "FlashMLA 保留形参但当前忽略"——索引由外部算好
        head_dim_v=head_dim_v, tile_scheduler_metadata=sched_meta,
        causal=False,                        # sparse attention 只支持非 causal（因果性在索引里）
        is_fp8_kvcache=True,                 # V4.1 的 KV cache 为 paged FP8
        indices=indices_in_kvcache.unsqueeze(1),  # ← TileLang convert_kv_index 的产物
        attn_sink=attn_sink, extra_k_cache=..., topk_length=index_length,
        enable_batch_invariant=batch_invariant)   # ← batch-invariant 直通 FlashMLA
```

三个入口的分工：

| 入口 | 阶段 | 做什么 |
|---|---|---|
| `prefill_attn_varlen_sparse` | prefill | varlen Q × gathered KV 的稀疏 attention（`flash_mla_sparse_fwd`），index 指向 gathered buffer |
| `decode_mla_mqa_sparse` | decode | Q × **paged FP8 KV cache** 直接算（`flash_mla_with_kvcache`），index 是物理槽位 |
| `fused_norm_rope_attn_rope_cast` | 融合路径 | **norm + rope + attention + rope + cast 一个 kernel**：query 传"未 norm 未 rope 的 B 投影输出"，输出直接是 per-32 量化的 (fp8, sf) 元组——喂给下一个 FP8 GEMM，省一次往返 |

三个工程细节：

1. **head padding**：FlashMLA 只接受 64/128 头，`_pad_attention_heads` 把不足的头数补零（输出再切回来）——vendor kernel 的约束在适配层消化；
2. **调度元数据共享**：`FlashMLASchedMeta` 按 `f"{ratio}_{has_packed_kv}_{use_swa}_{topk_tokens}"` 键存在 `flash_mla_sched_meta_dict`（defaultdict，首个消费者层创建）——同配置层共享一份 tile scheduler 元数据；
3. **`enable_batch_invariant`**：README 关键问题 2 的一半答案在这里——batch-invariant 不只是 TileLang kernel 的事，**vendor 的 FlashMLA 也开了对应开关**。

---

## 三、MoE：EPLB 感知路由 + Mega MoE（`v4_modules/mlp.py`）

### 3.1 初始化：物理/逻辑 expert 的三层结构

```python
self.num_shared_experts   = moe_module.n_shared_experts
self.num_routed_experts   = moe_module.n_logical_routed_experts      # 逻辑 expert（gate 直接选的）
self.num_extra_experts    = moe_module.n_routed_experts - self.num_logical_routed_experts   # EPLB 冗余
self.num_physical_experts = self.num_routed_experts + self.num_extra_experts               # 物理 slot
self.num_local_physical_experts = self.num_physical_experts // self.ep_size                # 均分到 EP rank
assert self.num_physical_experts % self.ep_size == 0                  # 引擎负责向上取整到整除（Day 1）
```

**EPLB 的运行时形态**：gate 选逻辑 expert → `eplb_layer.logical_to_physical_map` 映射到物理 slot（冗余副本是物理层的别名）→ `eplb_layer.logical_count` 记录逻辑计数。`gate_forward` 把这两张表直接喂给 gate kernel：

```python
gate_kwargs = dict(num_topk=..., routed_scaling_factor=..., ep_rank=...,
    mask=input_metadata.valid_token_mask,                 # padding 行不路由
    bias=self.e_score_correction_bias,                     # noaux_tc 偏置
    image_bias=..., image_token_mask=images_seq_mask,     # VL：图片 token 有专属路由偏置！
    force_random=gate_force_random,                       # 负载模拟（Day 3 profile）
    to_physical_map=eplb_layer.logical_to_physical_map, logical_count=eplb_layer.logical_count,
    scoring_func="sqrtsoftplus")                          # assert：V4.1 专用打分函数
if self.use_mega_gate:   # GPU
    topk_idx, topk_weights = deep_gemm.bf16_mega_gate(hidden_states, self.gate.weight, **gate_kwargs)
else:
    topk_idx, topk_weights = moe_topk_gate(self.gate(hidden_states), **gate_kwargs)   # tile_kernels
```

### 3.2 三条 mma 路径与 Mega MoE 的启用条件

```python
self.mma_type = "fp8xfp4" | "fp8xfp8" | "bf16xbf16"       # act × weight
self.use_mega_moe = ((PLATFORM.is_gpu or mma_type == "fp8xfp4")
                     and mma_type in (...)
                     and PG.is_within_nvl_domain(PG.expert_parallel_group()))   # ★ NVL 域内
```

- **Mega MoE 是 NVL 域内的"通用解"**（Day 1 Q&A："regardless of EP size or batch size, significantly faster than unfused MoE"）——代码条件 `is_within_nvl_domain` 精确对应"scale-up 域内"（Day 2 见过：NPU 恒 True，GPU 用 `deep_ep.get_physical_domain_size` 判断）；
- **FP4 权重只在 mega 路径**（`PLATFORM.is_gpu or fp8xfp4`）：`process_moe_weight` 把 FP8-QAT checkpoint 的 routed expert 权重用 `tile_kernels.quant.per_token_cast` 转成 e2m1 + ue8m0 的 MXFP4 布局（`use_packed_ue8m0`、`use_tma_aligned_col_major_sf`——与 Q&A "per-32, ue8m0, MXFP4" 逐字对应）；`_convert_mega_moe_weight` 再把两个矩阵重排进 `deep_gemm.transform_weights_for_mega_moe` 的布局；
- **非 mega 路径** = 经典 dispatch → grouped GEMM ×2 → swiglu → combine：

```python
def forward(self, hidden_states, hidden_states_bf16, input_metadata):
    if self.use_mega_moe:
        return self.mega_moe_forward(...)      # gate 直写 sym_buffer；deep_gemm.fp8_fp4_mega_moe 一次算完
    topk_idx, topk_weights = self.gate_forward(...)
    event_overlap = self.dispatch(...)          # DeepEP EPBuffer.dispatch（all-to-all）
    moe_in_hidden, _, dispatched_topk_weights, handle = event_overlap.current_stream_wait()
    moe_out_hidden = self.routed_experts_forward(...)   # grouped GEMM + swiglu + grouped GEMM
    combine_pre_evt = self.ep_buffer.capture()
    with limit_num_sms(envs.DEEPINFER_USERBUFFER_COMM_SMS, enabled=PLATFORM.is_gpu):
        shared_out = self.shared_forward(hidden_states)  # 共享 expert 与 combine 重叠
    event_overlap = self.combine(moe_out_hidden, shared_out, handle, combine_pre_evt)  # bias=shared_out！
    return event_overlap.current_stream_wait()
```

值得抄进笔记的三个细节：

1. **`cumulative_local_expert_recv_stats=self._eplb_layer().expert_count`**——dispatch 顺手把每 expert 接收计数累计进 EPLB 统计（Day 1 的 `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` 窗口就是给它攒数据）；
2. **共享 expert 与 combine 重叠**：`combine(bias=shared_hidden_states)` 把共享 expert 输出作为 bias 加进 all-to-all combine——通信与共享 expert 计算天然重叠，还省一次加法 kernel；
3. **`do_cpu_sync` 的权衡注释**（dispatch 参数）："`do_sync=True` makes yoco compaction a cpu bottleneck, while `=False` costs extra GPU memory. Mask mode saves no compute and the CPU is not the bottleneck, so it is disabled only in compaction mode"——连一个 bool 都有 yoco 模式感知。

### 3.3 grouped GEMM 与激活

```python
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, psum_num_recv_tokens_per_expert,
    recipe_a=(1, 32), recipe_b=(1, 32 if fp4_b else ...), use_psum_layout=True, expected_m_for_psum_layout=...)
tile_kernels.swiglu_forward_and_per_token_cast(gate_up_out, fmt="fp8", num_per_channels=32, 
    psum_tokens_per_expert=..., topk_weights=..., clamp_value=...)
```

- `use_psum_layout` / `expected_m_for_psum_layout`：GEMM 输出直接按 expert 分段布局（psum），swiglu 再按同布局原地处理并**顺手乘上 topk_weights**（combine 的加权融进激活）；
- `rstd_1ulp.py`（asc 目录独有）暗示 NPU 上的 norm 实现做到 1-ULP 舍入一致——双平台数值对齐的又一处痕迹。

---

## 四、五个 vendor 库接入点地图（README 关键问题 1）

| 库 | 管哪一段 | import 位置（接入点） | 关键 API |
|---|---|---|---|
| **FlashMLA** | MLA attention（prefill varlen 稀疏 / decode paged FP8 MQA / 融合 norm-rope-attn-rope-cast） | `layers/attention_backend.py:2-3`、`v4_modules/attn.py:14` | `flash_mla_with_kvcache`、`flash_mla_sparse_fwd`、`fused_norm_rope_attn_rope_cast`、`FlashMLASchedMeta` |
| **DeepGEMM** | 全部稠密/分组 GEMM（FP8/FP4）、Mega MoE、gate、SF 布局变换 | `layers/linear.py:3`、`v4_modules/{mlp,attn,hc,mhc_ops,index_attn}.py`、`worker/runtime_config.py` | `bf16_gemm_nt`、`m_grouped_fp8_fp4_gemm_nt_contiguous`、`fp8_fp4_mega_moe`、`bf16_mega_gate`、`transform_sf_into_required_layout`、`transform_weights_for_mega_moe`、`SymmBuffer`、`tf32_hc_prenorm_gemm` |
| **DeepEP** | MoE all-to-all dispatch/combine、engram 通信用 buffer | `v4_modules/mlp.py:17`、`engram.py:14`、`parallel_utils/parallel_state.py`（`get_physical_domain_size` 判 NVL 域） | `EPBuffer.dispatch/combine/capture`、`EPHandle`、`EventOverlap`、`EngramBuffer` |
| **TileKernels** | MoE gate（非 mega）、swiglu 激活、量化转换、engram 哈希、PDL/确定性开关 | `v4_modules/{mlp,engram,engram_metadata}.py`、`models/deepseek.py:7`、`worker/runtime_config.py` | `moe_topk_gate`、`swiglu_forward_and_per_token_cast`、`quant.per_token_cast`、`engram.engram_hash`、`set_pdl` / `use_deterministic_algorithms` |
| **DeepSelect** | top-k 选择：indexer 的 token 挑选 + 采样的 top-logprobs | `v4_modules/index_attn.py:12`、`layers/sampler.py:12`、`sampler_utils.py:4`、`npu_ops/deep_select.py`（NPU 补丁） | `topk`（接口就一个函数） |

外加**自研 TileLang kernel**（`tilelang_ops/`，33 个 wrapper + `cuda/` 30 个实现 + `asc/` 30 个实现）与 **csrc**（C++ 引擎）。分发机制值得一看（`ops.py` 全文 31 行）：

```python
_pkg = importlib.import_module(f"{__package__}.{PLATFORM.tilelang_ops_backend}")   # cuda 或 asc
for _info in pkgutil.iter_modules(_pkg.__path__):
    _mod = importlib.import_module(...)
    for _name, _obj in vars(_mod).items():
        if _name.endswith("_tl"): globals()[_name] = _obj     # 动态 re-export 全部 *_tl 工厂
```

**Day 1 的 `tilelang_ops_backend` 字段在这里兑现**：平台选择即目录选择，两套 kernel 同名同签名。`npu_ops/__init__.py` 里还有一张 monkeypatch 表（`"deepinfer.npu_ops.deep_select": {"deep_select.interface": "topk"}`）——NPU 侧对 vendor 库的定点替换。

---

## 五、TileLang kernel 精读

### 5.1 文件组织约定：wrapper + kernel + ref + test 四位一体

`tilelang_ops/` 根目录的每个 `xxx.py` 都是同一结构：**公开 wrapper**（形状断言 + 分发 `ops.xxx_tl`）+ **`xxx_ref` PyTorch 参考实现** + **pytest 参数化测试**（对照 ref）+ **benchmark 用例**（`@pytest.mark.benchmark`，记录 `bandwidth_gbs`）。`cuda/xxx.py` / `asc/xxx.py` 才是真正的 `@tilelang.jit` kernel。这个约定意味着：**每个 kernel 天然带单元测试与带宽基准**，读 kernel 时先读同文件测试反推语义是最高效路径。

### 5.2 `store_kv_cache` 逐行解读（README 动手任务）

**问题定义**：把 `[n, num_elements]` 的 bf16 KV 写进 paged cache，途中完成 FP8/FP4 量化，`slot_mapping[i] = -1` 的行跳过。这是一切 attention 的 KV 写入终点（Day 3 调用链 ⑧ 的 `store_kv_cache(kv_swa, kv_cache.swa_cache, slot_mapping_store_swa)`）。

**Tile 机器学习**（`cuda/store_kv_cache.py:49-67`）——`@tilelang.jit` 装饰的工厂按 layout 特化：

```python
input_dim = layout.num_elements                 # 每 token 的 bf16 元素数（如 576：512 latent + 64 rope）
if is_quantized:
    value_dim = quantized_elements               # FP8: 576B；FP4: 288B（两值一字节）
    sf_dim = layout.sf_storage_elements          # scale 字节数（FP4 per-16: 36B → 对齐存储）
    group_size = layout.elements_per_sf          # 16（FP4）或 32（FP8 ue8m0）
    groups = sf_dim
    blk, threads = min(max(4096 // input_dim, 1), 16), 64      # ★ 每 tile 的 token 数
else:
    blk, threads = max(4096 // input_dim, 1), 256               # bf16 路径线程更多
```

**blk 的推导逻辑**：一个 tile 处理 `blk` 个 token、每 token `input_dim` 个 bf16 元素，总元素数 ≈ 4096（`4096 // input_dim` 向下取整，夹在 [1, 16]）——**按寄存器/共享内存预算反推 tile 尺寸**，让每 block 的工作量恒定，不随 layout 变化。FP8 路径 threads=64（量化在寄存器里做，元素多但每线程活轻），bf16 路径 256（纯搬运，宽并行）。

**Grid 映射**（`cuda/store_kv_cache.py:99-106`）：

```python
with T.Kernel(T.ceildiv(n, blk), threads=threads) as bx:      # grid = ⌈n/blk⌉ 个 block
    T.pdl_sync()                                              # ★ PDL：等前序 kernel（slot_mapping 可能由依赖启动产出）
    if is_quantized:
        slot_idx = T.alloc_fragment((blk,), "int32")
        T.copy(slot_mapping[bx * blk : (bx + 1) * blk], slot_idx)   # 每线程一个 slot，驻留寄存器
        for i in T.Parallel(blk):
            if bx * blk + i >= n: slot_idx[i] = -1             # 越界行变 padding
```

一维 grid、**block `bx` 负责 token 区间 `[bx·blk, (bx+1)·blk)`**——token 与 slot 的对应关系在读入时就固定（`slot // block_size` 定页、`slot % block_size` 定页内偏移），没有任何跨 block 通信。

**FP8 量化路径**（`i`=tile 内 token 行，`j`=scale 组，`k`=组内元素）：

```python
for i, j, k in T.Parallel(blk, groups, group_size):        # ① 读入到 fragment（越界/无效组补零）
    channel[i, j, k] = kv[...] if 有效 else 0
T.reduce_absmax(channel, channel_amax, dim=-1)             # ② 组内绝对值最大
scale = max(amax, FP8_AMAX_MARGIN) / FLOAT8E4M3_MAX        # ③ amax → scale（留 margin 防 448 溢出）
exp_scale = ue8m0_exp_scale(scale)                         # ④ scale 取 2 的幂（ue8m0！）
scale_local = reinterpret((exp_scale+127) << 23)           #    乘子（fp32 位模式直接拼）
sf_ue8m0    = cast(-exp_scale + 127, uint8)                #    存储的除子
for i, j, k:  cast_result = cast(channel * scale_local, fp8)  # ⑤ 量化
for i, j, k:  if slot_idx[i] != -1: kv_cache_value[slot//bs, slot%bs, j*gs+k] = cast_result   # ⑥ 散射写
for i, j:     if slot_idx[i] != -1: kv_cache_sf[slot//bs, slot%bs, j] = sf_local
```

要点：**ue8m0 的实现是纯位操作**（`(exp+127)<<23` 拼 fp32 位模式做乘子、`-exp+127` 存 uint8 除子）——ue8m0（2 的幂）让 scale 乘除完全避开浮点除法；FP4 路径多一个 `sf_format == "e4m3"` 分支（NVFP4 式非幂 scale，用真正的除法）；写回按 `slot % block_size` 散射进 paged cache，**无效 slot 在写回时才过滤**（①⑤ 仍然计算——padding 行参与计算但落空，换来无分支的整齐循环）。

**NPU 侧差异**（wrapper `store_kv_cache.py:61-64`）：asc 路径不拆 value/sf 两个张量（`kv_cache[:, :, 0, :]` 整块传）；`_slot_scatter` 是绕开 `IndexPutV2` 32 位偏移回绕的 workaround（"TODO(A5): 待 torch_npu 修复或有替代 kernel 后移除"）。

**测试矩阵的启示**：`_STORE_TEST_CASES` 的 pattern 名就是边界条件清单——`cross_page`（slot 跨页）、`broken_run`（快路径"只看首尾 slot 判断连续段"的克星：保端点乱内部）、`leading_invalid`（SIMD 分类器要拒绝 -1 开头的连续段）、`invalid_tiles`（同 core 上整 tile 无效后的流水线退出再进入）、`word_aligned`（非 8 对齐的 stride）。**读测试用例名 = 免费的 kernel 边界文档**。

### 5.3 `swa_index`：pivots → dense index 的展开

`expand_swa_index(pivots[num_tokens, 4], width)` 把每 token 两段 KV `[off0, len0, off1, len1]` 展开成 `[num_tokens, width]` 的稠密索引（空位 -1）。这就是 Day 3 `kv_swa_full_index_pivots → kv_swa_full_index` 的展开 kernel——prefill SWA 每个 token 的可见域最多两段（历史窗口 + 当前 KV），用 4 个 int32 存两段、用时展开成 FlashMLA 要的稠密行。ref 实现两个 `torch.where` 就说完语义；测试特意构造 int32 回绕（`2147483645 + width`）和负 offset。

### 5.4 `flash_mla_utils.convert_kv_index`：逻辑 → 物理槽位

```python
topk_idxs[b, s, k] 逻辑 token → logical_block = idx // block_size
                         → physical_block = block_tables[b, logical_block]     # gather
                         → physical_idx = physical_block * block_size + idx % block_size
                         → 位置 ≥ context_lens 的置 -1
```

indexer 选出的 top-k 是**逻辑 token 位置**，FlashMLA 的 `indices` 要**物理 cache 槽位**——这个 kernel 把 Day 3 的 `block_tables` 翻译成 attention 能直接消费的索引，原地改写 `topk_idxs`（省一次分配）。benchmark 注释还强调 `full_context=True` 以保证重复原地转换在 warmup 后仍合法。

### 5.5 `schedule_mtp_varlen`（回顾 Day 2）

`compute_seqlens_q_by_confidence_rates(total_q, remap, prev_confidence_rates, seqlens_q)`——confidence-scheduled verification 的 GPU 侧调度：累积接受率 + 全局排序选 top-k 前缀 + 融合 remap scatter（新请求 confidence 置 0）。NPU 断言 `num_seqs * nextn * 4 ≤ 32KB`（rates_ub 超 32KB UB 报错）——**Ascend Unified Buffer 的容量约束写进了 Python 层**。

### 5.6 `fused_sample`：一次 kernel 的整个采样阶段（README 关键问题 3）

签名即地图（`fused_sample.py:17-56` docstring）：

```python
fused_sample(logits[Q, vocab], sampling_inputs, *, tag, is_verify, logprobs_out,
             last_nextn_token_logprobs, last_next_n_token_ids, mtp_info_read_rows)
  → (logprobs[Q, vocab],
     candidate_token_ids[Q],          # 采出的 token
     accept_result_verify[Q],         # DSpark reject sampling 判定
     draft_selected_logprobs[Q],      # 被选 draft 的 logprob（draft id 为 -1 → -inf）
     accept_probs[Q])                 # max(1 − Σ residual, 0)：confidence head 的训练信号！
```

**融合的内容**（单 kernel `ops.fused_sample_tl(vocab_size, do_sample, do_reject_sampling)`）：temperature 缩放 → top-k 截断 → top-p 截断 → log-softmax（`lse_workspace[Q,2]` 两遍法）→ **确定性采样**（SplitMix64 哈希，`seq_seeds` + `sample_positions` + `seed_offset`）→ **reject sampling**（`max(target_prob − exp(draft_logprob − DRAFT_PROB_PENALTY), 0)` 的固定点残差）→ top-logprobs。`tag` 参数让 draft 与 target 用不同 RNG 命名空间（`_MTP_SEED_OFFSET = 1000000007`）。

**对照 vLLM V1 的 sampler**：vLLM 的采样是 kernel 序列——softmax（或 log_softmax）→ top-k/top-p mask（`topk_logits` 原地）→ `probs.div_(temperature)` → `torch.multinomial`（philox counter RNG）→ logprobs 单独再算一遍，投机验证另有 `verify` kernel。粗数 **5~8 次 kernel launch、2~3 次 `[Q, vocab]` 显存往返**；fused_sample 是 **1 次 launch、logits 只读一遍**（`logits ... treated as read-only`）。收益量化：decode step 末尾的 launch 开销（每 kernel ~2-5μs CPU + gap）在小 batch 下占 step 的可观比例；`[Q, 129280]` fp32 的往返（~2MB/行 × 2~3 次）在 64 并发下就是几百 MB/s 的无谓带宽。**代价**是 kernel 复杂度与平台双实现（asc 版独立维护）。

### 5.7 batch-invariant 双子星（README 关键问题 2）

| kernel | 确定性设计 |
|---|---|
| `random_sampling.py` | docstring 明示："**Deterministic in (seq_seeds, sample_positions, is_nextn_predict): independent of batch size and row order**"。RNG 不是全局 generator（philox 状态随 batch 漂移），而是 **SplitMix64 哈希 `(seed, position)` → uniform**——`_hash_to_uniform_ref` 用 int64 模拟 uint64（`urshift` 补负数偏移）逐位复现 kernel 的位运算，测试可验证到比特级 |
| `sum_smaller_probs.py` | "Inclusive CDF after sorting tokens by ascending probability. **equal-probability tokens are ordered by column index**"——浮点求和顺序固定（按概率升序、同概率按列序），消除并行归约的顺序不确定性。这是 `DEEPINFER_COMPUTE_SUM_SMALLER_PROBS` 开关背后的 kernel（确定性采样需要它算精确的 accept 概率） |

加上 Day 4 已见的 `enable_batch_invariant`（FlashMLA）、`use_deterministic_algorithms`（deep_gemm/tile_kernels）、asc 独有的 `rstd_1ulp`，batch-invariant 是**贯穿 vendor 库、TileLang kernel、引擎调度（ZOS 的 remap 校正也保持数值一致）的纵向特性**——Day 6c 再收口。

---

## 六、Sampler：采样的编排层（`layers/sampler.py`）

`sampler.py`（683 行）是 fused_sample 的编排者，`forward` 的骨架（`Sampler.forward:114-263`）：

```python
def forward(self, mtp_module, embedding, hidden_states, sampler_inputs, main_input_metadata, ...):
    # ① SO mask 先行准备（独立事件 so_mask_ready，与 logits 计算重叠！）
    if driver_inputs is not None:
        so_indices, so_bitmasks, so_status, so_mask_ready = self._get_so_mask(seq_metadata_list)
    # ② graph padding 修剪 + logits（hidden @ embedding.weight 的 GEMM + gather）
    hidden_states = gather(hidden_states, selected_token_indices)
    logits = get_logits(hidden_states, embedding, self.vocab_size)
    # ③ prompt 分支：prompt logprobs（compute_prompt_logprobs）+ first_n_token_no_eos + SO mask → 采样
    # ④ decode 分支：
    (sampled_tokens, verified_tokens, accepted_token_last_logprobs, last_valid_offset,
     num_generated_tokens, num_verified_tokens, batch_confidence_error, emitted_stop_token
    ) = self._verification(logits, ...)          # fused_sample 的 reject sampling + golden 生成
    golden_tokens = sampled_tokens.token_ids
    if speculative_inputs is not None:
        predict_tokens = mtp_module.decode_forward(...)   # ★ DSpark draft：dflash graph replay（Day 3）
        if emitted_stop_token is not None:
            predict_tokens.confidence_rates.masked_fill_(emitted_stop_token.unsqueeze(1), 0.0)
                                                           # 遇 eos 的请求下一轮 confidence 清零
    sampled_token_ids_gpu = sampled_tokens.token_ids.int() # ZOS 的 GPU 常驻副本（Day 2）
    ...
```

四个要点：

1. **follower 早退**（`driver_inputs is None` 分支）：CP slave 只参与 logits 前的集合通信，采样与 draft 全部只在 driver——与 Day 3 `do_sample=self.is_driver_worker` 对应；
2. **SO mask 与 logits 重叠**：mask 构建是独立 CUDA event，`wait_event` 在 logits 算完后才执行——结构化输出的开销被藏进 logits GEMM；
3. **`topk`（DeepSelect）在 `sampler_utils.py` 里做 top-logprobs**：`DeepInferTopkInfo` 的注释"先 logps 再 tokens，和 deep_select 保持一致"——vendor 的输出布局契约反向约束了引擎的数据结构；
4. **`emitted_stop_token` → confidence 清零**：请求命中 EOS 后，其 draft 的 acceptance 预测强制归零——**业务语义（停止生成）直接写进 DSpark 的调度信号**。

---

## 七、C++ 侧承担了什么（为什么不全用 Python）

`csrc/` 的体量分布本身就是答案（`wc -l` 排序）：

| 文件 | 行数 | 职责 | 为什么必须在 C++ |
|---|---|---|---|
| `engine/scheduler.hpp` | 1329 | 调度器（Day 2：DP 投票、waiting/running、抢占克隆） | 每 step 一次的热路径；`py::gil_scoped_release` 让它不阻塞其他 Python 线程 |
| `engine/sequence.hpp` | 419 | Sequence/SequenceData/SequenceMetadata/状态机 | 池化对象的生命周期管理（Day 2 的 RequestOutputPool 句柄） |
| `engine/flops_estimator.hpp` | 330 | 逐层逐 token FLOPs 估算（YoCo 模式感知） | 调度准入的每请求成本模型，Python 循环会慢一个量级 |
| `engine/outputs.hpp` | 308 | RequestOutput 构建（build_request_outputs） | 每 token 一次的输出组装 |
| `engine/zos.cpp` | 307 | prepare_zos_inputs / update_metadata_for_sampling_zos | ZOS 的 pinned CPU 抽取（Day 2/3），per-token 循环 |
| `engine/mtp_varlen.hpp` | 275 | MTP varlen 调度辅助 | confidence 排序的 CPU 侧配套 |
| `engine/prepare_sample.cpp` | 284 | prepare_sampler_inputs | 采样输入的 pinned 组装 |
| `engine/block_manager.hpp` | 243 | BlockManager（分配/克隆/zero/pin） | KV 块簿记 |
| `engine/input_metadata/` | 3 个 cpp | prepare_prompt_v4 / prepare_decode_v4 / BlockLoadPlan | metadata 构建（Day 3）——per-token/per-block 循环 |
| `graph_timer/` | 2 个 cpp | CUDA graph 事件节点计时 | 需要拿原始 cudaGraph_t 匹配 event 节点 |
| `cuda_utils.cpp` / `asc_utils.cpp` | — | cuMemAlloc 直分 / ACL 工具 | 绕过 torch allocator 的原生分配 |

**规律**：C++ 收编的是三类代码——**每 step 必跑的簿记循环**（调度、metadata、输出组装）、**需要打破 GIL 边界的并发结构**（对象池 + 线程）、**必须摸原生 API 的设施**（cudaGraph、cuMemAlloc、pinned memory）。而计算密集的部分（attention/GEMM/采样）在 vendor 库与 TileLang 里，模型结构在 Python（nn.Module 的组装与 launch）——**Python 只做"胶水层"**。这与 vLLM V1"调度器在 Python"形成鲜明对照，也是 ZOS 能把 schedule 塞进 forward 间隙的前提（Day 2 的结论在这里找到物理基础）。

---

## 八、kernel 地图（今日核心产出，SVG）

一个 **decode step**（ZOS + CUDA graph + MTP_VARLEN + Mega MoE + 稀疏 attention 层）里每个 kernel 的名字、来源、输入输出：

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 940" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect width="1100" height="940" fill="#fafbfc"/>
  <text x="550" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">Decode Step Kernel 地图（ZOS + graph + 稀疏 attention 层 + Mega MoE）</text>

  <!-- legend -->
  <rect x="60" y="46" width="980" height="40" rx="8" fill="#f8fafc" stroke="#94a3b8"/>
  <rect x="76" y="56" width="14" height="14" fill="#c4b5fd"/><text x="96" y="68" font-size="11.5" fill="#334155">vendor（FlashMLA/DeepGEMM/DeepEP/TileKernels/DeepSelect）</text>
  <rect x="430" y="56" width="14" height="14" fill="#86efac"/><text x="450" y="68" font-size="11.5" fill="#334155">tilelang_ops 自研（cuda/ 或 asc/）</text>
  <rect x="700" y="56" width="14" height="14" fill="#fca5a5"/><text x="720" y="68" font-size="11.5" fill="#334155">csrc（C++）</text>
  <rect x="860" y="56" width="14" height="14" fill="#93c5fd"/><text x="880" y="68" font-size="11.5" fill="#334155">torch 原生</text>

  <!-- Phase 1: prepare -->
  <rect x="60" y="100" width="980" height="106" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.8"/>
  <text x="80" y="124" font-size="13.5" font-weight="700" fill="#14532d">Phase 0 · ZOS 修正 + metadata 构建（每 step 一次，图内）</text>
  <rect x="80" y="138" width="300" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="230" y="154" text-anchor="middle" font-size="10.5" fill="#14532d">compute_seqlens_q_by_confidence_rates（Day 2）</text>
  <rect x="80" y="168" width="300" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="230" y="184" text-anchor="middle" font-size="10.5" fill="#14532d">compute_seqlens_k_zos ← prev num_accepted（GPU）</text>
  <rect x="400" y="138" width="300" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="550" y="154" text-anchor="middle" font-size="10.5" fill="#14532d">gather_input_ids_zos：prev token D2D scatter 拼 input_ids</text>
  <rect x="400" y="168" width="300" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="550" y="184" text-anchor="middle" font-size="10.5" fill="#14532d">prepare_v4_decode_metadata：一次写 12 类字段</text>
  <rect x="720" y="138" width="300" height="54" rx="5" fill="#fca5a5" stroke="#dc2626"/>
  <text x="870" y="158" text-anchor="middle" font-size="10.5" fill="#7f1d1d">prepare_decode_zos_inputs（csrc/zos.cpp）</text>
  <text x="870" y="176" text-anchor="middle" font-size="10.5" fill="#7f1d1d">pinned CPU 抽取 remap/seqlens/block_tables</text>

  <!-- Phase 2: per-layer -->
  <rect x="60" y="222" width="980" height="420" rx="10" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.8"/>
  <text x="80" y="246" font-size="13.5" font-weight="700" fill="#581c87">Phase 1 · 层循环 × N（紧凑索引在图内 compute_attn_info 展开）</text>

  <rect x="80" y="262" width="460" height="180" rx="8" fill="#fae8ff" stroke="#a855f7" stroke-width="1.5"/>
  <text x="310" y="284" text-anchor="middle" font-size="12.5" font-weight="700" fill="#581c87">Attention（V4Attention.forward 11 步）</text>
  <rect x="94" y="296" width="200" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="194" y="312" text-anchor="middle" font-size="10" fill="#4c1d95">Wq/Wkv/Wv/out_proj：deep_gemm（FP8）</text>
  <rect x="304" y="296" width="222" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="415" y="312" text-anchor="middle" font-size="10" fill="#4c1d95">fused_norm_rope_attn_rope_cast（FlashMLA 融合）</text>
  <rect x="94" y="328" width="200" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="194" y="344" text-anchor="middle" font-size="10" fill="#14532d">store_kv_cache：量化写 swa/ca cache</text>
  <rect x="304" y="328" width="222" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="415" y="344" text-anchor="middle" font-size="10" fill="#14532d">get_kv_{ca,swa}_index_decode：pivots→物理索引</text>
  <rect x="94" y="360" width="200" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="194" y="376" text-anchor="middle" font-size="10" fill="#4c1d95">paged-mqa-logits（DeepGEMM）× indexer q·k</text>
  <rect x="304" y="360" width="222" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="415" y="376" text-anchor="middle" font-size="10" fill="#4c1d95">topk（DeepSelect）× indexer token 挑选</text>
  <rect x="94" y="392" width="200" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="194" y="408" text-anchor="middle" font-size="10" fill="#14532d">convert_kv_index：逻辑→物理槽位</text>
  <rect x="304" y="392" width="222" height="26" rx="5" fill="#c4b5fd" stroke="#7c3aed" stroke-width="2"/>
  <text x="415" y="409" text-anchor="middle" font-size="10.5" font-weight="700" fill="#4c1d95">flash_mla_with_kvcache（核心 attention）</text>
  <text x="310" y="432" text-anchor="middle" font-size="10" fill="#7e22ce">输入：q_b + paged FP8 cache + 物理索引 + sched_meta + attn_sink</text>

  <rect x="560" y="262" width="460" height="180" rx="8" fill="#fae8ff" stroke="#a855f7" stroke-width="1.5"/>
  <text x="790" y="284" text-anchor="middle" font-size="12.5" font-weight="700" fill="#581c87">MoE（DeepseekMoE.forward）</text>
  <rect x="574" y="296" width="200" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="674" y="312" text-anchor="middle" font-size="10" fill="#4c1d95">bf16_mega_gate（DeepGEMM，GPU）</text>
  <rect x="784" y="296" width="222" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="895" y="312" text-anchor="middle" font-size="10" fill="#4c1d95">moe_topk_gate（TileKernels，NPU/非 mega）</text>
  <rect x="574" y="332" width="432" height="26" rx="5" fill="#c4b5fd" stroke="#7c3aed" stroke-width="2"/>
  <text x="790" y="349" text-anchor="middle" font-size="10.5" font-weight="700" fill="#4c1d95">fp8_fp4_mega_moe（NVL 域内：gate 后一次算完，含共享 expert）</text>
  <rect x="574" y="368" width="130" height="22" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="639" y="383" text-anchor="middle" font-size="9.5" fill="#4c1d95">EPBuffer.dispatch</text>
  <rect x="712" y="368" width="140" height="22" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="782" y="383" text-anchor="middle" font-size="9.5" fill="#4c1d95">grouped GEMM ×2（DeepGEMM）</text>
  <rect x="860" y="368" width="146" height="22" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="933" y="383" text-anchor="middle" font-size="9.5" fill="#4c1d95">EPBuffer.combine（bias=共享）</text>
  <rect x="574" y="398" width="200" height="22" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="674" y="413" text-anchor="middle" font-size="9.5" fill="#4c1d95">swiglu_forward_and_per_token_cast（含 topk 加权）</text>
  <text x="790" y="432" text-anchor="middle" font-size="10" fill="#7e22ce">非 mega 路径 = dispatch → GEMM → swiglu → GEMM → combine（共享 expert 与 combine 重叠）</text>

  <!-- HC + engram -->
  <rect x="80" y="452" width="940" height="60" rx="8" fill="#fae8ff" stroke="#a855f7" stroke-width="1"/>
  <rect x="94" y="466" width="280" height="34" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="234" y="480" text-anchor="middle" font-size="10" fill="#4c1d95">HC 残差流：tf32_hc_prenorm_gemm（DeepGEMM）</text>
  <text x="234" y="494" text-anchor="middle" font-size="9.5" fill="#7c3aed">+ mhc_pre_apply_mix / hc_post</text>
  <rect x="394" y="466" width="280" height="34" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="534" y="480" text-anchor="middle" font-size="10" fill="#4c1d95">Engram：tile_kernels.engram_hash</text>
  <text x="534" y="494" text-anchor="middle" font-size="9.5" fill="#7c3aed">+ EngramBuffer 通信（DeepEP）</text>
  <rect x="694" y="466" width="300" height="34" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="844" y="480" text-anchor="middle" font-size="10" fill="#14532d">dflash_input_store：目标层特征抄送（TileLang）</text>
  <text x="844" y="494" text-anchor="middle" font-size="9.5" fill="#166534">层循环后 dflash.update_main_cache 统一写 draft KV</text>

  <!-- Phase 3: sample -->
  <rect x="60" y="656" width="980" height="128" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/>
  <text x="80" y="680" font-size="13.5" font-weight="700" fill="#1e3a8a">Phase 2 · 采样 + DSpark draft（execute_sample，driver only）</text>
  <rect x="80" y="694" width="290" height="24" rx="5" fill="#93c5fd" stroke="#2563eb"/>
  <text x="225" y="710" text-anchor="middle" font-size="10" fill="#1e3a8a">gather（graph padding 修剪）+ get_logits GEMM</text>
  <rect x="384" y="694" width="290" height="24" rx="5" fill="#86efac" stroke="#16a34a" stroke-width="2"/>
  <text x="529" y="710" text-anchor="middle" font-size="10.5" font-weight="700" fill="#14532d">fused_sample：temp+topk+topp+logsoftmax+采样+reject</text>
  <rect x="688" y="694" width="160" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="768" y="710" text-anchor="middle" font-size="10" fill="#4c1d95">topk（DeepSelect）× logprobs</text>
  <rect x="860" y="694" width="170" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="945" y="710" text-anchor="middle" font-size="10" fill="#14532d">sum_smaller_probs（可选）</text>
  <rect x="80" y="726" width="500" height="24" rx="5" fill="#c4b5fd" stroke="#7c3aed"/>
  <text x="330" y="742" text-anchor="middle" font-size="10" fill="#4c1d95">DSpark draft：dflash graph replay（get_fused_dflash_get_input + draft attention + draft MoE）</text>
  <rect x="594" y="726" width="290" height="24" rx="5" fill="#86efac" stroke="#16a34a"/>
  <text x="739" y="742" text-anchor="middle" font-size="10" fill="#14532d">store_mtp_info：draft 结果写 info pool（TileLang）</text>
  <text x="80" y="768" font-size="10" fill="#1e40af">SO mask 的构建事件与 logits GEMM 重叠（sampler.py:137-164）；遇 EOS 请求的 confidence 清零</text>

  <!-- Phase 4: post -->
  <rect x="60" y="798" width="980" height="60" rx="10" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5"/>
  <text x="80" y="820" font-size="13.5" font-weight="700" fill="#7c2d12">Phase 3 · post-process(N−1)（与 Phase 0-2 重叠，Day 2）</text>
  <rect x="80" y="830" width="290" height="22" rx="5" fill="#fca5a5" stroke="#dc2626"/>
  <text x="225" y="845" text-anchor="middle" font-size="10" fill="#7f1d1d">process_MTPInfo / process_outputs（csrc）</text>
  <rect x="384" y="830" width="290" height="22" rx="5" fill="#fca5a5" stroke="#dc2626"/>
  <text x="529" y="845" text-anchor="middle" font-size="10" fill="#7f1d1d">build_request_outputs（csrc/outputs.hpp）</text>
  <rect x="688" y="830" width="340" height="22" rx="5" fill="#fca5a5" stroke="#dc2626"/>
  <text x="858" y="845" text-anchor="middle" font-size="10" fill="#7f1d1d">clone/zero_states（抢占）+ DeferredRelease + do_save_kvcache（虚）</text>

  <!-- summary -->
  <rect x="60" y="872" width="980" height="52" rx="8" fill="#f8fafc" stroke="#94a3b8"/>
  <text x="550" y="892" text-anchor="middle" font-size="11.5" fill="#334155">一个 decode step ≈ Phase 0（4-5 kernel）+ N×(attention 7-8 + MoE 3-6 + HC/engram 2-3) + Phase 2（4-5 + draft 图）</text>
  <text x="550" y="912" text-anchor="middle" font-size="11.5" fill="#334155">全部在 CUDA graph 内（decode 路径）；Phase 0 由 ZOS 写入图 buffer，Phase 3 在 CPU 与图执行重叠</text>
</svg>
```

---

## 九、动手实验

### 实验 1：注释版 store_kv_cache（本文 5.2 即交付）

按 README 要求"写一段注释版逐行解读"——5.2 节就是。自查三问：blk 为什么夹在 [1,16]？无效 slot 为什么参与量化但不写回？ue8m0 的乘子为什么用位拼接而不是除法？

### 实验 2：手算 store 的 tile 参数

```
SWA KV（MXFP8 路径）：input_dim = 576（512 latent + 64 rope... 布局按 V41_FORMAT）
  blk = min(max(4096 // 576, 1), 16) = min(7, 16) = 7；threads = 64
  grid = ⌈n / 7⌉；每 block 处理 7 token × (576B 量化数据 + 18B scale)
  → 算一算 n=384（64 请求 × 6 token）时的 grid 数与每线程字节数，判断是否带宽饱和
FP4 路径（ca_cache）：quantized = 288B、sf per-16 → 36 组
  groups=36, group_size=16；cast_result 是 T.float4_e2m1fn（两值一字节）
```

### 实验 3（有硬件）：benchmark 抓 kernel 名单

```bash
python examples/benchmark.py --help        # 摸清参数（README 动手任务）
PROFILE=1 python examples/benchmark.py --decode_num_requests 64
```

拿 trace 对照第八节地图逐 kernel 打勾。重点关注：`store_kv_cache` 的带宽（test benchmark 记录 `bandwidth_gbs`）、mega_moe vs dispatch/combine 是否如预期切换（NVL 域内应只见 mega）、`fused_sample` 是否只有一个采样 kernel。

---

## 十、面试高频问题

**Q1：五个 vendor 库各管哪一段？**

A：FlashMLA 管 attention 本体（prefill varlen 稀疏 / decode paged FP8 MQA / 融合 norm-rope-attn-rope-cast），接入点 `layers/attention_backend.py`；DeepGEMM 管所有 GEMM——稠密（bf16_gemm_nt）、分组（m_grouped_fp8_fp4_gemm_nt_contiguous）、Mega MoE（fp8_fp4_mega_moe）、mega gate、SF 布局变换、甚至 HC 头的 tf32 prenorm GEMM；DeepEP 管 MoE all-to-all（EPBuffer.dispatch/combine）和 engram 通信（EngramBuffer），还提供 NVL 域判定；TileKernels 管 MoE gate（非 mega 路径 moe_topk_gate）、swiglu 激活（顺带 topk 加权与 FP8 cast）、量化转换（per_token_cast 到 MXFP4）、engram 哈希，以及 PDL/确定性算法的全局开关；DeepSelect 只有一个 topk 接口，但被两处关键消费：indexer 的 token 挑选和采样的 top-logprobs。

**Q2：fused_sample 相对 vLLM 的 sampler 融合了什么？收益？**

A：vLLM V1 采样是 kernel 序列：softmax/log_softmax → top-k → top-p mask → temperature 缩放 → multinomial 采样 → logprobs 另算，粗数 5-8 次 launch、2-3 次 [Q, vocab] 显存往返。fused_sample 单 kernel 完成 temperature/top-k/top-p/log-softmax/确定性采样（SplitMix64 哈希 seq_seed+position）/DSpark reject sampling/accept_probs 输出——1 次 launch、logits 只读一遍。收益：小 batch 下 launch 开销占比高（每 kernel 2-5μs CPU + gap），省 5 次 launch 直接砍掉采样阶段的固定开销；大 batch 下省 [Q,129280] fp32 的显存往返（带宽）。额外红利：reject sampling 与 accept_probs（confidence head 的训练信号）在同一 kernel 里产出，DSpark 验证零额外成本。

**Q3：store_kv_cache 的 tile 划分怎么定？**

A：量化路径 `blk = min(max(4096 // input_dim, 1), 16)`、threads=64；bf16 路径 `blk = max(4096 // input_dim, 1)`、threads=256。逻辑是"每 tile 元素预算恒定"：blk × input_dim ≈ 4096 个 bf16 元素（8KB），保证每 block 的寄存器/共享内存占用不随 KV 布局变化；量化路径夹上限 16 是因为 fragment 里要同时放 channel/amax/scale/cast_result 四份 blk×groups 空间。grid 一维：block bx 负责 token 区间 [bx·blk, (bx+1)·blk)，每线程一个 token 的 slot 驻留寄存器，`slot//block_size` 定页、`slot%block_size` 定页内偏移，无跨 block 通信。PDL 同步（T.pdl_sync）在读 slot_mapping 前执行——它可能由依赖启动的前序 kernel 产出。

**Q4：batch-invariant 在 kernel 层怎么实现？**

A：两个关键设计。采样随机数不用全局 generator（philox 状态随 batch 组成漂移），而是 SplitMix64 哈希 (seq_seed, sample_position)——docstring 明确"Deterministic in (seq_seeds, sample_positions): independent of batch size and row order"，ref 实现用 int64 逐位模拟 uint64 做到比特级验证。浮点求和（sum_smaller_probs 的 CDF）固定顺序：按概率升序、同概率按列序——消除并行归约的顺序不确定性。再往上：FlashMLA 开 enable_batch_invariant、deep_gemm/tile_kernels 开 use_deterministic_algorithms、NPU 有 rstd_1ulp 做 1-ULP 一致的 norm。

**Q5：MoE 的 Mega 路径和经典路径怎么选？**

A：`use_mega_moe = (is_gpu or fp8xfp4) and PG.is_within_nvl_domain(EP group)`——**NVL（scale-up）域内一律 mega**（对应 Q&A"域内通用解，任意 EP/batch 都快于非融合"），域外或 NPU 走经典 dispatch→grouped GEMM→swiglu→combine。mega 路径 gate 直写 SymmBuffer、`fp8_fp4_mega_moe` 一次调用完成 dispatch/计算/combine（含融合的共享 expert）；经典路径的亮点是共享 expert 与 combine 重叠（combine 的 bias 参数）+ dispatch 顺手累计 EPLB expert 计数。FP4 权重（Q&A 的 per-32 ue8m0 MXFP4）只在 mega 路径启用，权重在加载期由 per_token_cast 从 FP8-QAT checkpoint 转换。

**Q6：为什么 DeepInfer 的模型代码里没有 vLLM 那样的 attention backend 抽象层？**

A：vLLM 的 `v1/attention/backends/` 多后端（FlashAttention/FlashInfer/TRTLLM）服务于多硬件多模型的通用性；DeepInfer 只有 FlashMLA 一个后端，`attention_backend.py` 只是 146 行的薄适配（head padding、入口三选一、融合路径开关），抽象层被压到最小——因为**抽象的代价（间接层、形状协商、配置面）在单后端场景下是纯开销**。这是"通用框架 vs 生产专用框架"在代码形状上的直接体现：vLLM 的扩展点在"换 kernel"，DeepInfer 的优化点在"熔 kernel"（fused_norm_rope_attn_rope_cast、fused_sample、mega_moe 全是这个哲学）。

---

## 十一、今日总结（三分钟电梯版）

> V4.1 的层结构是"MLA 压缩 KV + 稀疏 indexer + CED 层共享 + YoCo consumer + HC 残差流"的复合体：`DeepseekDecoderLayer` 用 CrossLayerRuntime 在层间搬运 shared_hidden/pre_rope_kv，第一个 YoCo consumer 层做 full/compact 双路径（query 裁剪、KV 完整）。attention 的 11 步里，Q/KV 投影是 DeepGEMM FP8，KV 写入是 TileLang `store_kv_cache`（blk=min(4096/dim,16) 的恒定预算 tile、ue8m0 位拼接量化、PDL 同步），indexer 串起 DeepGEMM paged-mqa-logits + DeepSelect topk + TileLang convert_kv_index，核心是 `flash_mla_with_kvcache`（直接吃 paged FP8 cache 和物理索引）。MoE 三条 mma 路径，NVL 域内 Mega MoE（gate+dispatch+GEMM+combine 一次调用、共享 expert 融合），域外经典 DeepEP dispatch/combine 且共享 expert 与通信重叠；EPLB 的 logical→physical 映射直接喂给 gate kernel。采样是 `fused_sample` 单 kernel（温度/top-k/top-p/log-softmax/SplitMix64 确定性采样/reject sampling/accept_probs），比 vLLM 的 5-8 次 launch 省 5 次且 logits 只读一遍。C++ 收编全部每 step 簿记（scheduler 1329 行 + metadata/输出/采样输入组装），Python 只剩模型胶水。tilelang_ops 33 个 wrapper 全部"kernel+ref+test+benchmark"四位一体，cuda/asc 双实现按平台目录切换。

---

## 十二、今日自测题

1. `DeepseekDecoderLayer.forward` 传递的三元组是哪三个？分别喂给谁？
2. yoco compact 模式下，第一个 consumer 层为什么要同时算 full 和 compact 两条路径？
3. `V4Attention.forward` 里 prefill 的 SWA 为什么必须"先 load 后 store"？
4. `decode_mla_mqa_sparse` 的 `indices` 参数是什么？它是哪个 TileLang kernel 的输出？
5. `PackedWkv` 的 Gate 分支在什么条件下不存在？为什么？
6. Mega MoE 的三个启用条件？`is_within_nvl_domain` 在 NPU 上返回什么？
7. `store_kv_cache` 量化路径的 `blk` 怎么算？无效 slot 的行是否参与量化？
8. `random_sampling` 的确定性靠什么保证？与 `torch.multinomial` 的本质区别？
9. `fused_sample` 的 `tag` 参数和 `_MTP_SEED_OFFSET` 解决什么问题？
10. csrc 里行数最大的三个文件是什么？它们为什么必须在 C++？

<details>
<summary>参考答案（先自测再展开）</summary>

1. `(fp8 hidden_states, bf16 hidden_states_bf16, HCLazyPostData residual)`——fp8 喂 GEMM（Wq/Wkv 等），bf16 喂 gate/engram/KV 投影与 shared_hidden 捕获，lazy residual 延迟 HC 后处理（`hc_lazy.py`）。
2. query 侧被 compact 裁剪（省 consumer 层计算），但 **KV 必须覆盖全上下文**——CED 要求 decoder 的 block KV 从最后一个 encoder 层投影而来（"the KV still has to cover the full context"）。full 输入走 `consumer_full_hidden` 路径产出完整 KV，compact 输入成为 consumer 主路径。
3. prefill 的 SWA store 槽位可能与 load 装载的历史区间重叠（`slot_mapping_store_swa` vs `load_segments_swa` 都指向 swa ring）——先 store 会覆盖还没读进 gathered buffer 的旧值。
4. **物理 cache 槽位**（`physical_block × block_size + offset`），是 `convert_kv_index`（`flash_mla_utils.py`）的输出：indexer 的 top-k 逻辑位置经 block_tables 翻译，超 context_len 的置 -1。FlashMLA 的 `block_table=None`——分页寻址已被前置消化。
5. `kv_compress_ratio == 1`（C1 层）。C1 的单 token softmax 恒为 1，压缩无需门控；C2 才需要 A（值）+ Gate（logits）做 token 合并（`SoftmaxAPEGatedColumnParallelLora`，bf16 不参与 FP8）。
6. `(PLATFORM.is_gpu or mma_type=="fp8xfp4") and mma_type ∈ 三种 and PG.is_within_nvl_domain(EP组)`。NPU 上 `is_within_nvl_domain` 恒返回 True（parallel_state.py:131-134）。
7. `blk = min(max(4096 // input_dim, 1), 16)`（input_dim 是每 token 的 bf16 元素数）；threads=64。参与——量化照常计算（channel/amax/scale/cast 全做），只在**写回**时过滤 `slot_idx[i] != -1`，换来无分支整齐循环（越界行读入时已补零）。
8. SplitMix64 哈希 `(seq_seed, sample_position, is_nextn_predict)` 生成 uniform——种子是每序列每位置的函数，与 batch 大小/行序无关。`torch.multinomial` 用全局 generator（philox offset 随调用次数与 batch 组成漂移），同一请求换个 batch 位置结果就变。
9. draft 与 target 采样用不同 RNG 命名空间：非 default tag 的 seed 加 `1000000007` 偏移，保证 target 验证采样与 draft 生成采样的随机流不重合（避免系统性相关）。
10. `scheduler.hpp`（1329，每 step 调度+GIL 释放）、`sequence.hpp`（419，池化对象生命周期）、`flops_estimator.hpp`（330，调度准入成本模型）——共同点是**每 step 必跑的簿记循环**，Python 循环慢一个量级且受 GIL 限制。

</details>

---

## 十三、今日产出物

1. **kernel 地图**（第八节 SVG，README 指定产出）：decode step 的 kernel × 来源 × 输入输出全景。
2. **store_kv_cache 注释版解读**（5.2 节，README 动手任务交付）。
3. **差异清单 v1.3**（在 Day 3 的 18 条之上补充 3 条）：

| # | 维度 | vLLM V1 | DeepInfer | 源码证据 |
|---|---|---|---|---|
| 19 | attention 后端 | 多后端抽象（FlashAttn/FlashInfer/TRTLLM） | FlashMLA 单后端 + 薄适配层，"熔 kernel"哲学 | attention_backend.py（146 行） |
| 20 | 采样 | kernel 序列（softmax→topk→topp→multinomial） | fused_sample 单 kernel + SplitMix64 确定性采样 | fused_sample.py:17-120 |
| 21 | MoE 路径 | fused_moe（按后端）+ TP/EP | Mega MoE（NVL 域内一次调用）/ DeepEP dispatch-combine + EPLB 映射进 gate kernel | v4_modules/mlp.py:184-188, 397-401 |

**明日预告（Day 5）**：分布式并行——Attention DP 与 balanced CP（`prepare_cp_mirror` 的 zigzag 切分、`mirror_cp_index` 的镜像布局、userbuffer 的 CP 通信）、MoE EP + EPLB 的重排机制（`eplb/policy.py`、`shuffle.py`、NVLink symm_mem 权重搬迁）、`deep_cluster` 拓扑感知。今日 kernel 地图里的 CP 分支（cp_session / all_gather）将被展开。


