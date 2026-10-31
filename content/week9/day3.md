# Day 3：Worker 层 —— model_runner、cache_engine 与 CUDA graph

> **Week 9 · DeepInfer 源码精读 · Day 3**
> 前置知识：Day 2（异步调度四段式管线、`_correct_seqlens_qk`、MetadataZOS）；vLLM V1 的 gpu_model_runner 与 CUDAGraphRunner（Week 3）。
> 源码基线：commit `9bd9cf2`（2026-10-21）。今日主角：`deepinfer/worker/` 六个文件 + `deepinfer/model_executor/input_metadata/` 四个文件 + `csrc/engine/input_metadata/`。
> 今日用时：4~5 小时（源码走读 3h + 布局图/对照表 1h + 自测 0.5h）

---

## 今日学习目标

| # | 目标 | 验收标准 |
|---|---|---|
| 1 | 讲清 `input_metadata` 为什么分多套 | 对照表：base/V4、yoco、dflash、zos 四套各自"谁产生、谁消费、对应模型的什么特性、何时缺席" |
| 2 | 画出 decode step 调用链 | 从 scheduler 输出到 kernel launch 的完整链路，标注每个环节的文件路径与 CPU/GPU 侧 |
| 3 | 讲清 KV cache 物理布局 | 手绘两池（`gpu_blocks` / `gpu_states`）+ StateBlock 分区，推出 `state_size_to_disk = 20 × 134 × 528 B`，并指出开源边界（哪些接口是给未开源的 KV 系统留的） |
| 4 | 讲清 CUDA graph 机制 | capture 集合怎么生成（以 total_q 为键！）、共享 memory pool、ZOS 下 `input_on_device` 如何做到 graph 内零 H2D、dflash 为什么单独一个 runner |
| 5 | 对照 vLLM | capture 集合思路、KV 池组织、draft graph 三处的异同 |

今日阅读清单（对应 week9 README）：

| 文件 | 行数 | 主题 |
|---|---|---|
| `worker/worker.py`、`base.py` | 283+357 | Day 2 已读，今日补 `runtime_config.py`（69 行） |
| `worker/model_runner.py` | 644 | batch 输入准备、profile、graph capture、decode perf |
| `model_executor/input_metadata/{input_metadata,yoco,dflash,prepare_tensor_input_zos}.py` | 490+182+84+382 | 四套 metadata |
| `worker/cache_engine.py` | 978 | KV 分配与布局（本日最大文件） |
| `worker/cuda_graph_runner.py` + `dflash_cuda_graph_runner.py` | 168+100 | 双 graph runner |
| `worker/decode_perf_cache.py` | 166 | decode 性能曲线缓存 |
| `csrc/engine/input_metadata/`（选读） | 3 个 cpp | C++ 侧 prepare 与 BlockLoadPlan |

---

## 一、Worker 层运行期配置：`runtime_config.py`

Day 2 读过 `WorkerBase` 的资源生命周期；今天补上它调用的 `configure_worker_environment / configure_worker_kernels`（`worker/runtime_config.py`，69 行，但每行都是生产踩坑记录）：

```python
def configure_worker_environment():
    os.environ["EP_AVOID_RECORD_STREAM"] = "1"          # DeepEP：不记录 stream 依赖
    os.environ.pop("NCCL_NVLS_ENABLE", None)
    if PLATFORM.is_gpu:
        os.environ["NCCL_NO_CACHE"] = "NCCL_WIN_STRIDE,NCCL_GIN_TYPE,NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC"
        os.environ["NCCL_PXN_DISABLE"] = "1"            # expert rearrangement 禁用 peer access
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        os.environ["NCCL_GIN_ERROR_QUERY_SEC"] = "432000"
    # JIT kernel 缓存三库镜像：tilelang / deep_gemm / deep_ep 各占一个子目录，
    # 镜像预热目录（test_warmup / history_warmup）附在后面，miss 时编译进第一个目录
    if jit_cache_root:
        os.environ["TILELANG_CACHE_DIR"] = f"{jit_cache_root}/tilelang"
        os.environ["DG_JIT_CACHE_DIR"]  = f"{jit_cache_root}/deep_gemm{warmup_suffix}"
        os.environ["EP_JIT_CACHE_DIR"]  = f"{jit_cache_root}/deep_ep{warmup_suffix}"

def configure_worker_kernels():
    if PLATFORM.is_gpu:
        deep_gemm.set_pdl(True); tile_kernels.set_pdl(True)      # PDL：programmatic dependent launch
    if envs.DEEPINFER_BATCH_INVARIANT or PLATFORM.is_npu:
        deep_gemm.use_deterministic_algorithms(True)              # batch-invariant 的 kernel 级开关（Day 6c）
        tile_kernels.use_deterministic_algorithms(True)
```

三个知识点：

1. **PDL（Programmatic Dependent Launch）**：让下一个 kernel 在上一个结束前就开始 launch，压缩 kernel 间空隙——这是 kernel 级的"异步调度"，与引擎级的 ZOS 呼应；
2. **确定性算法开关**：`DEEPINFER_BATCH_INVARIANT=1` 不只是引擎层概念，一直下沉到 deep_gemm / tile_kernels 的算法选择（NPU 上无条件开启，注释："防止 CI 数值不稳定"）；
3. **JIT 缓存镜像**：tilelang / deep_gemm / deep_ep 都是 JIT 编译库，生产镜像里预置预热缓存避免首步编译——启动性能工程。

另一个函数 `expose_all_gpus_to_cuda_driver()`（engram 的 GPU 存储需要进程看到全部卡，即使 `CUDA_VISIBLE_DEVICES` 只圈了本 rank 的卡）是 CP slave 启动的第一件事（`_spawn_worker`），细节 Day 5 展开。

---

## 二、model_runner：从 C++ 调度结果到 GPU 输入

### 2.1 总分派：`_prepare_input` 的四路分支

Day 2 的时序图里 `execute_main_model` 只是一行；今天把它展开。`model_runner.py:148-230` 的 `_prepare_input` 按 **driver/follower × prompt/decode × ZOS/eager × graph/eager** 分派：

```python
def _prepare_input(self, seq_metadata_list, metadata_zos, total_q):
    if self.is_driver_worker:
        return self._prepare_input_driver(seq_metadata_list, metadata_zos, total_q)
    return self._prepare_input_follower()          # CP slave：等 driver 广播
```

driver 侧（`_prepare_input_driver`）的骨架：

```python
run_cuda_graph = not is_prompt and self.use_cuda_graph and bool(self.cuda_graph_runner.graphs)
if run_cuda_graph:
    graph_size, cuda_graph_used = self.cuda_graph_runner.select(total_q)        # bisect 补 padding
    self._prepare_input_tensors(
        seq_metadata_list + self.cuda_graph_runner.padding_seq_metadata_list[: graph_size - total_q],
        out_dict=cuda_graph_used.input_metadata_source.to_dict(),               # ★ 直接写 graph 的 buffer
        metadata_zos=metadata_zos)
    input_metadata = self._graph_input_metadata(cuda_graph_used, num_reqs)      # 设备上的常驻 metadata
else:
    input_metadata = self._prepare_input_tensors(seq_metadata_list, metadata_zos=metadata_zos)
    input_metadata = input_metadata.recursive_fn(lambda x: x.cuda(non_blocking=True))   # CPU → GPU
if PG.context_parallel_group().size() > 1:
    broadcast_tensor_dict({...})                                                 # follower 重建（graph 时只广播标量！）
```

三个关键观察：

1. **graph 模式下 prepare 写的是 graph 捕获时的 buffer**（`out_dict`）——这就是 CUDA graph 静态输入的"原地更新"模式，但 DeepInfer 把它做到了极致：ZOS 下连 H2D 都没有（4.3 节展开）；
2. **CP follower 在 graph 模式下不收 tensor**：广播的 dict 只含 `is_prompt / run_cuda_graph / total_q / num_reqs` 等标量（graph 模式分支不 `to_flatten_dict`），follower 用 `total_q` 自己 `select` 出同一张图——**图是按 rank 对称捕获的，输入靠各自的 buffer 写入**。只有 eager 模式才广播完整 metadata（`to_flatten_dict` 把 dict 字段摊平成 `"field.ratio"` 键，因为 broadcast 只支持一层嵌套）；
3. `use_cuda_graph = not ENFORCE_EAGER and engine_type != PROMPT`——**PROMPT 引擎永不捕图**（prefill 是 varlen 大 batch，图收益低且 capture 集合爆炸；对照 vLLM V1：prefill 默认也不走 full graph）。

### 2.2 四套 metadata：谁产生、谁消费（今日产出一）

`ModelMetadata.get_input_metadata_class_and_prepare_inputs_func()`（`model_metadata_utils.py:142-144`）是总分派：

```python
prepare_decode = prepare_decode_v4_zos if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else prepare_decode_v4
return (InputMetadataV4VL if self.is_vl else InputMetadataV4), prepare_prompt_v4, prepare_decode
```

**README 关键问题 1 的答案**：metadata 分多套不是因为"多种模型"，而是因为 **V4.1-Flash 一个模型身上的四个正交特性各自需要一套字段与视图**——

| 维度 | ① base/V4（`input_metadata.py` + C++ `prepare_tensor_input_v4.cpp`） | ② yoco（`yoco.py` mixin） | ③ dflash（`dflash.py` mixin） | ④ zos（`prepare_tensor_input_zos.py`） |
|---|---|---|---|---|
| **对应模型特性** | MLA 压缩 KV（C1/C2 双压缩率）+ indexer 稀疏注意力 + CED 分层（share-kv） | SWA + YoCo consumer 层（decoder bounded replay） | MTP 投机解码（DSpark draft attention） | 异步调度（上一步采样结果 GPU 常驻） |
| **谁产生** | C++ `prepare_prompt_v4 / prepare_decode_v4`（CPU 构建后 H2D） | C++ `prepare_prompt_v4` 按截断模式二选一产出 `yoco_*` 字段；Python mixin 只提供**视图方法** | Python `build_dflash_input`（MTP 阶段构造，随 DFlash graph 捕获/replay） | Python：C++ `prepare_decode_zos_inputs` 做 pinned CPU 抽取，TileLang kernel 在 **GPU 上**写字段 |
| **谁消费** | 所有层；`compute_attn_info` 在 forward 里展开成 attention 索引 | 第一个 YoCo consumer 层（`yoco_boundary_layer_idx = n_prefill_layer` 处 `get_consumer_view()` / `get_mask_view()` 切换） | DFlash draft attention（`models/mtp/dflash.py`） | decode + ZOS 的 `InputMetadataV4` 字段本身（graph buffer 原地写） |
| **关键字段** | `input_ids/positions/cu_seqlens_*`、`slot_mapping_store_{swa,ca}`、`kv_{swa,ca}_index_offset_len`、`block_tables`、`load_segments_{swa,ca}`、`positions_c`（按压缩率 "N" 分桶的 dict） | `yoco_consumer_{query_indices, cu_seqlens_q, swa_index_pivots, ks/ke, ...}`（compact 布局）或 `yoco_mask_swa_index_pivots`（mask 布局） | `dflash_newx_slot_mapping_store`、`dflash_kv_index`（fixed-stride：每请求 `dflash_block_size` 个 draft token） | 复用 ① 的字段名，但值由 `prepare_v4_decode_metadata` / `gather_input_ids_zos` / `gather_engram_tokens_zos` 在 GPU 上算出 |
| **何时缺席** | — | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=off`（默认）时字段全 None | `dflash_cfg` 为空或 `nextn=0` | ZOS 关闭（PROMPT 引擎）时退回 C++ `prepare_decode_v4` |
| **与 CUDA graph 的关系** | 紧凑字段（pivots/offset_len）写进 graph buffer；展开字段进 `get_clean_list()` 由图内 kernel 重算 | 视图切换发生在**层循环内**（eager prefill），不进图 | **整个 build_dflash_input 随 DFlash 图捕获** | `get_or_create(args.out)`：图模式复用捕获 buffer、eager 模式新分配 |

**为什么"紧凑字段 + forward 期展开"是核心设计**：`kv_ca_index_offset_len` 是 `[total_q, 2]`（每 token 一个 `[start, count]`），而展开后的 `kv_ca_index` 是 `[total_q, pad128(max_model_len/N)]`——差了三个数量级。把展开 kernel（`get_kv_ca_index_decode` 等 TileLang ops）放进 forward（即放进捕获的 graph 里），CPU/prepare 只需要写小小的 pivot 数组，**graph 的静态输入面积大幅缩小**。`get_clean_list()` 返回的正是这些"可重算字段"（`kv_ca_index / kv_swa_index / kv_ca_index_len / mlp_relu_schedule_metadata / flash_mla_sched_meta_dict`），capture 前置 None、replay 后由图内逻辑重新生成。

### 2.3 base/V4 字段速读（挑最密集的 15 个）

`InputMetadataV4` 的字段注释是全仓库写得最细的文档，值得抄进笔记。按用途分组：

| 组 | 字段 | 一句话 |
|---|---|---|
| 通用 | `input_ids` / `positions` / `valid_token_mask` / `gate_force_random` | varlen token、位置、padding 掩码、MoE 负载均衡的"随机 gate"开关（profile 时全 True 模拟最坏路由） |
| 通用 | `num_reqs` / `total_q` / `cp_mirror_len` | 真实请求数（不含 graph pad）；CP 镜像后的本 rank token 数 |
| 压缩 KV | `positions_c`（dict，按压缩率 "N"） | 每个 query 新产的压缩 key 的位置——`(orig_pos+1) % N == 0` 时才有一个新压缩 token |
| 压缩 KV | `slot_mapping_store_swa / slot_mapping_store_ca` | 每个 query token 往哪写：SWA ring（`state_id * swa_block_size + pos % swa_block_size`）或 ca block 池；-1 = 不写 |
| 索引 | `kv_swa_index_offset_len → kv_swa_index`（展开） | 每个 query 的 SWA 窗口 `[start_slot, len]`，指向 state ring |
| 索引 | `kv_ca_index_offset_len → kv_ca_index`（展开） | 压缩 key 的可见范围 `[start, count]`，展开成物理索引（prefill 指向 gathered kv，decode 经 `block_tables` 指向分页 cache） |
| prefill 装载 | `load_segments_swa / load_segments_ca` | **BlockLoadPlan**：块粒度装载描述符 `[src_block_id, src_begin, src_end, dst_offset]`，`src=-1` 表示零填充缺失历史；每序列最多 3 段——这是开源引擎内"前缀复用"的物理形态 |
| prefill 装载 | `kv_swa_full_index_pivots → kv_swa_full_index` | prefill SWA 可见域（历史窗口来自 swa_cache，本 chunk 新 kv 拼接），-1 = masked |
| decode | `block_tables` / `context_lens` / `query_indices` / `klen` | 每 token 的块表、上下文长、所属 seq、稀疏层需要的 key 数 |
| kernel 级 | `mlp_relu_schedule_metadata` / `flash_mla_sched_meta_dict` | deep_gemm paged-mqa-logits 与 FlashMLA 的调度元数据（首个消费者层构建，同配置层共享——defaultdict） |

两个跨切面的事实：

- **字段大量以压缩率为 key 的 dict**（`"1"` / `"2"`）：V4.1 的 dense 层 C1、sparse 层 C2 并存，一套字段两份实例；`to_flatten_dict` 专门把它们摊平成 `"positions_c.1"` 以便 CP 广播；
- **CP-mirror**（`mirror_cp_index / mirror_cp_reverse_index`）：balanced CP 的 prefill 布局重排，`gather(x, mirror_cp_index)` 把 varlen 张量重排到本 rank 的镜像布局。为什么叫 mirror、怎么 balanced——Day 5 专讲。

### 2.4 yoco 视图：层循环中段的"元数据变形"

`deepseek.py:190-207` 的层循环里藏着今天最精妙的一段：

```python
for i in range(self.num_layers):
    if input_metadata.is_prompt and i == self.yoco_boundary_layer_idx:      # = n_prefill_layer
        if envs.DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE == "compact":
            consumer_metadata = input_metadata.get_consumer_view()   # full → compact 布局切换
            cross_layer_runtime.consumer_full_metadata = input_metadata   # full 原件传给 full-source 路径
            input_metadata = consumer_metadata
        else:
            input_metadata = input_metadata.get_mask_view()          # 只改 mask，不做真截断
    hidden_states, residual = self.layers[i](hidden_states, kv_caches[i], input_metadata, ...)
```

- **compact 模式**：前 `n_prefill_layer` 个 encoder 层跑 full 布局（要为 CED 投影产出完整的 decoder block KV），从第一个 consumer 层起 query 行切换到 compact（只保留 keep_tokens 后缀）——**同一个 forward 里两套布局**，靠 `get_consumer_view()` 的浅拷贝视图切换，`CrossLayerRuntime` 在两条路径间搬运 `shared_hidden / pre_rope_kv` 等跨层值；
- **mask 模式**：不删行，只把 SWA 可见域 pivots 换成"滑窗 ∩ keep_tokens 矩形"（矩形外整行 -1）——计算省不了 embedding/FFN，但省 attention 的 key 访问。两种模式"SWA 可见域同式（C++ 共用 yoco_swa_visibility），故 attention 数值在保留区间内一致"（yoco.py 文件头注释），`examples/yoco/run_yoco_equiv_l2.sh` 做逐位等价验证。

### 2.5 一个 decode step 的调用链（今日核心产出，SVG）

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 920" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="d" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#475569"/>
    </marker>
  </defs>
  <rect width="1080" height="920" fill="#fafbfc"/>
  <text x="540" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">一个 decode step：从 scheduler 输出到 kernel launch（ZOS + CUDA graph + MTP_VARLEN）</text>

  <!-- stage 0 -->
  <rect x="60" y="52" width="620" height="44" rx="9" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="370" y="79" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1e1b4b">engine.step() → scheduler.schedule(pool)</text>
  <text x="710" y="79" font-size="11.5" fill="#6366f1" font-family="Consolas, monospace">engine/llm_engine.py:885 · csrc/engine/scheduler.hpp</text>
  <line x1="370" y1="96" x2="370" y2="118" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>

  <!-- stage 1 -->
  <rect x="60" y="120" width="620" height="64" rx="9" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>
  <text x="370" y="143" text-anchor="middle" font-size="13.5" font-weight="700" fill="#713f12">① seqlens 修正（Day 2 已讲，GPU 侧）</text>
  <text x="370" y="163" text-anchor="middle" font-size="11.5" fill="#854d0e">prepare_zos_inputs(C++) → confidence kernel → seqlens_k_zos → D2H pinned</text>
  <text x="710" y="152" font-size="11.5" fill="#a16207" font-family="Consolas, monospace">worker/worker.py:191 · zos.cpp</text>
  <line x1="370" y1="184" x2="370" y2="206" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>

  <!-- stage 2 -->
  <rect x="60" y="208" width="620" height="80" rx="9" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/>
  <text x="370" y="231" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1e3a8a">② model_runner.execute_main_model（禁止任何 CUDA 同步）</text>
  <text x="370" y="251" text-anchor="middle" font-size="11.5" fill="#1e40af">_prepare_input → [graph: select(total_q) 补 padding + 写 graph buffer | eager: CPU 构建 + H2D]</text>
  <text x="370" y="270" text-anchor="middle" font-size="11.5" fill="#1e40af">分派：prompt→C++ prepare_prompt_v4；decode→prepare_decode_v4_zos(ZOS)/C++(非ZOS)；CP>1 广播</text>
  <text x="710" y="245" font-size="11.5" fill="#2563eb" font-family="Consolas, monospace">worker/model_runner.py:148-230</text>
  <line x1="370" y1="288" x2="370" y2="310" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>

  <!-- stage 2b zos prepare detail -->
  <rect x="700" y="208" width="330" height="80" rx="9" fill="#dbeafe" stroke="#2563eb" stroke-width="1"/>
  <text x="865" y="230" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">prepare_decode_v4_zos 内部（全 GPU）</text>
  <text x="865" y="249" text-anchor="middle" font-size="10.5" fill="#1e40af">C++ prepare_decode_zos_inputs：pinned CPU 抽取</text>
  <text x="865" y="265" text-anchor="middle" font-size="10.5" fill="#1e40af">gather_input_ids_zos：prev token D2D scatter 拼 input_ids</text>
  <text x="865" y="281" text-anchor="middle" font-size="10.5" fill="#1e40af">prepare_v4_decode_metadata：一次 kernel 写 12 类字段</text>
  <line x1="700" y1="248" x2="680" y2="248" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="4,3"/>

  <!-- stage 3 graph or eager -->
  <rect x="60" y="312" width="300" height="96" rx="9" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.8"/>
  <text x="210" y="335" text-anchor="middle" font-size="13" font-weight="700" fill="#14532d">③a CUDA graph 路径</text>
  <text x="210" y="355" text-anchor="middle" font-size="11" fill="#166534">select(total_q)：bisect 补到下一个尺寸</text>
  <text x="210" y="371" text-anchor="middle" font-size="11" fill="#166534">graph.replay()（graph_timer ping-pong）</text>
  <text x="210" y="387" text-anchor="middle" font-size="11" fill="#166534">ZOS：graph 内零 H2D（input_on_device）</text>
  <rect x="400" y="312" width="280" height="96" rx="9" fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="540" y="335" text-anchor="middle" font-size="13" font-weight="700" fill="#7f1d1d">③b eager 路径</text>
  <text x="540" y="355" text-anchor="middle" font-size="11" fill="#991b1b">model(kv_caches, input_metadata)</text>
  <text x="540" y="371" text-anchor="middle" font-size="11" fill="#991b1b">（prefill 恒走此路；decode 仅在</text>
  <text x="540" y="387" text-anchor="middle" font-size="11" fill="#991b1b"> ENFORCE_EAGER / 图未捕获时）</text>
  <line x1="370" y1="310" x2="210" y2="312" stroke="#475569" stroke-width="0"/>
  <line x1="280" y1="290" x2="210" y2="312" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>
  <line x1="460" y1="290" x2="540" y2="312" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>
  <text x="710" y="330" font-size="11.5" fill="#16a34a" font-family="Consolas, monospace">worker/cuda_graph_runner.py:143</text>
  <line x1="210" y1="408" x2="210" y2="430" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>
  <line x1="540" y1="408" x2="540" y2="430" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>

  <!-- stage 4 model forward -->
  <rect x="60" y="432" width="620" height="210" rx="9" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.8"/>
  <text x="370" y="455" text-anchor="middle" font-size="13.5" font-weight="700" fill="#581c87">④ DeepseekModel.forward —— N 层循环（模型结构 Day 4 精读）</text>
  <rect x="80" y="468" width="270" height="56" rx="7" fill="#fae8ff" stroke="#a855f7"/>
  <text x="215" y="489" text-anchor="middle" font-size="11" font-weight="600" fill="#581c87">compute_attn_info：紧凑字段 → attention 索引</text>
  <text x="215" y="507" text-anchor="middle" font-size="10.5" fill="#7e22ce">TileLang：get_kv_*_index / expand_swa_index</text>
  <rect x="370" y="468" width="290" height="56" rx="7" fill="#fae8ff" stroke="#a855f7"/>
  <text x="515" y="489" text-anchor="middle" font-size="11" font-weight="600" fill="#581c87">yoco 边界（prefill）：get_consumer_view / mask_view</text>
  <text x="515" y="507" text-anchor="middle" font-size="10.5" fill="#7e22ce">i == n_prefill_layer 时 metadata 变形</text>
  <rect x="80" y="532" width="580" height="44" rx="7" fill="#fae8ff" stroke="#a855f7"/>
  <text x="370" y="550" text-anchor="middle" font-size="11" font-weight="600" fill="#581c87">DecoderLayer：MLA（FlashMLA vendor + store_kv_cache TileLang）</text>
  <text x="370" y="568" text-anchor="middle" font-size="10.5" fill="#7e22ce">MoE（DeepGEMM grouped GEMM + DeepEP all-to-all）· indexer（DeepSelect + paged-mqa-logits）</text>
  <rect x="80" y="582" width="580" height="48" rx="7" fill="#fae8ff" stroke="#a855f7"/>
  <text x="370" y="601" text-anchor="middle" font-size="11" font-weight="600" fill="#581c87">收尾：dflash.update_main_cache（目标层特征写 DFlash cache，随主图捕获）</text>
  <text x="370" y="619" text-anchor="middle" font-size="10.5" fill="#7e22ce">CP>1：userbuffer.cp.all_gather + mirror_cp_reverse_index 还原；HC head / output_norm</text>
  <text x="710" y="530" font-size="11.5" fill="#a855f7" font-family="Consolas, monospace">models/deepseek.py:154-248</text>
  <line x1="370" y1="642" x2="370" y2="664" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>

  <!-- stage 5 post process prev -->
  <rect x="60" y="666" width="620" height="40" rx="9" fill="#fff7ed" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="6,3"/>
  <text x="370" y="691" text-anchor="middle" font-size="12.5" font-weight="700" fill="#7c2d12">⑤ _post_process_step(prev)（Day 2：处理上一步输出，此处 GPU 正跑 ④）</text>
  <line x1="370" y1="706" x2="370" y2="724" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>

  <!-- stage 6 sample -->
  <rect x="60" y="726" width="620" height="74" rx="9" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8"/>
  <text x="370" y="749" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1e3a8a">⑥ execute_sample（d2h_evt 同步后）</text>
  <text x="370" y="769" text-anchor="middle" font-size="11.5" fill="#1e40af">prepare_sampler_inputs(C++) → model.sample：目标模型采样（fused_sample TileLang）</text>
  <text x="370" y="788" text-anchor="middle" font-size="11.5" fill="#1e40af">→ DSpark：draft 模型 run_drafts（dflash graph replay）+ confidence head + varlen 验证</text>
  <text x="710" y="760" font-size="11.5" fill="#2563eb" font-family="Consolas, monospace">model_runner.py:337 · layers/sampler.py · mtp/dflash.py</text>
  <line x1="370" y1="800" x2="370" y2="818" stroke="#475569" stroke-width="2" marker-end="url(#d)"/>

  <!-- stage 7 -->
  <rect x="60" y="820" width="620" height="40" rx="9" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="370" y="845" text-anchor="middle" font-size="12.5" font-weight="700" fill="#14532d">⑦ pre_sampler_sync：DeferredRelease + do_save_kvcache + SO pump → record sample_end_evt → return（上一步的输出）</text>

  <!-- legend -->
  <rect x="700" y="666" width="330" height="194" rx="9" fill="#f8fafc" stroke="#94a3b8"/>
  <text x="718" y="690" font-size="12" font-weight="700" fill="#0f172a">图例与要点</text>
  <text x="718" y="712" font-size="10.5" fill="#334155">蓝 = Python 引擎/runner；紫 = 模型 forward</text>
  <text x="718" y="730" font-size="10.5" fill="#334155">绿 = graph 路径；黄 = ZOS 修正；橙 = 延迟处理</text>
  <text x="718" y="752" font-size="10.5" fill="#334155">④⑤ 重叠：GPU 跑 forward(N) 时 CPU 做 post-process(N−1)</text>
  <text x="718" y="770" font-size="10.5" fill="#334155">③a 中 graph 捕获了 ④ 的全部 kernel + CP 广播 +</text>
  <text x="718" y="786" font-size="10.5" fill="#334155">  compute_attn_info 的展开 kernel（get_clean_list）</text>
  <text x="718" y="808" font-size="10.5" fill="#334155">⑥ 的 dflash graph 与主图共享 memory pool/stream</text>
  <text x="718" y="826" font-size="10.5" fill="#334155">每个箭头都发生在 CPU 侧的 launch 阶段，</text>
  <text x="718" y="842" font-size="10.5" fill="#334155">除 ⑤ 的一次 event 同步外无任何 GPU 等待</text>
</svg>
```

---

## 三、cache_engine：KV cache 的物理布局与开源边界

### 3.1 两池结构：分页 KV + 每序列 state 槽

`cache_engine.py`（978 行）的核心抽象只有两个物理池，都在 `CacheEngine.allocate_gpu_cache` 里分配：

```python
blocks = self._alloc_gpu_uint8((self.num_gpu_blocks, *block_shape), "kvcache")   # 分页 KV
gpu_states = self._alloc_gpu_uint8(state_shape, "state")                          # 每序列 state 槽
# state_shape = (get_num_state_slots(max_num_seqs), total_state_bytes)
#   num_state_slots = max_num_seqs × 1.2 + 1（DEEPINFER_NUM_STATE_SLOTS_RATIO）
```

三个立即可见的工程决策：

1. **`_alloc_gpu_uint8` 走 `cuda_utils.alloc_tensor`（cuMemAlloc 直分）**，绕过 torch caching allocator——KV 池是巨型、永不释放、地址必须稳定的分配（CUDA graph 捕获时指针被烧进图里），放进会碎片化的 caching allocator 是自找麻烦。注意对照 vLLM：vLLM 的 KV 也是独立分配（`torch.empty` 后由 allocator 管理），但 DeepInfer 更进一步完全脱离 torch 记账（profile 注释明说"不计入 torch 显存统计"）；
2. **`num_state_slots = max_num_seqs × 1.2 + 1`**：state 槽比并发上限多 20%——PD 传输/落盘中的"在途"序列也要占槽（Day 1 的 in-flight state KV）；
3. **slot 0 是 dummy**：`BlockManager` 保留不分配（deepseek.py:222 注释："CUDA graph 的 padding 行落在 placeholder 的 state 0"）。

### 3.2 层分类学：V4.1 没有 full attention

`GPUCacheInterface` 的三个静态分类器（`cache_engine.py:682-699`）定义了整个布局的坐标系：

| 分类 | 判定 | 存储形态 |
|---|---|---|
| SWA 层 | `sink_mode == "swa"` | **不占分页 KV**！住 state 槽的 ring（`swa_cache`） |
| sparse 层 | `index_topk > 0` | indexer 的 K 走分页池（`ica_cache`），可能带压缩 state |
| 压缩层 | `kv_compress_ratio ∈ {1, 2}` | 主 KV 走分页池（`ca_cache`），ratio>1 带压缩 state |
| full 层 | ratio=0 且非 swa | **直接 raise**——"KVCacheManager 不支持 full attention 层" |

每层的 `KVCache`（dataclass）：

```python
@dataclass
class KVCache:
    ica_cache  # 分页：[num_blocks, block_size//ratio, 1, index_bytes_per_token] uint8
    ca_cache   # 分页：[num_blocks, block_size//ratio, 1, bytes_per_token] uint8（V41: [data|scale] 每 token 288B FP4 或 528B MXFP8）
    swa_cache  # state 槽 ring：[num_state_seqs, swa_block_size, 1, 528] uint8（512B FP8 data + 16B UE8M0 scale）
    ica_state / ca_state  # state 槽内 fp32 压缩边界累加器：[num_state_seqs, ring_size, 2, dim]
```

**块内布局是把所有层打包进同一个 block**：`layout = [(layer_id, kv_offset, kv_length), ...]`——每个 block 的字节轴上依次排列各层的 ica/ca 段（每段 32B 对齐，总长对齐到 `lcm(align_to, bytes_per_token)`；DeepGEMM 要求 block_stride 是 128 的倍数）。对照 vLLM V1：`kv_cache` 是 per-layer 张量列表（`[num_blocks, block_size, kv_heads, head_dim]` 每层一份），block table 共享但物理张量按层分离。DeepInfer 的"单池 + 层偏移"让一个 block 的装载/落盘是**一段连续字节**（`load_segments` 的 `[src_block_id, src_begin, src_end, dst_offset]` 直接就是字节区间拷贝）——这是为 3FS 落盘和 PD 传输优化的布局。

**share-kv / share-index**（`_get_kv_mapped_layer_id`，`layer_info_shared_cfg.layer_kv_map / layer_index_k_map`）：某层可以复用前面层的 cache（"复用只能指向前面已经处理过的层"）。这是 **CED 架构的代码证据**——Q&A 说"decoder 层的 block KV 由最后一个 encoder 层的 hidden state 投影而来"，投影出的 KV 让多个 decoder 层共享同一份物理 cache，`process()` 里这些层 `kv_length == 0`、不推进 offset。

### 3.3 StateBlock：一个序列的 state 槽布局（SVG）

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 640" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <rect width="1080" height="640" fill="#fafbfc"/>
  <text x="540" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">KV cache 物理布局：gpu_blocks（分页池）+ gpu_states（StateBlock 槽）</text>

  <!-- gpu_blocks -->
  <rect x="50" y="60" width="470" height="240" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>
  <text x="285" y="86" text-anchor="middle" font-size="14" font-weight="700" fill="#1e3a8a">gpu_blocks：[num_gpu_blocks, bytes_per_block]（uint8，cuMemAlloc 直分）</text>
  <!-- one block magnified -->
  <rect x="75" y="104" width="420" height="120" rx="6" fill="#dbeafe" stroke="#2563eb"/>
  <text x="285" y="122" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">block #i 内部（所有层打包在字节轴上，每段 32B 对齐）</text>
  <rect x="90" y="132" width="60" height="34" fill="#bfdbfe" stroke="#2563eb"/>
  <text x="120" y="146" text-anchor="middle" font-size="9.5" fill="#1e3a8a">L0 ica</text>
  <text x="120" y="160" text-anchor="middle" font-size="9" fill="#1e40af">(index K)</text>
  <rect x="152" y="132" width="90" height="34" fill="#93c5fd" stroke="#2563eb"/>
  <text x="197" y="146" text-anchor="middle" font-size="9.5" fill="#1e3a8a">L0 ca (C1)</text>
  <text x="197" y="160" text-anchor="middle" font-size="9" fill="#1e40af">288B/2tok FP4</text>
  <rect x="244" y="132" width="60" height="34" fill="#bfdbfe" stroke="#2563eb"/>
  <text x="274" y="146" text-anchor="middle" font-size="9.5" fill="#1e3a8a">L1 ica</text>
  <text x="274" y="160" text-anchor="middle" font-size="9" fill="#1e40af">共享→跳过</text>
  <rect x="306" y="132" width="90" height="34" fill="#93c5fd" stroke="#2563eb"/>
  <text x="351" y="146" text-anchor="middle" font-size="9.5" fill="#1e3a8a">L1 ca (C2)</text>
  <text x="351" y="160" text-anchor="middle" font-size="9" fill="#1e40af">528B/tok MXFP8</text>
  <rect x="398" y="132" width="82" height="34" fill="#e2e8f0" stroke="#94a3b8"/>
  <text x="439" y="152" text-anchor="middle" font-size="9.5" fill="#475569">... 层间 pad</text>
  <text x="285" y="192" text-anchor="middle" font-size="10.5" fill="#1e40af">layout=[(layer, kv_offset, kv_length)]；share-kv 层 kv_length=0（CED：decoder 复用 encoder 投影 KV）</text>
  <text x="285" y="212" text-anchor="middle" font-size="10.5" fill="#1e40af">block 总长对齐 lcm(align_to, bytes_per_token)；DeepGEMM 要求 128B 倍数</text>
  <rect x="75" y="228" width="420" height="56" rx="6" fill="#f0f9ff" stroke="#0284c7"/>
  <text x="285" y="248" text-anchor="middle" font-size="11" font-weight="600" fill="#075985">block_tables: [total_q, max_num_blocks]（64 对齐）—— 逻辑块 → 物理块</text>
  <text x="285" y="268" text-anchor="middle" font-size="10.5" fill="#075985">BlockManager(C++) 分配/克隆/释放；128 token/block（KVSTORE_GRAIN）</text>

  <!-- gpu_states -->
  <rect x="560" y="60" width="470" height="240" rx="10" fill="#fef2f2" stroke="#dc2626" stroke-width="2"/>
  <text x="795" y="86" text-anchor="middle" font-size="14" font-weight="700" fill="#7f1d1d">gpu_states：[num_state_slots, total_state_bytes]（每序列一个 StateBlock 槽）</text>
  <rect x="585" y="104" width="420" height="120" rx="6" fill="#fee2e2" stroke="#dc2626"/>
  <rect x="595" y="120" width="230" height="90" fill="#86efac" stroke="#16a34a"/>
  <text x="710" y="140" text-anchor="middle" font-size="11.5" font-weight="700" fill="#14532d">stored region：SWA</text>
  <text x="710" y="158" text-anchor="middle" font-size="10" fill="#166534">swa_store 层（前 20 层）× ring</text>
  <text x="710" y="174" text-anchor="middle" font-size="10" fill="#166534">ring = swa_block_size(134) × 528B</text>
  <text x="710" y="190" text-anchor="middle" font-size="10" fill="#166534">= 70,752 B / 层</text>
  <text x="710" y="205" text-anchor="middle" font-size="10" font-weight="700" fill="#14532d">state_size_to_disk = 20 × 70,752 = 1,415,040 B ✓</text>
  <rect x="830" y="120" width="165" height="90" fill="#fecaca" stroke="#dc2626"/>
  <text x="912" y="140" text-anchor="middle" font-size="11" font-weight="700" fill="#7f1d1d">dropped region</text>
  <text x="912" y="157" text-anchor="middle" font-size="9.5" fill="#991b1b">YOCO consumer SWA</text>
  <text x="912" y="171" text-anchor="middle" font-size="9.5" fill="#991b1b">(n_layer − n_prefill 层)</text>
  <text x="912" y="187" text-anchor="middle" font-size="9.5" fill="#991b1b">ca_state / ica_state</text>
  <text x="912" y="201" text-anchor="middle" font-size="9.5" fill="#991b1b">(fp32 压缩边界累加器)</text>
  <text x="795" y="242" text-anchor="middle" font-size="10.5" fill="#991b1b">SWA 落盘分区只由 n_prefill_layer / n_layer 决定（与 PE/CE 截断模式无关）</text>
  <text x="795" y="262" text-anchor="middle" font-size="10.5" fill="#991b1b">num_state_slots = max_num_seqs × 1.2 + 1（在途序列占槽）；slot 0 = dummy（graph padding 落点）</text>
  <text x="795" y="284" text-anchor="middle" font-size="10.5" fill="#991b1b">KVSTORE_GRAIN = 128：只在 128 倍数位置落盘</text>

  <!-- numbers band -->
  <rect x="50" y="322" width="980" height="150" rx="10" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
  <text x="540" y="348" text-anchor="middle" font-size="13.5" font-weight="700" fill="#78350f">数字闭环（window_size=128, nextn=5 → 默认 DEEPINFER_MAX_BACKTRACKING_TOKENS = max(n, nextn+1) = 6）</text>
  <text x="540" y="374" text-anchor="middle" font-size="12" fill="#92400e">inference ring = window 128 + backtrack 6 = <tspan font-weight="700">134</tspan>（= Day 1 的 L−134 上限！128 重算窗 + 6 投机回退）</text>
  <text x="540" y="398" text-anchor="middle" font-size="12" fill="#92400e">kvstore ring = 134 + (PREFILL_STORE_EXTRA_SIZE 128 + yoco extra 140)（仅 DEEPINFER_PREFILL_STORE_EXTRA_STATE=1 时，默认关）</text>
  <text x="540" y="422" text-anchor="middle" font-size="12" fill="#92400e">load_model 断言：window_size + MAX_BACKTRACKING ≤ KEEP_TOKENS（140 ≥ 128+6 ✓）——保证回退后仍有完整 SWA 窗</text>
  <text x="540" y="446" text-anchor="middle" font-size="12" fill="#92400e">SWA 528B/token = 512B MXFP8 data + 16B UE8M0 scale；ca FP4 = per-16 e4m3 288B/2tok；ica = head_dim/2 + head_dim/32</text>

  <!-- interfaces band -->
  <rect x="50" y="490" width="980" height="128" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.8"/>
  <text x="540" y="516" text-anchor="middle" font-size="13.5" font-weight="700" fill="#14532d">开源边界：KV 系统的接口已备好，调用方未开源</text>
  <text x="540" y="542" text-anchor="middle" font-size="12" fill="#166534">select_state_slice_to_store(state_id, last_hit_pos, ..., TO_CE)：从 state 槽切出落盘/PE→CE 段（落盘要求 128 对齐；TO_CE 含 dropped 区）</text>
  <text x="540" y="566" text-anchor="middle" font-size="12" fill="#166534">load_state_from_store_to_prefill_cache(...)：把落盘 state 装回 prefill cache（extra-state 模式做 ring 重排）</text>
  <text x="540" y="590" text-anchor="middle" font-size="12" fill="#166534">engine 侧钩子：do_save_kvcache / pe_*_save_kvcache（虚函数，开源树里无覆盖者）；do_copy_state → copy_state buffer（SWA 落盘快照）</text>
  <text x="540" y="612" text-anchor="middle" font-size="11.5" fill="#166534">tests/worker/test_state_cache_ops.py 用真实 V4.1 config 验证这两个接口（无硬件时的契约来源）</text>
</svg>
```

`StateBlock.__init__` 的注释把 Q&A 的 SWA KV 描述逐句翻译成了代码（"需要落盘的 state 全部放在前面的 stored region...PE→CE 传输两部分；to disk 只取 inference 布局的前 state_size_to_disk 字节"），分区原则、对齐规则（SWA 每层 32B、整槽 `lcm(32, 528)`）都在。**手算验证 `state_size_to_disk = 20 层 × 134 token × 528 B = 1,415,040 B`**——与 Day 1 抄录的 Q&A 数字完全闭环；同时确认了 134 = window 128 + backtrack 6，而 `DEEPINFER_MAX_BACKTRACKING_TOKENS` 的默认值定义（`environ.py:132`）是 `max(显式值, nextn+1)`——**nextn=5 → 6**，这就是 L−134 里那个 6 的出处。

### 3.4 profile_run：显存估算的"最坏情况"

`model_runner.profile_run`（启动时 `_init_cache` 调用）构造合成 batch 跑一次前向，有三个 vLLM 没有的细节：

1. **`DEEPINFER_FORCE_IMBALANCE_RATIO = 1.8`**——profile 时人为制造 MoE 最坏不均衡（gate_force_random 全开），激活显存按最坏路由估算；
2. **`DEEPINFER_PREFILL_MAX_SUM_SEQLEN_K`**：合成序列的 sum(seqlen_k) 尽量顶到这个上限（vLLM 只用 max_num_batched_tokens 个 token 平摊）；seq_lens_k[0] 固定为 runtime_max_model_len——**同时覆盖"长上下文"和"大 batch"两种显存峰值形态**；
3. **临时 KV 池参与 profile**："用一小块真实 kv cache 跑 profile，计入 cache 读写 kernel 的显存开销"（cuMemAlloc 分配、随函数返回释放、置零防脏读）。

### 3.5 vLLM 对照小结

| 维度 | vLLM V1 | DeepInfer |
|---|---|---|
| 物理组织 | per-layer 张量，每层 `[num_blocks, block_size, kv_heads, head_dim]` | **单 uint8 池 + 层内偏移**，一个 block 含所有层（连续字节利于落盘/搬运） |
| KV 内容 | 均匀 per-layer（同构模型） | 异构：ica（indexer）/ ca（压缩主 KV）/ swa（ring）三种形态混布 |
| 独立 state 池 | 无（GQA KV 全在分页池） | `gpu_states` 每序列槽：SWA ring + fp32 压缩边界（为落盘 128 粒度设计） |
| 分配器 | torch allocator | cuMemAlloc 直分（graph 指针稳定 + 不进 torch 记账） |
| block manager | `BlockPool`（哈希链 prefix caching） | C++ `BlockManager`（clone_states/zero_blocks 支持抢占与 fork），**prefix caching/3FS 层未开源** |

---

## 四、CUDA graph：capture 集合、共享池与双 runner

### 4.1 capture 集合：以 total_q（token 数）为键

```python
def build_graph_batch_sizes(max_size: int, interval: int) -> list[int]:
    """Build power-of-two warmup sizes followed by fixed-width intervals."""
    interval = min(interval, max_size)
    batch_sizes, size = [], 1
    while size < interval:
        batch_sizes.append(size); size *= 2          # [1, 2, 4]（interval=8 时）
    batch_sizes.extend(range(interval, max_size, interval))
    batch_sizes.append(max_size)
    return batch_sizes
```

`CudaGraphRunner.__init__` 的 `max_batch_size = max_num_seqs × (1 + nextn)`——**单位是 query token 数**，不是请求数！这是 DSpark varlen 的直接后果：一个 step 的 total_q ∈ [num_reqs, num_reqs × (1+nextn)]，由上一步接受数决定，几乎是任意整数，所以：

- 键必须用 total_q（vLLM V1 的 `cudagraph_capture_sizes` 以 batch size 为键，因为非投机下 decode 的 token 数 = batch size；vLLM 开 MTP 后同样按 `bs × (1+draft)` 推导 token 数，思想同源）；
- 粒度取"2 的幂 + 固定间隔 8"（`DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL`），小 batch 稀疏（pad 浪费比例高但绝对值小）、大 batch 每 8 个 token 一档（pad 浪费 ≤7 token）。

**手算两例**（今日实验 3）：

- 离线样例（max_num_seqs=4, nextn=5）：max = 4×6 = 24 → sizes = [1, 2, 4, 8, 16, 24]，共 6 张图；
- 生产级（max_num_seqs=256, nextn=5）：max = 1536 → sizes = [1,2,4] + range(8,1536,8)（190 个）+ 1536 ≈ **194 张图**。

`select(total_q)` 用 `bisect_left` 找第一个 ≥ total_q 的尺寸，差值用 `padding_seq_metadata_list[:graph_size - total_q]` 个占位 seq 填充（占位 seq 的 draft 信息写到被 store kernel 跳过的 slot 0）。

### 4.2 capture 流程：从大到小、共享内存池

`CudaGraphRunner.capture`（`cuda_graph_runner.py:82-141`）的顺序本身就是设计：

```python
self.padding_seq_metadata_list = [padding_seq_metadata] * self.max_batch_size
input_metadata_source_full = prepare_input_fn(self.padding_seq_metadata_list)      # 全尺寸 CPU/设备 metadata
input_metadata_device_full = input_metadata_source_full.recursive_fn(.cuda())      # ZOS 下与 source 共享存储！
output_full = model.allocate_output(self.max_batch_size)

for batch_size in reversed(self.batch_sizes):          # ★ 从最大开始
    input_metadata_source = input_metadata_source_full.recursive_fn(_cut)   # 前缀切片视图
    input_metadata_device = input_metadata_device_full.recursive_fn(_cut)
    output = output_full.narrow(batch_size)
    # warmup（side stream）→ 清空 get_clean_list 字段 → capture：
    with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=self.graph_stream,
                          capture_error_mode="thread_local"):
        model_output = _run_model(model, kv_caches, input_metadata_source, input_metadata_device, not self.input_on_device)
        output.copy_(model_output)
    if PLATFORM.is_gpu: graph.instantiate()
    if self.graph_memory_pool is None: self.graph_memory_pool = graph.pool()   # 后续图共享第一张的池
```

要点逐条：

1. **从大到小捕获 + 全体共享第一个图的 memory pool**：大图先占住峰值显存，小图复用同一池——所有图合计的额外显存 ≈ 最大那张图。`_cut` 前缀切片保证小图的 buffer 是大图 buffer 的行视图（`get_or_create_stacked` 的断言就在守护这个"等距行视图"不变量：capture 流程被改动破坏零拷贝前提时立即报错）；
2. **`get_clean_list()` 清空可重算字段再捕获**：图内包含 `compute_attn_info` 的展开 kernel，replay 时自动重算——"紧凑字段 + 图内展开"（2.2 节）；
3. **`capture_error_mode="thread_local"`**：捕获失败只终结本线程——避免别的线程（SO 编译等）的 CUDA 调用污染捕获；
4. **平台差异**：GPU 用 `torch.cuda.CUDAGraph(keep_graph=True)`（保留原始 cudaGraph_t 供 graph_timer 匹配事件节点）+ 显式 `instantiate()`；NPU"只有 instantiated graph"，不支持这两个参数——**双平台抽象在 graph 层的又一次现身**；
5. **graph_timer**：`GraphHandle(graph)` 注册后，`replay()` 前做 `flip_event_buffer`——ping-pong 事件缓冲区实现**逐图计时**而不打断 replay（`csrc/graph_timer/` 的 C++ 侧在 Day 7 profiling 详述）。

### 4.3 ZOS 与 graph 的合体：`input_on_device`

`CudaGraphData` 的 docstring 一句话点破："Source tensors are on CPU, or **share device tensor storage under ZOS**"。两条路径对比：

| | 非 ZOS（PROMPT/显式关闭） | ZOS（默认） |
|---|---|---|
| `input_metadata_source` | pinned CPU 张量 | **设备张量**（`prepare_decode_v4_zos` 的 `args.out=None` 时直接在 CUDA 上分配，`.cuda()` 无操作返回自身） |
| graph 内是否含 H2D | 含（捕获了 `recursive_copy_`，replay 时从 pinned 源拷入） | **不含**（`copy_inputs=not self.input_on_device=False`） |
| replay 前的输入写入 | CPU 写 pinned 源 → 图内 H2D | **TileLang/C++ kernel 直接写图捕获的设备 buffer**（`get_or_create(args.out)` 复用） |

这回答了一个 Day 2 遗留的问题：ZOS 的"零开销"不只靠穿插，还靠**输入路径完全设备化**——metadata 的生产（`prepare_v4_decode_metadata` 一次 kernel 写 12 类字段、`gather_input_ids_zos` 的 D2D scatter 拼 token）全部发生在 GPU 上，CPU 在整个 prepare 阶段只做 `prepare_decode_zos_inputs` 的 pinned 抽取（C++，无 Python 循环）。

### 4.4 dflash_cuda_graph_runner：为什么 DSpark 要单独的图（README 关键问题 3）

```python
self.dflash_use_cuda_graph = (self.use_cuda_graph and not DEEPINFER_DFLASH_ENFORCE_EAGER
                              and self.nextn > 0 and model_cfg.gpt.get("dflash_cfg") is not None)
self.dflash_cuda_graph_runner = DFlashCudaGraphRunner(max_num_seqs=self.scheduler_config.max_num_seqs,
                                                      do_sample=self.is_driver_worker)
```

**分工边界**（`dflash_cuda_graph_runner.py` 全文 100 行）：

| 维度 | 主模型 CudaGraphRunner | DFlashCudaGraphRunner |
|---|---|---|
| batch 键 | **total_q（token 数）**，max = max_num_seqs×(1+nextn) | **req_cnt（请求数）**，max = max_num_seqs |
| 原因 | varlen 验证的 query 行数任意 | draft 是 **fixed-stride**：每请求固定 `dflash_block_size` 个 draft token（`dflash_kv_index` 形状 `[req_cnt × dflash_block_size, pad64(window+block)]`） |
| 捕获对象 | `model(kv_caches, input_metadata)` 整个 forward | `module.run_drafts(main_input_metadata, decode_input, nextn_kv_caches)`——**DFlash draft 模型的自回归生成** |
| memory pool / stream | 自建 | **复用主图的 pool 和 stream**（`capture_dflash_model` 显式传入，日志还打印 DFlash 图的额外显存） |
| 输入更新 | prepare 写 buffer / graph 内 H2D | `copy_dflash_decode_input`：replay 前把真实 `MTPDecodeInput` 拷进捕获 buffer（`MTPDecodeInput.build_padding` 造 pad 行） |
| 输出 | `CausalLMForwardOutput.narrow` | `MTPPredictInfo`（token_ids + confidence_rates）；**`do_sample` 只在 driver**（CP slave 不采样） |
| 消费时机 | engine.step 的第 2 段 | **execute_sample 内**（主 forward 之后，作为 `model.sample` 的一部分被调用） |

为什么必须单独一个 runner？三个原因：**① batch 维度不同**（total_q vs req_cnt，两套尺寸集合没法共用键）；**② 生命周期不同**（draft 跑在 sample 阶段，嵌在主图之外，但又要复用主图的池子省显存）；**③ 输入输出类型完全不同**（`MTPDecodeInput`：`first_token_positions`/`last_valid_offset`/noise token 填充，输出是 `MTPPredictInfo`）。`main_input_metadata` 直接取最大主图的 `input_metadata_device`——**draft 的 KV 索引直接复用主模型的 metadata**（`build_dflash_input` 是它的浅拷贝改写），这是"draft 看 target 的 KV"的物理实现。

对照 vLLM V1：EAGLE/MTP 的 draft 模型同样有独立 graph runner（相同思想——draft 的 batch 语义与 target 不同）；DeepInfer 特有的是 fixed-stride 布局（为 confidence-scheduled varlen 验证服务，Day 6a 展开）和与主图共享 pool/stream 的显存节俭。

---

## 五、decode_perf_cache：调度器的硬件感知从哪来

Day 2 看到 `scheduler.decode_perf_curve = driver_worker.decode_perf_curve` 和 `update_mtp_runtime`——confidence-scheduled verification 需要"每个 batch size 下 forward/sample 各花多少毫秒"的硬件曲线。这条曲线的来源是 `model_runner.profile_decode_perf`（`warm_up_model` 里、图捕获之后调用）：

```python
# 各 batch size 共享最大 batch buffer 的切片视图，填一次即可作用于所有 graph
input_metadata_source.valid_token_mask.fill_(True)       # 全有效 = 最坏情况
input_metadata_source.gate_force_random.fill_(True)      # MoE 最坏路由
for batch_size in reversed(graph_runner.batch_sizes):
    # Phase 1: NUM_WARMUP=1 次热身（间隔 EP 组 all_reduce 同步）
    # Phase 2: NUM_MEASURE=5 次 CUDA event 计时
    # Phase 3: 截尾均值 —— 去掉最大最小各 1 个（NUM_TRIM=1）
    sum_time = ...; dist.all_reduce(SUM, group=PG.expert_parallel_group())    # EP 组取平均！
    avg_time_ms = sum_time.item() / group.size()
```

三个细节：

1. **EP 组 all_reduce 平均**：同一 batch size 下不同 rank 的 expert 负载不同，取全组均值才是"这个 batch size 的单 step 成本"——调度器的预算模型必须与 EP 行为一致；
2. **截尾均值**（5 测去头尾）+ warmup：对抗调度噪声和 JIT 残余；
3. **曲线可缓存**：`DecodePerfCache`（`decode_perf_cache.py`）把 `[(bs, ms)]` 存到 `DEEPINFER_DECODE_PERF_CACHE_DIR`：
   - cache key = sha256(`{time_unit:ms, model, batch_sizes, dp/cp/ep sizes}`)——**并行度进 key**，换拓扑自动失效；
   - 文件名 `<hash>.<uuid4>.<unix秒>[.tmp]`；TTL 默认 1 小时，**清理阈值比读取 TTL 宽 2 小时**（"文件先对读取过期，再被清理，避免并发进程读到一半被删"）；
   - 原子写：tmp + fsync + `os.replace`；EP rank 0 读写后 `broadcast_object_list` 给全组。

与 Day 2 的 `update_mtp_runtime`（运行期实测反馈）合起来，DSpark 的调度预算是**"启动时 profile 的静态曲线 + 运行时 event 实测的动态校正"**双层结构——这正是 Q&A "based on a profiled hardware throughput curve" 的完整含义。

---

## 六、动手实验

### 实验 1（无硬件替代方案）：从 `tests/worker/` 反推契约

`tests/worker/` 只有两个文件，但信息密度很高：

| 测试 | 反推出的契约 |
|---|---|
| `test_state_cache_ops.py` | **加载真实 V4.1 config（不加载权重）**验证 `select_state_slice_to_store / load_state_from_store_to_prefill_cache`：KVSTORE_GRAIN=128 对齐断言、extra-state 开/关两套初始化、`_deepcopy_unalias_layers` 隔离 config 污染。→ 这两个类方法就是 KV 系统开源边界的**接口规格书** |
| `test_userbuffer.py` | userbuffer（CP 通信的共享缓冲层）的契约（Day 5 展开） |

加上 Day 2/3 已积累的契约，更新后的 **model_runner 输入输出契约表**（今日产出一）：

| 接口 | 输入 | 输出 | 不变量 |
|---|---|---|---|
| `_prepare_input` | seq_metadata_list（driver）/ 无（follower） | `(InputMetadata, total_q, run_cuda_graph, cuda_graph_used, graph_size)` | follower 与 driver 选出**同一张图**（靠 total_q 广播 + 对称捕获）；禁止 CUDA 同步 |
| `execute_main_model` | seq_metadata_list, gpu_cache, metadata_zos | `MainModelOutput(model_output, input_metadata, graph_size)` | 函数内**任何** cuda synchronize 都会被 `forbid_cuda_sync` 变成异常 |
| `execute_sample` | seq_metadata_list, input_metadata, model_output | `SamplerOutput` | 同上；CPU 需要的字段在最终 sync 前 `sampler_epilogue` D2H |
| `capture` / `capture_dflash_model` | model, kv_caches, prepare_input_fn | 无（填充 self.graphs） | 主图必须先于 DFlash 图捕获（assert）；图间共享 memory pool |
| `profile_run` | cache_config | 无（副作用：cache_config.num_gpu_blocks） | FORCE_IMBALANCE_RATIO=1.8 的最坏情况；临时 KV 池不进 torch 记账 |

### 实验 2：手算 capture 集合与 padding 浪费

给定 `max_num_seqs=64, nextn=3, interval=8`（benchmark 的 sweep 参数组合之一）：

```
max_batch_size = 64 × (1+3) = 256
batch_sizes = [1, 2, 4] + range(8, 256, 8) + [256]
            = [1,2,4,8,16,...,248,256]  → 3 + 31 + 1 = 35 张图
total_q = 100（比如 30 个请求、平均接受 2.3 个 draft）→ bisect_left → 选 104
padding = 4 个占位 seq；浪费比例 4/104 ≈ 3.8%
```

再算一个边界：total_q = 5 → 选 8，pad 3（37.5% 浪费但绝对值 3 行）——理解"小 batch 用 2 的幂、大 batch 用固定间隔"的动机。

### 实验 3（有硬件）：profile 一个 decode step

```bash
PROFILE=1 python examples/benchmark.py --decode_num_requests 64
# 或运行期：DEEPINFER_PROFILE_HTTP_ENABLE=1 启动后 curl 端口（base 6070）触发 profile_server.py
```

对照 trace 检查今日调用链的预测：decode step 里应该看到（按时间序）`prepare_v4_decode_metadata` / `gather_input_ids_zos`（TileLang，修正+输入构建）→ `compute_attn_info` 的 index 展开 → FlashMLA / DeepGEMM / DeepEP 序列 → `fused_sample` → dflash 的 draft kernel 序列（`dflash` 前缀）。注意 examples/README 的告诫：trace 内的 step 时间含 profiler 开销（bsz=1 时近 50% 膨胀），**只看 kernel 组成，别拿来算 SPS**。

---

## 七、面试高频问题

**Q1：DeepInfer 的 input_metadata 为什么分出 yoco / dflash / zos 多套？**

A：不是多模型，是 V4.1-Flash 一个模型的四个正交特性各需要一套字段。base/V4（C++ `prepare_prompt_v4/prepare_decode_v4`）覆盖 MLA 压缩 KV、indexer 稀疏注意力和 CED 层共享的通用字段；yoco mixin 服务 SWA+YoCo 截断——C++ 按截断模式二选一产出 compact 或 mask 字段，Python 在层循环的 `n_prefill_layer` 边界用 `get_consumer_view/get_mask_view` 切换视图，同一 forward 内 encoder 层跑 full、consumer 层跑 compact；dflash mixin 服务 DSpark——draft attention 的 fixed-stride 输入（每请求固定 dflash_block_size 个 draft token），在 MTP 阶段由 `build_dflash_input` 从主 metadata 浅拷贝改写；zos 服务异步调度——`prepare_decode_v4_zos` 把 decode 的 metadata 构建整个搬到 GPU（C++ pinned 抽取 + TileLang kernel 写图捕获的设备 buffer），输入长度由上一步 confidence 在 GPU 上修正。

**Q2：CUDA graph 的 capture 集合怎么确定？和 vLLM 有何不同？**

A：`build_graph_batch_sizes(max, interval=8)`：interval 以下的 2 的幂（1,2,4）+ interval 到 max 的固定间隔 + max，且键是 **total_q（query token 数）**而非请求数，max = max_num_seqs × (1+nextn)。用 token 数做键是 DSpark varlen 的必然：total_q 由上一步接受数决定，取值 [num_reqs, num_reqs×(1+nextn)] 任意整数。生产配置（256 seq、nextn 5）约 194 张图，但通过"从大到小捕获 + 全体共享第一张图的 memory pool"把总显存压到约等于最大一张图。vLLM V1 的 `cudagraph_capture_sizes` 思路相同（密集小尺寸 + 稀疏大尺寸），非投机下以 batch size 为键、开 MTP 后同样按 bs×(1+draft) 换算成 token 数。

**Q3：ZOS 下 CUDA graph 里还有 H2D 吗？怎么做到的？**

A：没有。`input_on_device=True` 时 capture 跳过 `recursive_copy_`，且 `input_metadata_source` 本身就是设备张量（`prepare_decode_v4_zos` 在 `args.out=None` 时直接在 CUDA 上分配，之后 `.cuda()` 是无操作）。replay 前，prepare 通过 `get_or_create(args.out)` 拿到图捕获时的设备 buffer，用 TileLang kernel 原地写——`prepare_v4_decode_metadata` 一次 kernel 写 12 类字段，`gather_input_ids_zos` 用 D2D scatter 把上一步的采样/draft token 拼进 input_ids。CPU 全程只做 C++ 的 pinned 抽取。非 ZOS 路径则是经典做法：pinned CPU 源 + 图内捕获的 H2D copy。

**Q4：dflash_cuda_graph_runner 和主 runner 的分工边界？**

A：三个维度都不同。batch 键：主图按 total_q（varlen 验证行数任意），dflash 图按 req_cnt（draft 是 fixed-stride，每请求固定 dflash_block_size 行）；生命周期：主图在 execute_main_model 阶段 replay，dflash 图在 execute_sample 阶段（model.sample 内的 run_drafts）replay，但**共享主图的 memory pool 和捕获 stream** 省显存；输入输出：主图消费 InputMetadata 产出 hidden，dflash 图消费 `MTPDecodeInput`（first_token_positions/last_valid_offset + noise token padding）产出 `MTPPredictInfo`（draft token + confidence），且 `do_sample` 只在 driver worker（CP slave 不采样）。draft 的 KV 索引复用主图的 `input_metadata_device`——draft 读 target 的 KV。

**Q5：KV cache 布局上 DeepInfer 和 vLLM 最大的不同是什么？为什么要这么设计？**

A：vLLM 是 per-layer 同构张量（`[num_blocks, block_size, kv_heads, head_dim]` 每层一份）；DeepInfer 是单 uint8 池 + 层偏移（一个 block 的字节轴上打包所有层的 ica/ca 段）外加独立的每序列 state 槽（SWA ring + fp32 压缩边界）。原因有三：V4.1 的层是异构的（indexer K、C1/C2 压缩 KV、SWA ring 三种形态，还有 share-kv 的 CED 复用），同构张量表达不了；单池让一个 block 的装载/落盘是一段连续字节——`load_segments` 描述符 `[src_block_id, src_begin, src_end, dst_offset]` 直接映射成字节区间拷贝，为 3FS 落盘和 PE→CE 传输优化；分配走 cuMemAlloc 直分，指针稳定（CUDA graph 烧指针）且不进 torch 记账。

**Q6：开源的 cache_engine 到底开到了哪一层？**

A：开到了"单机物理布局 + 落盘切装"层：两池的分配（`allocate_gpu_cache`）、逐层视图切分（`KVCacheManager.process`）、StateBlock 分区（stored/dropped、state_size_to_disk）、以及落盘/回载的接口（`select_state_slice_to_store` / `load_state_from_store_to_prefill_cache`，tests/worker 有规格级测试）。没开的是调用方：engine 的 `do_save_kvcache / pe_*_save_kvcache` 是虚函数，开源树里没有覆盖者——3FS 存储集群、kvcache-store、跨机 SWA 目录、GDR swap-in 那一层全部在内网。也就是说：**字节怎么切开源了，切完发给谁没开源**。

**Q7（开放题）：如果让你实现"prefill 时把 paged KV 历史装载进 gathered buffer"，用现有原语怎么写？**

A：现成答案就是 `BlockLoadPlan`（`csrc/engine/input_metadata/block_load_plan.hpp`）：对每个序列把 `block_table` 上的 [token_begin, token_end) 区间翻成 `BlockLoadSegment` 列表——分页范围 `append_paged_range`（跨块拆段）、环形范围 `append_ring_range`（SWA ring 的回绕）、缺失历史 `append_zero_fill`（src_block_id=-1，零填充），输出 `[N,4]` int32 描述符 + total_k。执行侧一个 gather kernel 按段拷贝即可。这解释了 `load_segments_swa/ca` 字段的形状 `[num_load_segments, 4]`、"每序列最多 3 段"（前缺口 + 主区间 + 后回绕）以及为什么 src=-1 要零填（chunked prefill 的全量 gather 会读未写入区域）。

---

## 八、今日总结（三分钟电梯版）

> DeepInfer 的 worker 层把"batch 准备"做成了四路分派：prompt 走 C++ `prepare_prompt_v4`（YoCo 截断字段二选一），decode 在 ZOS 下走 Python `prepare_decode_v4_zos`——C++ 做 pinned 抽取、TileLang kernel 在 GPU 上把 12 类 metadata 字段直接写进 CUDA graph 捕获的设备 buffer（`input_on_device`，图内零 H2D）。metadata 的核心设计是"紧凑字段 + forward 期展开"：CPU/prepare 只写 `[total_q, 2]` 的 pivots，`compute_attn_info` 的展开 kernel 在图内重算大索引（`get_clean_list`）。KV cache 是"单 uint8 池 + 层偏移 + 每序列 state 槽"：一个 block 打包所有层的 ica/ca（32B 对齐、CED share-kv 复用、cuMemAlloc 直分），state 槽按 `[stored SWA | dropped]` 分区，`state_size_to_disk = 20 × 134 × 528B` 与官方 Q&A 完全闭环（134 = 128 窗口 + 6 回退 = nextn+1）。CUDA graph 以 total_q 为键（varlen 投机的必然），2 的幂 + 间隔 8 的尺寸集合、从大到小捕获共享内存池；DSpark 另有按 req_cnt 键的 dflash 图，共享主图 pool/stream，跑 fixed-stride draft。decode 性能曲线由启动 profiling（EP 组平均、截尾均值、文件缓存）+ 运行期 event 反馈（`update_mtp_runtime`）双层供给调度器。

---

## 九、今日自测题

1. `prepare_decode_v4_zos` 与 `prepare_decode_v4` 的选择由哪个环境变量决定？PROMPT 引擎下用哪个？
2. `kv_ca_index_offset_len` 和 `kv_ca_index` 的形状差多少？为什么前者在 CPU/prepare 侧、后者在图内展开？
3. `get_clean_list()` 返回的字段在 capture 前被做什么？replay 后谁重建它们？
4. `build_graph_batch_sizes(24, 8)` 返回什么？`select(total_q=5)` 选中哪个尺寸、补几个占位 seq？
5. ZOS 下 `input_metadata_source` 在 CPU 还是 GPU？非 ZOS 下 graph replay 时 H2D 发生在哪一步？
6. `state_size_to_disk = 1,415,040 B` 的三个因子分别是什么？134 里的 6 从哪个环境变量的哪个默认规则来？
7. V4.1 的 attention 层有哪几类？full attention 层出现在 config 里会发生什么？
8. `load_segments` 描述符的四个整数是什么？`src_block_id=-1` 表示什么？为什么每序列最多 3 段？
9. dflash 图和主图的 batch 键分别是什么？为什么不同？
10. `DecodePerfCache` 的 cache key 包含哪些字段？为什么并行度必须进 key？

<details>
<summary>参考答案（先自测再展开）</summary>

1. `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`（`model_metadata_utils.py:143`）。PROMPT 引擎该变量默认为 0（`EngineType.default_zero_overhead_scheduling` 不含 PROMPT），且 prompt 恒走 C++ `prepare_prompt_v4`——ZOS 变体只有 decode 版。
2. `[total_q, 2]` vs `[total_q, pad128(max_model_len/N)]`——差约三个数量级。小 pivot 数组是 CUDA graph 的静态输入（CPU/prepare 只写小数组），大索引由 `get_kv_ca_index_prefill/decode` 在 forward（即图内）展开，兼顾图复用与输入面积。
3. 置 None（清空），让图捕获的 `compute_attn_info` 在 replay 时重新生成。清单见 `InputMetadataV4.get_clean_list()`：`kv_ca_index_len / kv_swa_index / kv_ca_index / mlp_relu_schedule_metadata / flash_mla_sched_meta_dict`。
4. `[1, 2, 4, 8, 16, 24]`。`bisect_left` 选 8，补 8−5=3 个占位 seq（占位 metadata 的 draft 信息写到被 store kernel 跳过的 slot 0）。
5. ZOS：GPU（`prepare_decode_v4_zos` 在设备上分配，与 device 副本共享存储，`.cuda()` 是无操作）。非 ZOS：H2D 被捕获在图内——capture 时录下了从 pinned `input_metadata_source` 到 device 的 `recursive_copy_`，replay 时图自己执行这次拷贝，CPU 只需在 replay 前写好 pinned 源。
6. 20 层（swa_store_num_layer，前 n_prefill_layer 层）× 134 token（swa_block_size = window 128 + backtrack 6）× 528 B（512B MXFP8 data + 16B UE8M0 scale）。6 来自 `DEEPINFER_MAX_BACKTRACKING_TOKENS = max(显式值, DEEPINFER_USE_NEXTN_DECODE + 1)`，nextn=5 → 6。
7. SWA（sink_mode=="swa"，住 state ring）、sparse（index_topk>0，indexer K 走分页池）、压缩层（kv_compress_ratio 1/2，主 KV 走分页池），可组合。full 层（ratio=0 且非 swa）触发 `AssertionError`——KVCacheManager 明确不支持。
8. `[src_block_id, src_begin, src_end, dst_offset]`：源块号、源区间（左闭右开）、目标偏移。−1 表示零填充缺失历史（chunked prefill 的 gather 会读未写入区域，零填充保证有限值）。三段 = 前缺口零填 + 主装载区间 + SWA ring 回绕段。
9. 主图按 total_q（token 数，max_num_seqs×(1+nextn)）——varlen 验证的行数由上一步接受数决定；dflash 图按 req_cnt（请求数，max_num_seqs）——draft 是 fixed-stride，每请求固定 dflash_block_size 行，与接受数无关。
10. `{time_unit:ms, model, batch_sizes, data/context/expert_parallel_size}`。同一模型在不同并行拓扑下 EP 组内的负载分布不同，测出的 per-step 毫秒数不可复用；并行度进 key 让拓扑变化自动失效缓存。

</details>

---

## 十、今日产出物

1. **metadata 四套实现对照表**（2.2 节表格，README 指定产出）+ **model_runner 契约表**（实验 1）。
2. **调用链笔记**：2.5 节 SVG（一个 decode step 从 scheduler 输出到 kernel launch，含 ZOS/graph/CP 分支）+ 3.3 节 SVG（KV 两池物理布局与数字闭环）。
3. **差异清单 v1.2**（在 Day 2 的 15 条之上补充 3 条）：

| # | 维度 | vLLM V1 | DeepInfer | 源码证据 |
|---|---|---|---|---|
| 16 | KV 物理组织 | per-layer 同构张量 | 单 uint8 池 + 层偏移 + 每序列 state 槽（cuMemAlloc 直分） | cache_engine.py `KVCacheManager.process` |
| 17 | CUDA graph 键 | batch size（MTP 时换算 token 数） | total_q 直键 + input_on_device 图内零 H2D | cuda_graph_runner.py:74,95、prepare_tensor_input_zos.py |
| 18 | metadata 构建 | Python（`gpu_model_runner` 按注意力后端构建） | C++/GPU 分层：C++ prompt、GPU 侧 decode（ZOS），紧凑字段 + 图内展开 | csrc/engine/input_metadata/、get_clean_list |

4. **手算记录**：实验 2 的 capture 集合与 padding 浪费、3.3 节的 1,415,040 B 推导。

**明日预告（Day 4）**：模型与 kernel——`models/deepseek.py` 与 `v4_modules/` 的 MLA/MoE/YOCO 模块结构，`attention_backend.py` 的 FlashMLA 接入，五个 TileLang kernel 精读（`store_kv_cache / fused_sample / flash_mla_utils / schedule_mtp_varlen / swa_index`），以及 vendor 五库的接入点地图。今日调用链里的 ④ 框（DecoderLayer 内部）将被完全展开。


