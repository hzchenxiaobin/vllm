# Week 9：DeepInfer 源码精读 —— DeepSeek 生产级推理框架

> **本周目标**：能完整讲清 DeepInfer 相对 vLLM 做了哪些生产级改造，落到源码级细节：异步调度、Attention DP / balanced CP、MoE EP + EPLB、DSpark 投机解码、SWA bounded replay、batch-invariant 推理。
> **前置**：Week 1–8 已完成 vLLM V1 源码精读（调度链路、KV cache block pool、prefix caching、CUDA graph、量化、投机解码、PD 分离）。本周大量内容与 vLLM V1 对照着看。
> **源码路径约定**：仓库根指 `DeepInfer/`（github.com/deepseek-ai/DeepInfer）。阅读前先 `git log -1` 记录 commit hash。
> **硬件说明**：完整跑起来需要 SM100（B200）或 Ascend 950。没有硬件时以"源码精读 + 文档 + 单测阅读"为主，所有动手任务都标注了替代方案。

---

## 0. 仓库地图（Day 1 前扫一遍）

```
DeepInfer/
├── deepinfer/               # Python 主包
│   ├── engine/              # LLMEngine、参数、profiling、统计
│   ├── entrypoints/         # 请求入口
│   ├── worker/              # worker / model_runner / cache_engine / cuda_graph_runner
│   ├── model_executor/      # 模型定义、layers、tilelang_ops、eplb、并行工具
│   ├── distributed/         # shm_broadcast（共享内存广播）
│   ├── ve/                  # 多模态（vision encoder）处理
│   ├── structure_output/    # 结构化输出（regex / JSON schema / EBNF）
│   └── platform.py          # NVIDIA / Ascend 平台抽象
├── csrc/                    # C++/CUDA 扩展（engine、graph_timer）
├── deep_cluster/            # 硬件拓扑感知
├── examples/                # offline_inference、benchmark、sweep（Pareto 复现）
├── eval/                    # AIME / GPQA / HLE 精度评测
├── tests/                   # PPL 基线回归、分布式、worker 测试
└── doc/inference_deployment_qa.md   # 官方部署 Q&A（本周最重要的文档）
```

与 vLLM 的对应关系（官方明确说"originally built primarily on vLLM"）：

| vLLM V1 | DeepInfer | 关注点 |
|---|---|---|
| `v1/engine/llm_engine.py` | `deepinfer/engine/llm_engine.py` | 引擎循环、异步调度 |
| `v1/worker/gpu_model_runner.py` | `deepinfer/worker/model_runner.py` | batch 准备、metadata 构建 |
| `v1/attention/backends/` | `deepinfer/model_executor/layers/attention_backend.py` | FlashMLA 后端 |
| `v1/spec_decode/` | `deepinfer/model_executor/models/mtp/dflash.py` | DSpark / MTP |
| `v1/core/kv_cache_manager.py` | `deepinfer/worker/cache_engine.py` | KV 管理（注意：完整 KV cache 系统未开源） |

---

## Day 1：总览 —— 设计目标、部署形态与平台抽象

### 阅读清单
1. `README.md` 全文 + feature details 折叠块（每个 feature 对应的环境变量先抄进笔记）。
2. `doc/inference_deployment_qa.md` 全文 —— 官方对三大设计目标的阐述：多厂商硬件、prefill 用普通 8 卡机 / decode 用 64–128 卡超节点、KV cache 保留 72 小时以上。
3. `pyproject.toml`、`setup.py`、`build.sh`：搞清构建产物（`deepinfer/_C`）和 `DEEPINFER_ACCELERATOR` 如何切换平台。
4. `deepinfer/platform.py`：NVIDIA / Ascend 双平台如何抽象。

### 关键问题（能口头回答才算过完）
- 为什么 prefill 和 decode 要用不同规格的机器？（对照 Week 5 PD 分离的内容）
- README 说"完整服务还需要 KV cache 系统、VE/PD 分离、DP 负载均衡"，开源部分缺了什么、留了哪些接口？
- DeepInfer 只支持 DeepSeek-V4.1-Flash 一个模型 —— 从代码结构上找证据（`model_executor/models/` 下只有 deepseek / deepseek_vl）。

### 动手
- 跑一遍构建：`export DEEPINFER_ACCELERATOR=nvidia && ./build.sh`（无硬件则通读 `build.sh` 与 `scripts/build_vendor.sh`，理清 vendor 依赖：FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect）。
- 画一张一页纸架构图：entrypoint → engine → worker → model_runner → kernels，标注每个组件的文件路径。

### 产出
- 架构图 + 一份"DeepInfer vs vLLM V1 差异清单"（本周持续补充）。

---

## Day 2：引擎与请求生命周期 —— 异步调度（zero-overhead batch scheduler）

### 阅读清单
1. `examples/offline_inference.py` + `examples/_launch.py`：入口参数（`--ep` / `--dp` / `--cp`）如何传入引擎。
2. `deepinfer/engine/llm_engine.py`：引擎主循环，找到 schedule 与 execute 的交叠点。
3. `deepinfer/engine/arg_utils.py`、`engine_type.py`：引擎配置如何组织。
4. `deepinfer/sequence.py`、`sampling_params.py`：请求与采样参数的数据结构（注意 `seq_seed`，Day 6 确定性采样要用）。
5. `deepinfer/engine/stats_logger.py`：统计指标上报。

### 关键问题
- "异步调度（默认开启）"具体怎么实现 CPU 调度与 GPU 执行的零开销重叠？对照 vLLM V1 的 async scheduling（Week 3 Day 19），异同点是什么？
- 引擎进程模型是怎样的？对照 vLLM 的 API server / EngineCore 分离，DeepInfer 保留了哪些、砍掉了哪些？
- `shm_broadcast`（`distributed/shm_broadcast.py`）在请求下发中扮演什么角色？

### 动手
- 有硬件：`python examples/offline_inference.py` 跑通默认样例，开 `DEEPINFER_ENABLE_MTP_VARLEN=1` 对比开关前后的输出与速度。
- 无硬件：精读 `tests/engine/` 下的测试，从测试反推引擎接口契约。

### 产出
- 一张"请求生命周期"时序图：从 `SamplingParams` 进来到 token 产出，标出异步调度的 overlap 区间。

---

## Day 3：Worker 层 —— model_runner、cache_engine 与 CUDA graph

### 阅读清单
1. `deepinfer/worker/worker.py`、`base.py`：worker 抽象与设备管理。
2. `deepinfer/worker/model_runner.py`：batch 输入如何准备，`input_metadata/` 下的三种 metadata（`input_metadata.py` / `dflash.py` / `yoco.py` / `prepare_tensor_input_zos.py`）分别服务什么场景。
3. `deepinfer/worker/cache_engine.py`：KV cache 的分配与布局。注意官方说 KV cache 系统未完整开源，理清开源部分到哪一层。
4. `deepinfer/worker/cuda_graph_runner.py` + `dflash_cuda_graph_runner.py`：为什么 DSpark（投机解码）需要单独的 graph runner。
5. `deepinfer/worker/runtime_config.py`、`decode_perf_cache.py`：运行期配置与 decode 性能缓存。

### 关键问题
- `input_metadata` 为什么要分出 dflash / yoco / zos 多套？各自对应模型的什么特性（MTP 投机解码 / SWA / ？）？
- CUDA graph 的 capture 集合怎么确定？和 vLLM 的 `cudagraph_capture_sizes` 思路有何不同？
- `dflash_cuda_graph_runner` 与普通 runner 的分工边界在哪里？

### 动手
- 有硬件：用 `examples/_trace.py` 或 `engine/profile_server.py` 抓一次 profile，定位一个 decode step 的时间分布。
- 无硬件：读 `tests/worker/`，把 model_runner 的输入输出契约整理成表格。

### 产出
- metadata 三套实现的对照表 + "一个 decode step 从 scheduler 输出到 kernel launch"的调用链笔记。

---

## Day 4：模型与 Kernel —— MLA、MoE、TileLang ops

### 阅读清单
1. `deepinfer/model_executor/models/deepseek.py`、`decoder_layer.py`、`v4_modules/`：V4.1 模型结构实现，重点看 MLA attention 与 MoE 层。
2. `deepinfer/model_executor/layers/attention_backend.py`：FlashMLA 后端如何接入。
3. `deepinfer/model_executor/layers/tilelang_ops/`：自研 TileLang kernel 全集 —— 挑 5 个精读：`flash_mla_utils.py`、`store_kv_cache.py`、`fused_sample.py`、`schedule_mtp_varlen.py`、`swa_index.py`。
4. `deepinfer/model_executor/layers/linear.py`、`sampler.py`、`sampler_utils.py`：并行线性层与采样器。
5. `csrc/` 与 `csrc/engine/`：C++ 侧承担了什么（为什么不全用 Python）。

### 关键问题
- FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect 五个 vendor 库各管哪一段？在代码里的接入点分别在哪？
- `tilelang_ops` 里哪些 kernel 是为了 batch-invariant 准备的（`random_sampling.py`、`sum_smaller_probs.py` 值得看）？
- fused_sample 相对 vLLM 的 sampler 做了哪些融合？收益是什么（少几次 kernel launch / 显存往返）？

### 动手
- 挑一个 TileLang kernel（建议 `store_kv_cache.py`），写一段注释版逐行解读，搞清 tile 划分与 grid 映射。
- 有硬件：`python examples/benchmark.py --help` 摸清参数，跑一组小规模 prefill+decode。

### 产出
- 一页"kernel 地图"：一个 decode step 里每个 kernel 的名字、来源（vendor / tilelang / csrc）、输入输出。

---

## Day 5：分布式并行 —— Attention DP / balanced CP、MoE EP + EPLB

### 阅读清单
1. `examples/offline_inference.py --ep 8` / `--dp 8` / `--cp 8` 三种跑法（`examples/README.md`），理解参数语义。
2. `deepinfer/model_executor/parallel_utils/`：并行组如何建立。
3. `deepinfer/model_executor/eplb/`：Expert Parallelism Load Balancing —— 对照 DeepSeek 开源的 EPLB 论文/仓库。
4. `deepinfer/distributed/shm_broadcast.py`：节点内广播机制。
5. `deep_cluster/topology/`、`hardware/`：硬件拓扑感知如何影响并行切分。
6. `doc/inference_deployment_qa.md` 中 EP/CP 相关章节（回头再读一遍，此时应该能看懂细节了）。

### 关键问题
- "Attention DP + balanced CP" 与 vLLM 的 TP/PP/DP 组合有什么本质区别？为什么 MLA 适合 attention DP？
- balanced CP 的 "balanced" 体现在哪？`tilelang_ops/prepare_cp_mirror.py`、`compute_seqlens_k_zos.py` 给了什么线索？
- EPLB 的再平衡触发条件、迁移粒度是什么？

### 动手
- 手算：给定 64 卡 decode 超节点、V4.1 的 expert 数与层数，推一遍 EP 切分与每张卡的 expert 分布；再推 EPLB 热点迁移前后的负载差异。
- 有硬件：`python examples/offline_inference.py --ep 2 --dp 8`，观察 DP rank 间的输出组织方式（`examples/README.md` 提到 PPL 基线按 DP rank 分 key）。

### 产出
- 一张并行策略矩阵图：DP / CP / EP 分别切什么维度、通信原语是什么、对应代码文件。

---

## Day 6：高级特性四连 —— DSpark 投机解码、SWA bounded replay、batch-invariant、结构化输出

### 6a. DSpark + confidence-scheduled verification（上午）
- 阅读：`deepinfer/model_executor/models/mtp/dflash.py`、`tilelang_ops/schedule_mtp_varlen.py`、`tilelang_ops/dflash.py`、`layers/mtp_info_pool.py`、`input_metadata/dflash.py`。
- 环境变量：`DEEPINFER_ENABLE_MTP_VARLEN=1`、`DEEPINFER_USE_NEXTN_DECODE`（draft 最大长度，默认 5）。
- 对照 Week 4 的投机解码：varlen draft 调度和 vLLM 的固定长度 draft 有何区别？"confidence-scheduled verification" 省掉了什么？

### 6b. SWA bounded replay（下午前半）
- 环境变量：`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`、`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`。
- 阅读：`input_metadata/yoco.py`、`tilelang_ops/swa_index.py`、`examples/yoco/`。
- 关键问题：为什么保留的 suffix 必须覆盖 "SWA 窗口 + 最大回退长度"？手算一遍 V4.1 默认配置下 140 是怎么来的。

### 6c. Batch-invariant 与确定性采样（下午后半）
- 环境变量：`DEEPINFER_BATCH_INVARIANT=1`（NVIDIA only）；确定性 = batch-invariant + 关 MTP + 固定 `seq_seed`。
- 阅读：`tilelang_ops/random_sampling.py`、`sum_smaller_probs.py`、`layers/sampler.py`。
- 对照 Thinking Machines 的 batch-invariant 博客：DeepInfer 在哪些 kernel 上保证了 batch 不变性？`tests/` 里找验证测试。

### 6d. 结构化输出（穿插）
- 阅读：`deepinfer/structure_output/`、`examples/_structured_output.py`、`tilelang_ops/structure_output.py`。
- 关键问题：regex / JSON schema / EBNF 三种约束如何编译成 mask？mask 在采样循环的哪一步施加？

### 产出
- 四个特性各写一段"面试官能听懂的三分钟版"：问题是什么、DeepInfer 怎么解、开关是哪个环境变量。

---

## Day 7：性能复现、精度评测与本周复盘

### 阅读与动手
1. `examples/benchmark.py` + `examples/README.md` 的 "Stage decoupling" 一节：prefill/decode 两阶段如何解耦测量，`--prefill_batch_size` / `--decode_num_requests` 语义，为什么 `N` 和 `K` 独立。
2. `examples/sweep/run_sweep.sh` + `plot_pareto.py`：官方 Pareto 图（`doc/images/prefill_pareto.png` / `decode_pareto.png`）的复现流程。有硬件就跑一个子集，无硬件则读懂脚本里每个 sweep 维度。
3. `eval/README.md`、`eval/run_eval.py`：AIME / GPQA / HLE 评测链路。
4. `tests/models/test_offline_inference.py` + `examples/refresh_offline_decode_ppl.py`：PPL 基线回归机制 —— 为什么基线按 platform/workload/DP rank 分 key？多节点刷新为什么要文件锁？
5. `deepinfer/engine/profile_server.py`、`csrc/graph_timer/`：profiling 工具链。

### 复盘（对照 Day 1 的差异清单逐项收尾）
- 能白板讲清：一次 decode step 的完整调用链（engine → worker → model_runner → kernel）。
- 能对比讲清：DeepInfer vs vLLM V1 在调度、并行、投机解码、确定性上的四个核心差异。
- 能回答："如果让你给 DeepInfer 补一个 prefix caching 路由层（未开源部分），接口该怎么设计？"

### 产出（本周最终交付物）
1. 一页架构图（Day 1 版本的迭代终稿）。
2. 差异清单终稿：DeepInfer vs vLLM V1，每条带源码路径。
3. kernel 地图（Day 4）+ 并行矩阵（Day 5）。
4. 四个高级特性的三分钟口述稿（Day 6）。
5. 一份面试 Q&A：从本周"关键问题"里筛 15 条，自问自答。

---

## 附录：环境变量速查

| 变量 | 作用 | 首次出现 |
|---|---|---|
| `DEEPINFER_ACCELERATOR` | 构建/运行平台（nvidia / ascend） | build.sh |
| `DEEPINFER_HF_MODEL_DIR` | 指向本地模型副本 | README |
| `DEEPINFER_ENABLE_MTP_VARLEN` | DSpark varlen 投机解码开关 | Day 6a |
| `DEEPINFER_USE_NEXTN_DECODE` | draft 最大长度（默认 5） | Day 6a |
| `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | SWA bounded replay 模式 | Day 6b |
| `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | 保留 suffix 长度（默认 140） | Day 6b |
| `DEEPINFER_BATCH_INVARIANT` | batch-invariant 推理（NVIDIA） | Day 6c |
| `ASCEND_LAUNCH_BLOCKING` / `ASCEND_SLOG_PRINT_TO_STDOUT` | Ascend 调试 | examples/README |
