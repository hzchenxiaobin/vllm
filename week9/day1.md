# Day 1：DeepInfer 总览 —— 设计目标、部署形态与平台抽象

> **Week 9 · DeepInfer 源码精读 · Day 1**
> 前置知识：Week 1–8 已完成 vLLM V1 源码精读（调度链路 Week 3、投机解码 Week 4、PD 分离 Week 5、量化 Week 6）。今天开始把主线切换到 DeepSeek 生产级推理框架 **DeepInfer**，大量内容与 vLLM V1 **对照着看**。
> 源码基线：`github.com/deepseek-ai/DeepInfer`，commit `9bd9cf2`（2026-10-21，"Bump DeepEP to f6aa38fe and drop the SSH->HTTPS rewrite (#157)"）。阅读任何源码前先 `git log -1` 记录 commit hash，本周所有结论都锚定在这个版本。
> 今日用时：3~4 小时（阅读 2h + 动手画图 1h + 自测 0.5h）

---

## 今日学习目标

| # | 目标 | 验收标准 |
|---|---|---|
| 1 | 讲清 DeepInfer 的三大设计目标 | 脱稿口述 3 分钟：多厂商硬件、prefill/decode 分级部署、KV cache 72h+ 保留，每条能给出一个量化数字 |
| 2 | 画出生产系统架构图 | 一页纸手绘：E/P/D 分离 + 3FS 存储集群 + 4 个负载均衡器 + Recipe，标注并行度（EP2–8 / EP64–128） |
| 3 | 理清开源边界 | 表格列出"生产有 / 开源给了 / 完全没给"，并指出代码里留下的接口线索 |
| 4 | 读懂构建与平台抽象 | 讲清 `DEEPINFER_ACCELERATOR` 从 build 到 runtime 的完整链路（build.sh → build_vendor.sh → setup.py → platform.py） |
| 5 | 启动差异清单 | "DeepInfer vs vLLM V1"差异清单 ≥ 10 条，每条带源码路径（本周持续补充，Day 7 收口） |

今日阅读清单（对应 week9 README）：

1. `DeepInfer/README.md` 全文 + feature details 折叠块（环境变量抄进笔记）
2. `DeepInfer/doc/inference_deployment_qa.md` 全文 —— **本周最重要的文档**，官方对三大设计目标的阐述
3. `pyproject.toml`、`setup.py`、`build.sh` —— 构建产物（`deepinfer/_C`）与平台切换
4. `deepinfer/platform.py` —— NVIDIA / Ascend 双平台抽象

---

## 一、DeepInfer 是什么：定位、血缘与规模

### 1.1 一句话定位

DeepInfer 是 **DeepSeek 自用的大规模生产推理框架**，支撑 DeepSeek-V4.1-Flash 的线上服务。开源 README 第一句就说得很清楚：

> "DeepInfer is the high-performance inference framework that powers DeepSeek's large-scale production services."

生产规模的量化（来自 `doc/inference_deployment_qa.md`）：

- 单集群日服务 **155 万亿（155 trillion）token**
- 每 GPU 小时营收 **38 美元**（谷电价计）
- KV cache 存储集群约 **100 PiB SSD**，逻辑容量约 **14.4 TB/GPU**

### 1.2 血统：站在 vLLM 肩膀上

README 的 Acknowledgments 一节明确写了：

> "DeepInfer was originally built primarily on **vLLM**, with additional inspiration from **SGLang** and **TensorRT-LLM**."

仓库里到处是证据：`setup.py` 第 2 行注释 `# Derived from vLLM; modified by DeepSeek.`，`engine/llm_engine.py` 的 `LLMEngine` docstring 几乎逐句复刻 vLLM 的经典表述（"receives requests from clients and generates texts ... iteration-level scheduling and efficient memory management"）。**但"derived from"不等于"等于"**——本周的主线就是找出 DeepInfer 为生产服务做的那些 vLLM 没有的改造：

| vLLM V1 | DeepInfer | 本周关注点 |
|---|---|---|
| `v1/engine/llm_engine.py`（Python） | `deepinfer/engine/llm_engine.py` + `deepinfer/_C/engine`（**C++ 调度器**） | Day 2 异步调度 |
| `v1/worker/gpu_model_runner.py` | `deepinfer/worker/model_runner.py` | Day 3 metadata 三套实现 |
| `v1/attention/backends/`（FlashAttention/FlashInfer 多后端） | `deepinfer/model_executor/layers/attention_backend.py`（FlashMLA 单后端） | Day 4 |
| `v1/spec_decode/`（ngram/EAGLE/MTP） | `deepinfer/model_executor/models/mtp/dflash.py`（DSpark） | Day 6a |
| `v1/core/kv_cache_manager.py`（完整开源） | `deepinfer/worker/cache_engine.py`（**KV cache 系统未完整开源**） | Day 3 |

### 1.3 支持范围：一个模型、两种硬件

- **模型**：只有 DeepSeek-V4.1-Flash（默认 checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017`）。
- **硬件**：NVIDIA SM100（B200 级）或 Huawei Ascend 950 NPU。
- **环境**：Linux、GCC ≥ 13、Python 3.12、CUDA 13（NVIDIA）/ CANN 9.2（Ascend）。

**"只支持一个模型"的代码证据**（week9 README 关键问题 3）：`deepinfer/model_executor/models/` 下只有三个实现——

```
deepinfer/model_executor/models/
├── deepseek.py          # V4.1 主干
├── deepseek_vl.py       # 多模态封装
├── v4_modules/          # 23 个模块文件：mla_attn / moe / yoco / hc / compress / engram ...
├── mtp/dflash.py        # DSpark draft 模型
└── vision_encoder/      # ViT
```

没有 vLLM 式的 `registry.py` + 数百个模型文件；`model_executor/model_metadata_utils.py`、`model_loader.py` 也只服务于这一个家族。这是**单模型深度优化**的典型取舍：kernel、metadata、CUDA graph 全部针对 V4.1-Flash 的结构（MLA + MoE + YOCO/SWA + CED）手工打磨，不做通用抽象。对照 vLLM 的"通用性优先"，这是两套完全不同的工程哲学——面试时可以作为"通用框架 vs 生产专用框架"的经典对比案例。

### 1.4 Feature 清单与环境变量笔记（README 折叠块抄录）

README 列出的 8 个 feature 与对应开关，先原样抄进笔记（本周 Day 2–6 会逐个深入）：

| Feature | 一句话说明 | 开关 / 环境变量 |
|---|---|---|
| Optimized kernels | FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect + 自研 TileLang kernel | 构建期决定（vendor） |
| DSpark + confidence-scheduled verification | 投机解码，varlen 验证 | `DEEPINFER_ENABLE_MTP_VARLEN=1`；`DEEPINFER_USE_NEXTN_DECODE`（draft 最大长度，默认 5） |
| Decoder SWA bounded replay | 只重放有界 suffix，砍 decoder prefill 计算 | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` + `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` |
| Asynchronous scheduling | CPU 调度与 GPU 执行零开销重叠 | 默认开启（`DEEPINFER_ZERO_OVERHEAD_SCHEDULING`） |
| Batch-invariant inference | batch 组成不同结果仍相同 | `DEEPINFER_BATCH_INVARIANT=1`（NVIDIA only） |
| Deterministic sampling | 相同输入 + 采样参数 → 相同输出 | `DEEPINFER_BATCH_INVARIANT=1` + `DEEPINFER_ENABLE_MTP_VARLEN=0` + 固定 `seq_seed`（NVIDIA only） |
| Structured outputs | regex / JSON schema / EBNF 约束生成 | `SamplingParams(so_args={"schema_type": ..., "schema": ...})` |
| Parallelism | Attention DP / balanced CP、MoE EP + EPLB | `--ep` / `--dp` / `--cp` 启动参数 |

---

## 二、三大设计目标精读（doc/inference_deployment_qa.md）

官方 Q&A 开篇给了推理系统的本质定义，值得背诵：

> "A model defines the boundary of what is possible; serving those capabilities to a massive user base at a reasonable cost is the core problem an **inference framework** has to solve."

V4.1 Flash 推理系统从立项起围绕三个目标构建：

### 目标 1：支持多厂商加速器

在 NVIDIA 和华为 Ascend 上都能跑，且都要**吃满**各自平台的性能。这件事的难点不在"能跑"，而在三处：

1. **kernel 生态**：两边各自有一套算子栈（CUDA/Triton/TileLang vs CANN/AscendC）。DeepInfer 的做法是把核心 kernel 双份维护——`vendor/` 下 DeepGEMM / DeepEP 各有 `-Ascend` 变体 submodule，TileLang kernel 按 `tilelang_ops_backend`（`cuda` / `asc`）选择 `tilelang_ops/cuda/` 或 `tilelang_ops/asc/` 目录。
2. **数值格式**：FP8 scale factor 类型不同（GPU `int32` vs NPU `int16`，见 `platform.py` 的 `fp8_sf_type`）。
3. **通信库**：NCCL vs HCCL，连"限制通信占用多少 SM"这种细节都要各自适配（`DEEPINFER_NCCL_NUM_SMS`）。

代码证据链（今天先建立索引，Day 4/6 展开）：

```
deepinfer/platform.py                     # Platform / PlatformGPU / PlatformNPU
deepinfer/npu_ops/                        # NPU 专属算子补丁
deepinfer/model_executor/layers/tilelang_ops/
    ├── cuda/                             # NVIDIA 版 TileLang kernel
    └── asc/                              # Ascend 版 TileLang kernel
vendor/{DeepGEMM,DeepEP}{,-Ascend}/       # 双份 vendor 库（.gitmodules 六个 submodule）
```

### 目标 2：prefill 用普通 8 卡机，decode 用 64–128 卡超节点

这是**本周最重要的一个设计决策**，官方原话：

> "ordinary 8-GPU machines for prefill, ordinary 64–128-GPU supernodes for decode, with computation, communication, and scheduling designed around the load characteristics of the two stages."

**为什么 prefill 和 decode 要用不同规格的机器？**（week9 README 关键问题 1，对照 Week 5 的 PD 分离）

Week 5 讲过 PD 分离的通用动机：prefill 是 compute-bound、decode 是 memory-bound，混部会互相拖累 TTFT 和 TBT。DeepSeek 在此基础上多走了一步——**两个阶段连并行拓扑和机型都不同**：

| 维度 | Prefill（普通 8 卡机） | Decode（64–128 卡超节点） |
|---|---|---|
| 负载特性 | compute-bound：FLOPs ≈ 2·P·N_tokens | memory-bound：每 step 读全部被激活权重 |
| 并行度 | EP2–8 / DP2–8，**HBM 允许的最小值** | EP64–128 / DP64–128（Wide-EP），**GPU 允许的最大值** |
| 机型要求 | 一台 NVLink 全互联的普通 8 卡机即可 | 需要 64–128 卡的 scale-up 域（超节点） |
| 关注指标 | 大 chunk 的 GEMM 效率、TTFT | 每 step 访存量、TBT/TPS |
| 机型成本 | 便宜，量大 | 昂贵，少量 |

定量直觉（自算，帮助记忆）：

```
Prefill 一个 chunk 的耗时 ≈ 2 · P_active · N_tokens / (util × PeakFLOPS)
    → 追求 FLOPS，卡数只要够放权重（EP2–8 就能放下 MoE 权重 + chunk 激活）

Decode 一个 step 的耗时下界 ≈ 每 step 读的权重字节数 / HBM 带宽
    → MoE 路由 expert 每 step 只激活一部分，但 batch 内的 token 会命中几乎全部 expert
    → Wide-EP：expert 权重切成 64–128 份分摊到所有卡，每卡每 step 读的字节数降到 1/64~1/128
    → 同时 batch 聚合（64+ 并发/rank）让 GEMM 回到高效率区间
```

官方用 EP8 vs EP64 的 benchmark 证明了 Wide-EP 的收益（Q&A 配图 `wide_ep_decode_pareto.png`），并给出 EPLB 的实测不均衡度 `moe_lb`（最慢 rank / 均值）：prefill EP4DP4 = 1.074，decode EP64 = 1.090——即再平衡后最慢 rank 只比均值慢 7–9%。

配套的部署经济学：

- 最小可部署单元是 **1P + 1D**，实践中 P:D 的 GPU 配比约 **1:1**；
- 由于 decode 推荐 EP64 起步，**最小高效部署单元 = 64 卡 prefill + 64 卡 decode**；
- VE（vision encoder）单独部署，**单卡一个实例**即可，推荐用便宜的卡；
- 生产上 **TP、CP 都不用**（Q&A 明说 "TP and CP are not used for now"）——注意代码里保留了 balanced CP 支持（`examples/offline_inference.py --cp 8`、`parallel_utils/cp.py`），这是"代码能力 > 生产配置"的一个例子，Day 5 细讲。

### 目标 3：KV cache 保留 72 小时以上

> "provide a longer window for context reuse, cutting repeated prefill computation on a cache hit, while accounting for the capacity and management overhead that long retention requires."

量化事实（Q&A "Cluster & storage" + "KV cache storage" 两节）：

| 项 | 数值 |
|---|---|
| 存储介质 | 专用 3FS 集群（不是本机内存分层！），约 100 PiB SSD |
| 单台存储服务器 | 20 × 30.73 TB SSD，读 ≥ 40 GB/s，写 ≥ 5 GB/s，容量至少容纳 24h 数据 |
| 逻辑容量 | 约 14.4 TB / 在线 GPU |
| TTL | agent 流量 3 天，chatbot 流量 7 天 |
| 冗余 | 3FS 自身 10+2 纠删码（EC 组容忍 2 节点同时故障）；读并发访问 11 个节点、任意 10 份即可解码返回 |
| 回收 | 3FS GC Worker 按 LRU 淘汰，保持磁盘水位以下；kvcache-store 侧无过期逻辑 |
| 经验值 | chatbot 流量 TTL 从 7 天砍到 3 天，损失 0.5%–1% 命中率 |

为什么 TTL 要 72h+？因为 **agent / 多轮对话的长尾命中**：少量 KV cache 会被读非常多次，丢掉的代价远大于存储成本。这直接决定经济账：命中一次省掉的 prefill FLOPs ≈ 2·P·hit_tokens，而存 1TB SSD 的成本摊到 3~7 天里几乎可以忽略。

**为什么 KV cache 放专用存储集群而不是本机内存分层？**（Day 1 口头能答即可，细节 Day 3/Day 7 展开）

1. **解耦 prefix 亲和与负载均衡**：任何 prefill 机器都能命中任何 prefix，路由不必迁就数据位置（Q&A 原话 "avoiding a conflict between prefix affinity and the load-balancing routing policy"）；
2. **存储节点与计算节点独立扩缩容**；
3. **计算节点故障率比存储节点高一个数量级以上**，KV 放存储侧不受计算节点宕机影响。

缓存路径上的三个环节如何被 overlap 掉（Q&A "How is the cache-related overhead overlapped away?"）：

```
匹配（Matching）：Entrypoint 以 CHECK 角色对本地 kvcache-store 做 hash 链 prefix 匹配，
                 只查 metadata；但命中长度必须在路由前算出 → 无法隐藏
取数（Fetching）：独立进程异步把 KV 从 3FS 拉进主机共享内存
换入（Swap-in）： 引擎进程内的线程把共享内存 → HBM，目标是逐层 H2D 与前向计算重叠；
                 CUDA copy engine 对细粒度 KV 块拷贝性能差，改用 GDR：RDMA NIC 打满
                 PCIe 带宽且不占 SM
```

---

## 三、生产系统全貌：E/P/D 分离 + 专用存储 + 四个负载均衡器

Q&A 的 "Inference system overview" 一节给出系统骨架：**解耦的 VE/Prefill/Decode（E/P/D）架构 + 专用存储节点**。整体形态如下（本文第一张架构图，也是今日产出的"一页纸架构图"的生产系统版）：

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 780" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#475569"/>
    </marker>
    <marker id="arrb" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#2563eb"/>
    </marker>
  </defs>
  <rect width="1060" height="780" fill="#fafbfc"/>
  <text x="530" y="34" text-anchor="middle" font-size="21" font-weight="700" fill="#0f172a">DeepSeek V4.1 Flash 生产推理系统总架构（doc/inference_deployment_qa.md）</text>

  <!-- Client + Recipe -->
  <rect x="60" y="56" width="380" height="58" rx="9" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="250" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#1e1b4b">用户请求：Chat Completions / Responses / Messages API</text>
  <text x="250" y="100" text-anchor="middle" font-size="12" fill="#475569">↓ deepseek-recipe（Rust）：多协议 → 统一会话格式，分段并行编码</text>
  <line x1="250" y1="114" x2="250" y2="138" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Entrypoint -->
  <rect x="60" y="140" width="380" height="58" rx="9" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="250" y="164" text-anchor="middle" font-size="15" font-weight="700" fill="#14532d">Entrypoint（CHECK 角色）</text>
  <text x="250" y="184" text-anchor="middle" font-size="12" fill="#475569">hash 链 prefix 匹配（仅 metadata）；SWA KV 内存目录按 key 分片</text>
  <line x1="250" y1="198" x2="250" y2="222" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Load balancers -->
  <rect x="60" y="224" width="940" height="64" rx="9" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
  <text x="530" y="246" text-anchor="middle" font-size="14" font-weight="700" fill="#78350f">负载均衡层（未开源）</text>
  <text x="530" y="270" text-anchor="middle" font-size="12" fill="#92400e">VE LB（机器随机 + 最少 patch） · Prefill LB（4 档桶 + compute quota） · Decode LB（请求数 + KV 总长双维） · EPLB（冗余 expert + 统计再平衡）</text>
  <line x1="200" y1="288" x2="200" y2="316" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="530" y1="288" x2="530" y2="316" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="860" y1="288" x2="860" y2="316" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>

  <!-- VE column -->
  <rect x="70" y="318" width="240" height="190" rx="10" fill="#fdf4ff" stroke="#a855f7" stroke-width="1.5"/>
  <text x="190" y="342" text-anchor="middle" font-size="15" font-weight="700" fill="#581c87">VE（Vision Encoder）</text>
  <text x="190" y="364" text-anchor="middle" font-size="12" fill="#6b21a8">单卡一实例，用便宜卡</text>
  <text x="190" y="386" text-anchor="middle" font-size="12" fill="#6b21a8">pad → ViT → downsample → aligner</text>
  <text x="190" y="408" text-anchor="middle" font-size="12" fill="#6b21a8">机器粒度随机路由</text>
  <text x="190" y="430" text-anchor="middle" font-size="12" fill="#6b21a8">卡选择：(w/p)×(h/p) patch 数最少</text>
  <text x="190" y="460" text-anchor="middle" font-size="11" fill="#7e22ce">open-source: deepinfer/ve/</text>
  <text x="190" y="478" text-anchor="middle" font-size="11" fill="#7e22ce">multimodal_processor.py</text>

  <!-- Prefill column -->
  <rect x="340" y="318" width="290" height="190" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>
  <text x="485" y="342" text-anchor="middle" font-size="15" font-weight="700" fill="#1e3a8a">Prefill 池（普通 8 卡机 × N）</text>
  <text x="485" y="366" text-anchor="middle" font-size="12" fill="#1e40af">EP2–8 / DP2–8：HBM 允许的最小值</text>
  <text x="485" y="388" text-anchor="middle" font-size="12" fill="#1e40af">chunked prefill：预算解二次成本模型</text>
  <text x="485" y="410" text-anchor="middle" font-size="12" fill="#1e40af">cost(q,c) 求 q，≤32768 token 预算</text>
  <text x="485" y="432" text-anchor="middle" font-size="12" fill="#1e40af">DP 组内预算 all-reduce MIN</text>
  <text x="485" y="460" text-anchor="middle" font-size="11" fill="#2563eb">compute-bound · 追求 GEMM 效率与 TTFT</text>

  <!-- Decode column -->
  <rect x="660" y="318" width="330" height="190" rx="10" fill="#fef2f2" stroke="#dc2626" stroke-width="2"/>
  <text x="825" y="342" text-anchor="middle" font-size="15" font-weight="700" fill="#7f1d1d">Decode 池（64–128 卡超节点）</text>
  <text x="825" y="366" text-anchor="middle" font-size="12" fill="#991b1b">EP64–128 / DP64–128（Wide-EP）</text>
  <text x="825" y="388" text-anchor="middle" font-size="12" fill="#991b1b">EPLB：1 冗余 expert/卡 → EP64 时 7 个/卡</text>
  <text x="825" y="410" text-anchor="middle" font-size="12" fill="#991b1b">DSpark 投机解码 + 异步调度</text>
  <text x="825" y="432" text-anchor="middle" font-size="12" fill="#991b1b">moe_lb = 1.090（1000 步窗口 / 10k 步重排）</text>
  <text x="825" y="460" text-anchor="middle" font-size="11" fill="#dc2626">memory-bound · Wide-EP 砍每 step 访存量</text>

  <!-- P to D transfer -->
  <path d="M630,413 C660,413 660,413 660,413" stroke="#475569" stroke-width="0"/>
  <line x1="630" y1="400" x2="660" y2="400" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arr)"/>
  <text x="645" y="390" text-anchor="middle" font-size="10" fill="#7f1d1d">KV</text>

  <!-- Storage band -->
  <rect x="70" y="540" width="600" height="210" rx="10" fill="#f0fdf4" stroke="#059669" stroke-width="2"/>
  <text x="370" y="566" text-anchor="middle" font-size="15" font-weight="700" fill="#064e3b">KVCache Storage：专用 3FS 集群（~100 PiB SSD，未开源）</text>
  <text x="96" y="594" font-size="12" fill="#065f46">• 单台：20 × 30.73 TB SSD，读 ≥ 40 GB/s，写 ≥ 5 GB/s，容 24h 数据</text>
  <text x="96" y="616" font-size="12" fill="#065f46">• 10+2 纠删码：读并发 11 节点、任意 10 份可解码；GC Worker LRU 保水位</text>
  <text x="96" y="638" font-size="12" fill="#065f46">• TTL：agent 3 天 / chatbot 7 天（3 天 TTL 损失 chatbot 命中率 0.5–1%）</text>
  <text x="96" y="660" font-size="12" fill="#065f46">• 容量：约 14.4 TB 逻辑 / 在线 GPU</text>
  <text x="96" y="682" font-size="12" fill="#065f46">• 命中策略：block KV 定长度，SWA KV 定是否回退一块（≤ L−134）</text>
  <text x="96" y="704" font-size="12" fill="#065f46">• Fetching 异步进程 + Swap-in GDR（RDMA NIC 打满 PCIe，不占 SM）</text>
  <text x="96" y="730" font-size="11" fill="#047857">main KV：per-16 e4m3（无全局 scale 的 NVFP4） · indexer KV：per-32 ue8m0 MXFP4</text>

  <rect x="700" y="540" width="290" height="210" rx="10" fill="#fff7ed" stroke="#ea580c" stroke-width="2"/>
  <text x="845" y="566" text-anchor="middle" font-size="14" font-weight="700" fill="#7c2d12">SWA KV Cache（本机内存）</text>
  <text x="716" y="594" font-size="12" fill="#7c2d12">• StateBlock 固定槽：前 20 层</text>
  <text x="716" y="616" font-size="12" fill="#7c2d12">• 1,415,040 B/seq：</text>
  <text x="726" y="634" font-size="12" fill="#7c2d12">20 层 × 134 token × 528 B (MXFP8)</text>
  <text x="716" y="656" font-size="12" fill="#7c2d12">• 128-token 粒度存储，不进 3FS</text>
  <text x="716" y="678" font-size="12" fill="#7c2d12">• 内存目录跨机命中 → 400G VLAN 回拉</text>
  <text x="716" y="700" font-size="12" fill="#7c2d12">• bounded replay：只重算最近 128 token</text>
  <text x="716" y="730" font-size="11" fill="#c2410c">prefill 不做 KV offload；decode KV 也写盘</text>

  <line x1="370" y1="508" x2="370" y2="540" stroke="#2563eb" stroke-width="2" marker-end="url(#arrb)"/>
  <text x="382" y="528" font-size="11" fill="#2563eb">swap-in / 写入</text>
  <line x1="825" y1="508" x2="845" y2="540" stroke="#2563eb" stroke-width="2" marker-end="url(#arrb)"/>
  <text x="858" y="528" font-size="11" fill="#2563eb">写入</text>
</svg>
```

记住四个数字，面试直接加分：**EP64–128（decode Wide-EP）、1:1（P:D 配比）、64+64（最小高效单元）、72h+（KV TTL）**。

### 3.1 四个负载均衡器（未开源，但 Q&A 给了算法级描述）

| LB | 机制要点 | 关键参数 |
|---|---|---|
| **Prefill LB** | ① 按**总上下文长度**分 4 档桶：<16K / 16–64K / 64–256K / ≥256K，按上一轮各档 compute-quota 份额动态重分配 engine 数，每档有独立批内配额上限；② 跨 DP 组：engine 按剩余配额拉活（**无 prefix 亲和路由**，因为任何 engine 都能命中 3FS 上任何 prefix）；③ DP 组内：请求进当前配额负载最低的 rank；④ 单卡内：FIFO 出队，chunk 大小不是定值，而是从时间预算解出来 | `compute_quota_limit`（ns）；拟合二次成本模型 `cost(q, c)`（q=本 chunk token 数，c=已算 prefix 长度），在 32768-token 预算内取满足剩余时间预算的最大 q；预算再经 DP 组 all-reduce MIN 收紧 |
| **Decode LB** | 在所有 decode 实例的所有 GPU 间均匀路由；同时平衡两个维度：**请求数**（决定每 step batch size）与 **KV 总长**（决定 attention 成本）。启发式：短请求发往"并发低但 KV 总长大"的卡，长请求发往"KV 总长小"的卡 | DualPath 组内调度算法（Q&A 点名） |
| **VE LB** | 图像按**机器粒度**随机路由；机器内选当前处理 patch 数最少的卡，patch 数直接由尺寸算出 `(w/patch)×(h/patch)` | — |
| **EPLB** | 每卡除自身 expert 外持有若干**冗余 expert**（热点副本）；统计最近 N 步各 expert 负载，每 M 步重排；权重经 NVLink `symm_mem` 在组内 shuffle | prefill：4 冗余/卡、100 步窗口、1000 步重排，`moe_lb`=1.074(EP4DP4)/1.045(EP2DP2)；decode：1 冗余/卡（EP64 时取整为 7 个/卡，不含 shared expert）、1000 步窗口、10000 步重排，`moe_lb`=1.090 |

> 💡 交叉验证：EPLB prefill 的 100 步统计窗口正好等于 `deepinfer/environ.py` 里 `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE = EnvVar.int(default=100)` 的默认值——生产参数直接泄露在开源代码的默认值里，这是读源码读出生产配置的典型乐趣。

整体效果：负载均衡后的在线系统约达到**理想均衡离线 benchmark 的 80%**。

### 3.2 KV cache 三层布局

| 内容 | 介质 | 副本 | 管理方 |
|---|---|---|---|
| in-flight block KV + state KV | 推理 GPU 显存 | — | Prefill / Decode 实例 |
| SWA KV cache | **本机** host 内存 | 单副本，不上 3FS | P/D 实例；跨机命中走 400G VLAN 回拉 |
| paged block KV cache | 专用 3FS 存储集群 | 无应用层副本（靠 3FS 自身冗余） | kvcache-store |

SWA KV 的细节（Day 6b 精读，今天记结论）：每个序列占一个固定槽 `StateBlock`，分为 `[stored: SWA] [dropped: YOCO consumer SWA + ca_state/ica_state]` 两个区域；只在 **128-token 整数倍**位置存储；每序列仅 `20 层 × 134 token × 528 B = 1,415,040 B` 落盘（528 B/token 是 MXFP8 SWA KV 格式，前 20 层是 stored layers）。

### 3.3 SWA bounded replay 与命中策略（先记结论）

- **bounded replay**：从位置 s 重算最近 128 个 token，且 query i 只对 `[max(s, i−127), i]` 范围内的 key 做 attention → 重算成本变成**常数**，与上下文长度和层数无关。
- **两个应用点**：encoder 侧（SWA KV miss 时重算命中 prefix 的最后 128 token，block KV 原样复用不重算）；decoder 侧（精确重建需 2560 token × 20 decoder 层，实际只重算 128 token，结果仅用于 decode、不进 prefix cache，**prefill 总计算量近乎减半**）。
- **命中策略**：命中长度由 block KV 决定；最后一个块要不要重算由 SWA KV 决定——SWA miss 则回退一个块。上限 `L − 134`（128 窗口 + 6 token 投机回退）。`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` 默认 140 = 134 + 余量。

### 3.4 低精度计算（FP4 三处）

| 位置 | 格式 |
|---|---|
| main KV | per-16, e4m3 —— 类似 NVFP4 但**无全局 scale** |
| indexer KV | per-32, ue8m0, MXFP4 |
| MoE 路由 expert 权重 | per-32, ue8m0, MXFP4 |

（ue8m0 = 指数位 8、尾数位 0 的纯 2 的幂 scale，MXFP4 微缩放格式。对照 Week 6 量化日的内容。）

### 3.5 其余值得记录的生产细节

- **跨数据中心部署**：技术上可行（V4.1 Flash KV 压缩得好），每 8 GPU 约 400 Gbps 互联带宽；实际部署在相邻数据中心。
- **自动扩缩容**：prefill 按未缓存 token 数定机器数，decode 按运行中请求数；限速且有上下界。decode engine（CE）收到停机通知后进入 **1 分钟 grace period**，把未完成请求连同 KV cache 一次性转发给其他 CE。配额利用 > 0.8 时**禁止缩容**。
- **容错**：重启分钟级；靠 PE/CE 配额冗余 + 请求级重路由，而不是缩短单引擎重启时间。**EP failover**：大 EP 实例全体 GPU 构成单一通信域，单卡故障即实例不可用；重启时平台用当前健康 GPU 重新确定 EP size。
- **kernel 后端**：Mega MoE 是 scale-up 域内通用解（任意 EP/batch 都快于非融合 MoE）；开源 FlashMLA 缺一处手调 SASS，decode kernel 慢 5–10%，端到端差距 ≤ 5%。

---

## 四、开源边界：给了什么、缺了什么、留了什么接口

week9 README 关键问题 2：README 明说完整服务还缺三样——**KV cache 系统、VE/PD 分离、DP 负载均衡**。逐项对照代码盘点：

| 生产组件 | 开源状态 | 代码线索 / 留下的接口 |
|---|---|---|
| 引擎核心（调度、batch、异步执行） | ✅ 完整开源 | `deepinfer/engine/llm_engine.py` + C++ 核心 `csrc/engine/`（Scheduler、BlockManager、SequenceMetadata...） |
| Worker / model_runner / CUDA graph | ✅ | `deepinfer/worker/`（`model_runner.py`、`cuda_graph_runner.py`、`dflash_cuda_graph_runner.py`） |
| 模型与 kernel（MLA/MoE/YOCO/DSpark） | ✅ | `model_executor/models/`、`layers/tilelang_ops/`、`vendor/` |
| 结构化输出 | ✅ | `deepinfer/structure_output/`（compiler/mask/matcher） |
| 多模态预处理 | ✅ | `deepinfer/ve/`（multimodal_processor、processor、schema） |
| VE/P/D 分离 | ⚠️ **引擎功能大部分已实现，未开源部署层** | Q&A 原话 "most required engine functionality is already implemented"；`engine/engine_type.py` 有 `EngineType` 枚举、`LLMEngine.__init__` 接受 `engine_type` 参数 |
| KV cache 系统（3FS 存储 + 路由） | ❌ 未开源 | `worker/cache_engine.py` 只到单机 KV 布局层；`deepinfer/sequence.py` 无跨机 cache 句柄；Day 7 复盘要设计这层接口 |
| 四个负载均衡器（含 EPLB 的调度侧） | ⚠️ EPLB 核心开源，DP 侧 LB 未开源 | `model_executor/eplb/`（policy.py、shuffle.py）+ `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` 等环境变量；Prefill/Decode/VE LB 完全没有对应代码 |
| Online serving（API server） | ❌ | `LLMEngine` docstring 明说 "This release does not include an online-serving wrapper"；只有 `entrypoints/llm.py` 离线入口 |
| deepseek-recipe（token 编解码） | ➡️ 独立仓库 | github.com/deepseek-ai/deepseek-recipe（Rust + Python binding） |

**判断依据的三处源码证据**：

1. `deepinfer/engine/llm_engine.py:140-141` —— "This release does not include an online-serving wrapper."（无在线服务入口）
2. `DeepInfer/README.md:38-42` —— "Required for a complete inference service (not included in this release): A KV cache system; VE/Prefill/Decode disaggregation (most required engine functionality is already implemented); Load balancing across data-parallel replicas..."
3. Q&A 第 21 行 —— KV cache 管理与负载均衡组件"tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced"，细节见知乎架构博客。

这个边界决定了本周的学习策略：**引擎与 kernel 走源码，系统级设计走 Q&A 文档**。两者对照时时刻警惕"文档描述的是生产系统，代码只是它的单机内核"。

---

## 五、仓库地图与构建体系

### 5.1 仓库地图（Day 1 前扫一遍）

```
DeepInfer/                       # 仓库根（commit 9bd9cf2）
├── deepinfer/                   # Python 主包
│   ├── engine/                  # LLMEngine、arg_utils、profile_server、stats_logger
│   ├── entrypoints/             # llm.py（离线入口，无 API server）
│   ├── worker/                  # worker / model_runner / cache_engine / cuda_graph_runner
│   ├── model_executor/          # models、layers、tilelang_ops、eplb、parallel_utils、input_metadata
│   ├── distributed/             # shm_broadcast（共享内存广播）
│   ├── ve/                      # 多模态（vision encoder）处理
│   ├── structure_output/        # 结构化输出（regex / JSON schema / EBNF）
│   ├── npu_ops/                 # Ascend 专属算子
│   └── platform.py              # NVIDIA / Ascend 平台抽象
├── csrc/                        # C++/CUDA 扩展：engine/（调度器）、graph_timer/、utils/
├── deep_cluster/                # 硬件拓扑感知：hardware/（探测）、topology/（NUMA 绑定）
├── vendor/                      # 六个 submodule：FlashMLA、DeepGEMM、DeepEP、DeepSelect（+各 -Ascend 版）
├── examples/                    # offline_inference、benchmark、sweep、yoco
├── eval/                        # AIME / GPQA / HLE 精度评测
├── tests/                       # PPL 基线回归、distributed、engine、worker、ve
└── doc/inference_deployment_qa.md   # 官方部署 Q&A（本周最重要的文档）
```

注意一个细节：**调度器在 C++ 里**。`engine/llm_engine.py` 直接 `from deepinfer._C.engine import Scheduler, BlockManager, FlopsEstimator, RequestOutputPool, SequenceMetadata, ...`——这些符号来自 `csrc/engine/`（pybind_scheduler.cpp、zos.cpp、input_metadata/ 等）。vLLM V1 的调度器是纯 Python（`v1/core/sched/scheduler.py`），DeepInfer 把它下沉到 C++，这是异步调度做到"zero-overhead"的底层前提之一（Day 2 展开）。

### 5.2 构建链路：从 `DEEPINFER_ACCELERATOR` 到 `deepinfer._C`

`build.sh` 只有三步，但每步都有信息量：

```bash
# ① 校验平台变量（只接受 nvidia | ascend，不设直接报错退出）
case "${DEEPINFER_ACCELERATOR:-}" in nvidia|ascend) ;; *) exit 1;; esac

# ② 装平台专属依赖；NVIDIA 还强制把 NCCL 钉到 2.32.3
pip install -r "requirements-${DEEPINFER_ACCELERATOR}.txt"
pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3   # nvidia only

# ③ 先编 vendor 依赖，再编 DeepInfer 本体
bash ./scripts/build_vendor.sh
bash ./scripts/build.sh build
```

**vendor 构建**（`scripts/build_vendor.sh`）——理清"五个 vendor 库"的真实来源：

| 库 | 来源 | NVIDIA | Ascend | 管什么（Day 4 精读） |
|---|---|---|---|---|
| FlashMLA | `vendor/FlashMLA` submodule | ✅ | ✅ | MLA attention（decode/prefill kernel） |
| DeepGEMM | `vendor/DeepGEMM` submodule | ✅ | `vendor/DeepGEMM-Ascend` | FP8/FP4 稠密 GEMM（含 grouped MoE GEMM） |
| DeepEP | `vendor/DeepEP` submodule | ✅ | `vendor/DeepEP-Ascend` | MoE all-to-all dispatch/combine 通信 |
| DeepSelect | `vendor/DeepSelect` submodule | ✅ | ✅ | 选择/路由相关算子 |
| TileKernels + TileLang | **pip 安装**，不是 submodule | `tilelang==0.1.15`、`tile-kernels==2.1.0` | 同左（kernel 走 `tilelang_ops/asc/`） | 自研 kernel 框架与算子库 |

`build_vendor.sh` 的流程：`git submodule update --init` → 逐包 `python setup.py bdist_wheel`（DeepSelect/FlashMLA 通过 `*_BUILD_TARGET_PLATFORM` 环境变量选 CUDA/ASCEND 目标；DeepGEMM 加 `DG_FORCE_BUILD=1`）→ wheel 落到 `build/<platform>-<arch>/vendor-wheels/` → `pip install --no-deps --no-index` 离线装入 `deep_select flash_mla deep_ep deep_gemm`。

**本体构建**（`setup.py`）——产物是单个扩展 `deepinfer._C`：

```python
IS_NPU = accelerator_kind() == "ascend"          # 注意：来自 deep_cluster.hardware！

COMMON_SOURCES = [                                # 两个平台共享的 C++ 源
    "csrc/pybind.cpp",
    "csrc/engine/pybind.cpp",                     # 引擎核心绑定
    "csrc/engine/pybind_sequence.cpp",
    "csrc/engine/pybind_scheduler.cpp",           # ← C++ Scheduler 在这里导出
    "csrc/engine/pybind_output.cpp",
    "csrc/engine/zos.cpp",                        # zero-overhead scheduling 相关
    "csrc/engine/input_metadata/block_load_plan.cpp",
    "csrc/engine/input_metadata/prepare_tensor_input_base.cpp",
    "csrc/engine/input_metadata/prepare_tensor_input_v4.cpp",
    "csrc/engine/prepare_sample.cpp",
    "csrc/graph_timer/pybind.cpp",
]
# NPU 分支：NpuExtension + asc_utils.cpp + graph_timer_asc.cpp，-DWITH_NPU -fopenmp
# GPU 分支：CUDAExtension + cuda_utils.cpp + graph_timer_cuda.cpp，链接 cuda stubs
```

两个值得注意的工程细节：

1. `setup.py` 断言 CXX11 ABI（`torch._C._GLIBCXX_USE_CXX11_ABI`）——与 vendor wheel 的 ABI 对齐，版本错配会直接 assert 失败而不是运行时崩溃。
2. `pyproject.toml` 的 pytest markers 直接反映双平台 CI：`ci`（每条流水线）、`ci_npu`（NPU 流水线）、`gpu`、`benchmark`、`long_context`；还规定 `--import-mode=importlib` 避免 collect 测试文件时触发重型 `__init__` 导入链（会 import `deep_gemm`）。

---

## 六、平台抽象：deepinfer/platform.py 精读

### 6.1 整体结构

`platform.py` 只有 227 行、三个类 + 一个工厂函数，但它是"多厂商硬件"目标的**运行时锚点**：

```python
class PlatformEnum(enum.Enum):
    GPU = enum.auto()
    NPU = enum.auto()

class Platform:                    # 抽象基类
    enum: PlatformEnum             # 平台类型
    tilelang_ops_backend: str      # TileLang kernel 目录名（"cuda" / "asc"）
    distributed_backend: str       # torch.distributed 后端（"nccl" / "hccl"）
    fp8_sf_type: torch.dtype       # FP8 scale factor 类型
    # 方法：pin_host_memory / unpin_host_memory / get_distributed_options

class PlatformGPU(Platform): ...   # tilelang_ops_backend="cuda", distributed_backend="nccl"
class PlatformNPU(Platform): ...   # tilelang_ops_backend="asc",  distributed_backend="hccl"

@functools.cache
def get_platform() -> Platform:
    from deep_cluster.hardware import accelerator_kind
    return PlatformNPU() if accelerator_kind() == "ascend" else PlatformGPU()
```

三个关键设计点：

1. **探测与抽象分离**：`get_platform()` 不看 `DEEPINFER_ACCELERATOR`，而是调 `deep_cluster/hardware/device.py` 的 `accelerator_kind()`——后者靠**文件系统探测**（`ascend.is_available()` / `nvidia.is_available()` 检查 `/dev` 下的设备节点），探测结果 `@functools.cache` 缓存。构建期（`DEEPINFER_ACCELERATOR`）选目标平台，运行期（`accelerator_kind()`）探测实际平台，两条链路分工明确。
2. **`@functools.cache` 单例**：`PlatformNPU.__init__` 里有大量副作用（改环境变量、import torch_npu、解析 ACL API），必须保证只执行一次。
3. **字段即契约**：`tilelang_ops_backend` 直接决定加载 `tilelang_ops/cuda/` 还是 `tilelang_ops/asc/` 的 kernel 目录；`fp8_sf_type`（GPU `int32` / NPU `int16`）贯穿所有 FP8/FP4 量化 kernel。

### 6.2 GPU vs NPU 差异对照表

| 维度 | PlatformGPU | PlatformNPU |
|---|---|---|
| TileLang 后端 | `cuda` | `asc` |
| 分布式后端 | `nccl` | `hccl` |
| FP8 sf 类型 | `torch.int32` | `torch.int16` |
| pin host memory | `cudaHostRegister/Unregister` | `aclrtHostRegisterV2(ACL_HOST_REG_PINNED)` / `aclrtHostUnregister`（ctypes 从 `libascendcl.so` 解析） |
| 分布式 options | `ProcessGroupNCCL.Options`，`max_ctas = min_ctas = DEEPINFER_NCCL_NUM_SMS`（默认 32） | `ProcessGroupHCCL.Options` |
| 特殊处理 | — | 一大段 `__init__` 副作用（见下） |

`PlatformNPU.__init__` 的副作用清单（每一行都是一个真实踩坑记录）：

```python
# 1. 设备可见性对齐：CUDA_VISIBLE_DEVICES 必须与 ASCEND_RT_VISIBLE_DEVICES 一致
#    （torch_npu 的 transfer_to_npu 会把 torch.cuda.* 映射到 torch.npu.*）
# 2. PYTORCH_NPU_ALLOC_CONF=expandable_segments:True   # 减少 NPU 内存碎片
# 3. TASK_QUEUE_ENABLE=0                               # 关闭任务队列
# 4. HCCL_BUFFSIZE=16 + 三个 HCCL 超时/端口环境变量
# 5. TORCH_NPU_DEVICE_CAPABILITY=9.0                   # TODO(NPU): 等 torch_npu 修复的临时 hack
# 6. TILELANG_DEFAULT_TARGET=ascend（setdefault）
#    ↑ 关键：transfer_to_npu 让 torch.cuda.is_available() 在纯 NPU 机器上返回 True，
#      TileLang 的目标探测会误判成 CUDA；必须显式钉死默认 target
# 7. torch.npu.Stream.cuda_stream = property(...)       # 让读 .cuda_stream 的代码拿到 NPU stream
# 8. import deepinfer.npu_ops                           # TODO: 计划移除的补丁
```

`pin_host_memory` 的 NPU 实现还处理了两个"良性"错误码：507910（已注册，视为成功静默）与 507911（未注册，unregister 时静默）；其他失败**每次都告警**——注释写明这是"whole-node degradation canary"（整机退化的金丝雀）。这些错误处理细节是 shm_broadcast 共享内存 DMA 快路径的稳定性基础（Day 2/5 展开）。

对照 vLLM：vLLM 也有 `vllm/platforms/` 平台接口（`PlatformEnum` 同名、`Platform` 基类、`inference_mode`/capability 体系），思想同源（DeepInfer 本就 derived from vLLM）。差异在于 vLLM 支持 CUDA/ROCm/TPU/NPU/XPU/HPU/CPU 等十余种平台、通过 `check_and_update_config` 大规模改写引擎配置；DeepInfer 只保留 GPU/NPU 两平台，接口面小得多——**单模型 + 双平台的专用框架不需要那么宽的抽象**。

### 6.3 deep_cluster：硬件拓扑感知

`deep_cluster/` 是独立于 `deepinfer` 打包的第二个 Python 包（`setup.py` 的 `find_packages` 同时收两者），两个子模块：

- `hardware/`：设备发现——`accelerator_kind()`（cpu/nvidia/ascend）、`physical_device_count()`、`device_numa_node()`（NVIDIA 从 `/sys/bus/pci/devices/<bus_id>/numa_node` 读）、`get_bf16_tflops_limit()`（engine 用它做成本估算）等；
- `topology/bind.py`：NUMA 绑定（`worker/base.py` 的 `_bind_numa` 使用）。

把"硬件探测/拓扑"从平台抽象里再拆一层，是因为**引擎的成本模型和 worker 的 CPU 亲和都要用它**，而不只是 platform 选择（`llm_engine.py` 第一屏就 `from deep_cluster.hardware import get_bf16_tflops_limit`）。

构建 → 探测 → 运行的平台选择全链路（今日第二张图）：

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 460" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="a2" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#475569"/>
    </marker>
  </defs>
  <rect width="1000" height="460" fill="#fafbfc"/>
  <text x="500" y="32" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">DeepInfer 平台选择：构建期 vs 运行期</text>

  <!-- build time -->
  <rect x="40" y="60" width="920" height="150" rx="12" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="500" y="88" text-anchor="middle" font-size="15" font-weight="700" fill="#1e3a8a">构建期（显式环境变量 DEEPINFER_ACCELERATOR=nvidia|ascend）</text>
  <rect x="70" y="104" width="180" height="80" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="160" y="134" text-anchor="middle" font-size="13" font-weight="600" fill="#1e3a8a">build.sh</text>
  <text x="160" y="154" text-anchor="middle" font-size="11" fill="#1e40af">requirements-&lt;accel&gt;.txt</text>
  <text x="160" y="170" text-anchor="middle" font-size="11" fill="#1e40af">NCCL 钉 2.32.3（nvidia）</text>
  <rect x="300" y="104" width="270" height="80" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="435" y="130" text-anchor="middle" font-size="13" font-weight="600" fill="#1e3a8a">scripts/build_vendor.sh</text>
  <text x="435" y="150" text-anchor="middle" font-size="11" fill="#1e40af">nvidia: DeepSelect/FlashMLA/DeepEP/DeepGEMM</text>
  <text x="435" y="166" text-anchor="middle" font-size="11" fill="#1e40af">ascend: DeepEP-Ascend / DeepGEMM-Ascend 替换</text>
  <rect x="620" y="104" width="310" height="80" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="775" y="130" text-anchor="middle" font-size="13" font-weight="600" fill="#1e3a8a">setup.py → deepinfer._C</text>
  <text x="775" y="150" text-anchor="middle" font-size="11" fill="#1e40af">NpuExtension（asc_utils + graph_timer_asc）</text>
  <text x="775" y="166" text-anchor="middle" font-size="11" fill="#1e40af">CUDAExtension（cuda_utils + graph_timer_cuda）</text>
  <line x1="250" y1="144" x2="300" y2="144" stroke="#475569" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="570" y1="144" x2="620" y2="144" stroke="#475569" stroke-width="2" marker-end="url(#a2)"/>

  <!-- runtime -->
  <rect x="40" y="250" width="920" height="180" rx="12" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="500" y="278" text-anchor="middle" font-size="15" font-weight="700" fill="#14532d">运行期（文件系统探测，不看环境变量）</text>
  <rect x="70" y="296" width="300" height="100" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="220" y="322" text-anchor="middle" font-size="13" font-weight="600" fill="#14532d">deep_cluster.hardware.accelerator_kind()</text>
  <text x="220" y="344" text-anchor="middle" font-size="11" fill="#166534">ascend.is_available() 检查 NPU 设备节点</text>
  <text x="220" y="360" text-anchor="middle" font-size="11" fill="#166534">nvidia.is_available() 检查 /dev/nvidia*</text>
  <text x="220" y="376" text-anchor="middle" font-size="11" fill="#166534">→ "ascend" | "nvidia" | "cpu"（@cache）</text>
  <rect x="430" y="296" width="230" height="100" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="545" y="322" text-anchor="middle" font-size="13" font-weight="600" fill="#14532d">platform.get_platform()</text>
  <text x="545" y="344" text-anchor="middle" font-size="11" fill="#166534">PlatformGPU / PlatformNPU</text>
  <text x="545" y="360" text-anchor="middle" font-size="11" fill="#166534">tilelang_ops_backend / fp8_sf_type</text>
  <text x="545" y="376" text-anchor="middle" font-size="11" fill="#166534">pin_host_memory / dist options</text>
  <rect x="720" y="296" width="210" height="100" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="825" y="322" text-anchor="middle" font-size="13" font-weight="600" fill="#14532d">消费方</text>
  <text x="825" y="344" text-anchor="middle" font-size="11" fill="#166534">tilelang_ops/cuda|asc/</text>
  <text x="825" y="360" text-anchor="middle" font-size="11" fill="#166534">量化 kernel 的 sf 类型</text>
  <text x="825" y="376" text-anchor="middle" font-size="11" fill="#166534">NCCL(max_ctas=32) / HCCL</text>
  <line x1="370" y1="346" x2="430" y2="346" stroke="#475569" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="660" y1="346" x2="720" y2="346" stroke="#475569" stroke-width="2" marker-end="url(#a2)"/>
  <line x1="775" y1="184" x2="775" y2="250" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#a2)"/>
  <text x="790" y="222" font-size="11" fill="#64748b">产物在对应平台上运行</text>
</svg>
```

---

## 七、动手实验

### 实验 1（无硬件替代方案）：通读构建脚本，填出 vendor 依赖表

对应 week9 README 的动手任务（无 SM100/Ascend 950 时以源码精读为主）。步骤：

```bash
cd DeepInfer
git log -1 --format="%H %ci %s"        # 记录 commit hash：9bd9cf2 (2026-10-21)
cat build.sh                            # 三步：依赖 → NCCL 钉版 → vendor + 本体
cat scripts/build_vendor.sh             # submodule → wheel → 离线安装
cat .gitmodules                         # 六个 submodule（含两个 -Ascend 变体）
cat requirements.txt                    # 注意 tilelang/tile-kernels 来自 pip
cat setup.py                            # deepinfer._C 的双平台编译分支
```

验收：不看资料默写"五个 vendor 库分别管什么"（答案见 5.2 的表格），并说出**哪两个库不是 submodule 而是 pip 包**（TileKernels、TileLang）。

有硬件时（SM100 / Ascend 950）：`export DEEPINFER_ACCELERATOR=nvidia && ./build.sh`，观察 vendor wheel 产出目录 `build/nvidia-<arch>/vendor-wheels/`。

### 实验 2：画一页纸代码架构图

要求：entrypoint → engine → worker → model_runner → kernels，标注每个组件的文件路径。下图为参考答案（本周 Day 2/3 会逐层深入，此图持续迭代）：

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 700" font-family="'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif">
  <defs>
    <marker id="a3" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#475569"/>
    </marker>
  </defs>
  <rect width="1060" height="700" fill="#fafbfc"/>
  <text x="530" y="32" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">DeepInfer 开源代码一页纸架构（commit 9bd9cf2）</text>

  <!-- entrypoints -->
  <rect x="40" y="56" width="300" height="96" rx="10" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="190" y="82" text-anchor="middle" font-size="14" font-weight="700" fill="#1e1b4b">离线入口（无 online serving）</text>
  <text x="190" y="106" text-anchor="middle" font-size="12" fill="#4338ca" font-family="Consolas, monospace">examples/offline_inference.py</text>
  <text x="190" y="126" text-anchor="middle" font-size="12" fill="#4338ca" font-family="Consolas, monospace">deepinfer/entrypoints/llm.py (LLM)</text>

  <!-- engine -->
  <rect x="40" y="184" width="480" height="240" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>
  <text x="280" y="210" text-anchor="middle" font-size="14" font-weight="700" fill="#1e3a8a">deepinfer/engine/llm_engine.py（LLMEngine）</text>
  <rect x="60" y="224" width="290" height="180" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="205" y="248" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a8a">C++ 引擎核心 deepinfer/_C/engine</text>
  <text x="205" y="272" text-anchor="middle" font-size="11" fill="#1e40af">Scheduler / SchedulerOutputs</text>
  <text x="205" y="290" text-anchor="middle" font-size="11" fill="#1e40af">BlockManager / FlopsEstimator</text>
  <text x="205" y="308" text-anchor="middle" font-size="11" fill="#1e40af">SequenceMetadata(Pool) / RequestOutputPool</text>
  <text x="205" y="326" text-anchor="middle" font-size="11" fill="#1e40af">build_request_outputs / process_MTPInfo</text>
  <text x="205" y="352" text-anchor="middle" font-size="11" fill="#2563eb">源码：csrc/engine/（pybind_scheduler.cpp</text>
  <text x="205" y="368" text-anchor="middle" font-size="11" fill="#2563eb">zos.cpp input_metadata/ prepare_sample.cpp）</text>
  <text x="205" y="392" text-anchor="middle" font-size="11" fill="#7f1d1d">vLLM 对照：v1/core/sched/scheduler.py（Python）</text>
  <rect x="366" y="224" width="140" height="180" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="436" y="250" text-anchor="middle" font-size="12" font-weight="600" fill="#1e3a8a">engine/ 其余</text>
  <text x="436" y="276" text-anchor="middle" font-size="11" fill="#1e40af">arg_utils.py</text>
  <text x="436" y="294" text-anchor="middle" font-size="11" fill="#1e40af">engine_type.py</text>
  <text x="436" y="312" text-anchor="middle" font-size="11" fill="#1e40af">stats_logger.py</text>
  <text x="436" y="330" text-anchor="middle" font-size="11" fill="#1e40af">profile_server.py</text>
  <text x="436" y="348" text-anchor="middle" font-size="11" fill="#1e40af">sequence.py</text>
  <text x="436" y="366" text-anchor="middle" font-size="11" fill="#1e40af">sampling_params.py</text>
  <text x="436" y="390" text-anchor="middle" font-size="11" fill="#1e40af">（seq_seed）</text>
  <line x1="190" y1="152" x2="190" y2="184" stroke="#475569" stroke-width="2" marker-end="url(#a3)"/>

  <!-- shm broadcast -->
  <rect x="560" y="248" width="220" height="112" rx="10" fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>
  <text x="670" y="274" text-anchor="middle" font-size="13" font-weight="700" fill="#713f12">deepinfer/distributed/</text>
  <text x="670" y="296" text-anchor="middle" font-size="12" fill="#a16207" font-family="Consolas, monospace">shm_broadcast.py</text>
  <text x="670" y="318" text-anchor="middle" font-size="11" fill="#a16207">共享内存广播：engine ↔ worker</text>
  <text x="670" y="336" text-anchor="middle" font-size="11" fill="#a16207">pin_host_memory 提供 DMA 快路径</text>
  <line x1="520" y1="300" x2="560" y2="300" stroke="#475569" stroke-width="2" marker-end="url(#a3)"/>

  <!-- worker -->
  <rect x="40" y="456" width="480" height="220" rx="10" fill="#fef2f2" stroke="#dc2626" stroke-width="2"/>
  <text x="280" y="482" text-anchor="middle" font-size="14" font-weight="700" fill="#7f1d1d">deepinfer/worker/</text>
  <rect x="60" y="496" width="200" height="90" rx="8" fill="#fee2e2" stroke="#dc2626"/>
  <text x="160" y="520" text-anchor="middle" font-size="12" font-weight="600" fill="#7f1d1d">worker.py / base.py</text>
  <text x="160" y="542" text-anchor="middle" font-size="11" fill="#991b1b">设备管理、NUMA 绑定</text>
  <text x="160" y="560" text-anchor="middle" font-size="11" fill="#991b1b">（vLLM: v1/worker/）</text>
  <rect x="280" y="496" width="220" height="164" rx="8" fill="#fee2e2" stroke="#dc2626"/>
  <text x="390" y="520" text-anchor="middle" font-size="12" font-weight="600" fill="#7f1d1d">model_runner.py</text>
  <text x="390" y="542" text-anchor="middle" font-size="11" fill="#991b1b">batch 准备 + metadata 构建</text>
  <text x="390" y="560" text-anchor="middle" font-size="11" fill="#991b1b">input_metadata/: base / dflash</text>
  <text x="390" y="576" text-anchor="middle" font-size="11" fill="#991b1b">/ yoco / prepare_zos</text>
  <text x="390" y="598" text-anchor="middle" font-size="11" fill="#991b1b">cuda_graph_runner.py</text>
  <text x="390" y="614" text-anchor="middle" font-size="11" fill="#991b1b">dflash_cuda_graph_runner.py</text>
  <text x="390" y="634" text-anchor="middle" font-size="11" fill="#991b1b">（DSpark 单独 graph runner）</text>
  <rect x="60" y="598" width="200" height="62" rx="8" fill="#fee2e2" stroke="#dc2626"/>
  <text x="160" y="620" text-anchor="middle" font-size="12" font-weight="600" fill="#7f1d1d">cache_engine.py</text>
  <text x="160" y="640" text-anchor="middle" font-size="11" fill="#991b1b">KV 开源层（系统未开源）</text>
  <line x1="280" y1="424" x2="280" y2="456" stroke="#475569" stroke-width="2" marker-end="url(#a3)"/>

  <!-- model executor -->
  <rect x="560" y="56" width="460" height="360" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="2"/>
  <text x="790" y="82" text-anchor="middle" font-size="14" font-weight="700" fill="#14532d">deepinfer/model_executor/</text>
  <rect x="580" y="96" width="200" height="150" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="680" y="118" text-anchor="middle" font-size="12" font-weight="600" fill="#14532d">models/</text>
  <text x="680" y="140" text-anchor="middle" font-size="11" fill="#166534">deepseek.py（唯一主干）</text>
  <text x="680" y="156" text-anchor="middle" font-size="11" fill="#166534">deepseek_vl.py</text>
  <text x="680" y="172" text-anchor="middle" font-size="11" fill="#166534">v4_modules/：mla_attn moe</text>
  <text x="680" y="188" text-anchor="middle" font-size="11" fill="#166534">yoco hc compress engram...</text>
  <text x="680" y="204" text-anchor="middle" font-size="11" fill="#166534">mtp/dflash.py（DSpark）</text>
  <text x="680" y="220" text-anchor="middle" font-size="11" fill="#166534">vision_encoder/</text>
  <rect x="800" y="96" width="200" height="150" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="900" y="118" text-anchor="middle" font-size="12" font-weight="600" fill="#14532d">layers/</text>
  <text x="900" y="140" text-anchor="middle" font-size="11" fill="#166534">attention_backend.py</text>
  <text x="900" y="156" text-anchor="middle" font-size="11" fill="#166534">（FlashMLA 单后端）</text>
  <text x="900" y="172" text-anchor="middle" font-size="11" fill="#166534">linear.py / sampler.py</text>
  <text x="900" y="188" text-anchor="middle" font-size="11" fill="#166534">mtp_info_pool.py</text>
  <text x="900" y="204" text-anchor="middle" font-size="11" fill="#166534">sampler_utils.py</text>
  <text x="900" y="220" text-anchor="middle" font-size="11" fill="#166534">rotary_embedding.py</text>
  <rect x="580" y="262" width="420" height="66" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="790" y="284" text-anchor="middle" font-size="12" font-weight="600" fill="#14532d">layers/tilelang_ops/（自研 TileLang kernel，30+ 文件）</text>
  <text x="790" y="306" text-anchor="middle" font-size="11" fill="#166534">store_kv_cache / fused_sample / schedule_mtp_varlen / swa_index / random_sampling ...</text>
  <rect x="580" y="342" width="200" height="58" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="680" y="364" text-anchor="middle" font-size="12" font-weight="600" fill="#14532d">eplb/</text>
  <text x="680" y="384" text-anchor="middle" font-size="11" fill="#166534">policy.py shuffle.py</text>
  <rect x="800" y="342" width="200" height="58" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="900" y="364" text-anchor="middle" font-size="12" font-weight="600" fill="#14532d">parallel_utils/</text>
  <text x="900" y="384" text-anchor="middle" font-size="11" fill="#166534">parallel_state cp.py userbuffer</text>
  <line x1="790" y1="416" x2="790" y2="446" stroke="#475569" stroke-width="2" marker-end="url(#a3)"/>
  <line x1="790" y1="446" x2="790" y2="456" stroke="#475569" stroke-width="0"/>

  <!-- kernel band -->
  <rect x="560" y="456" width="460" height="220" rx="10" fill="#fdf4ff" stroke="#a855f7" stroke-width="2"/>
  <text x="790" y="482" text-anchor="middle" font-size="14" font-weight="700" fill="#581c87">kernel 层</text>
  <rect x="580" y="496" width="420" height="80" rx="8" fill="#fae8ff" stroke="#a855f7"/>
  <text x="790" y="518" text-anchor="middle" font-size="12" font-weight="600" fill="#581c87">vendor/（submodule → wheel）+ pip</text>
  <text x="790" y="538" text-anchor="middle" font-size="11" fill="#7e22ce">FlashMLA（MLA attn）· DeepGEMM（GEMM）· DeepEP（MoE all-to-all）</text>
  <text x="790" y="556" text-anchor="middle" font-size="11" fill="#7e22ce">DeepSelect（选择算子）· TileKernels/TileLang（pip: tilelang 0.1.15, tile-kernels 2.1.0）</text>
  <rect x="580" y="590" width="200" height="60" rx="8" fill="#fae8ff" stroke="#a855f7"/>
  <text x="680" y="612" text-anchor="middle" font-size="12" font-weight="600" fill="#581c87">tilelang_ops/cuda|asc/</text>
  <text x="680" y="632" text-anchor="middle" font-size="11" fill="#7e22ce">按平台选 kernel 目录</text>
  <rect x="800" y="590" width="200" height="60" rx="8" fill="#fae8ff" stroke="#a855f7"/>
  <text x="900" y="612" text-anchor="middle" font-size="12" font-weight="600" fill="#581c87">csrc/ → deepinfer._C</text>
  <text x="900" y="632" text-anchor="middle" font-size="11" fill="#7e22ce">engine + graph_timer + utils</text>

  <!-- foundation band -->
  <rect x="40" y="56" width="0" height="0" fill="none"/>
  <text x="530" y="694" text-anchor="middle" font-size="12" fill="#64748b">地基：deepinfer/platform.py（GPU/NPU 抽象）+ deep_cluster/（hardware 探测 + topology NUMA 绑定）+ structure_output/ + ve/</text>
</svg>
```

### 实验 3（选做）：从环境变量反推功能版图

`deepinfer/environ.py` 是全部 `DEEPINFER_*` 环境变量的唯一登记处。通读一遍，把变量按"调度 / EPLB / DSpark / YOCO / CUDA graph / SO"分组——今天先建立索引，Day 2–6 每天回来对号入座。几个今天就能对上的默认值：

| 变量 | 默认 | 对应本文内容 |
|---|---|---|
| `DEEPINFER_ACCELERATOR` | `nvidia` | 5.2 构建链路 |
| `DEEPINFER_NCCL_NUM_SMS` | 32 | 6.2 NCCL 通信 SM 上限 |
| `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | 100 | 3.1 EPLB prefill 统计窗口 |
| `DEEPINFER_USE_NEXTN_DECODE` | 5 | 1.4 DSpark draft 长度 |
| `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | `off`（可选 `mask`/`compact`） | 3.3 bounded replay |
| `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL` | 8 | Day 3 伏笔 |
| `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | bool | Day 2 异步调度 |

---

## 八、面试高频问题

**Q1：为什么 DeepSeek 线上 decode 用 EP64–128，prefill 只用 EP2–8？**

A：两个阶段的资源瓶颈不同。prefill 是 compute-bound，大 chunk 矩阵乘追求 FLOPS 利用率，卡数只需够放权重和激活（HBM 允许的最小 EP/DP 即可），普通 8 卡 NVLink 机器就够；decode 是 memory-bound，每 step 要读全部被激活的权重，Wide-EP 把 MoE expert 切到 64–128 卡上、每卡每 step 访存量降到 1/64–1/128，同时 64+ 并发/rank 聚合出大 batch 让 GEMM 回到高效区间，配合 EPLB（moe_lb≈1.09）后 TPS 显著高于 EP8（官方有 EP8 vs EP64 Pareto 对比）。代价是必须用 64–128 卡超节点。经济上 P:D 配比 1:1，最小高效单元 64+64 卡。

**Q2：DeepInfer 相对 vLLM V1 最核心的改造是什么？**

A：四点——① 调度器下沉 C++（`csrc/engine/`，`deepinfer._C.engine.Scheduler`），配合 zero-overhead 异步调度把 CPU 调度彻底移出 GPU 关键路径；② 并行拓扑为 V4.1 定制：attention DP（#DP=#EP）+ balanced CP + MoE Wide-EP + EPLB，生产不用 TP；③ kernel 栈全面自研（FlashMLA/DeepGEMM/DeepEP/TileKernels/DeepSelect + 30+ TileLang kernel）；④ 针对 SWA/YOCO 结构的 bounded replay，decoder prefill 计算近乎减半。再加一条工程判断：它放弃通用性换深度——只支持一个模型、两种硬件。

**Q3：KV cache 为什么要放专用 3FS 存储集群，而不是各机器本地内存分层？**

A：三个理由——① 解耦 prefix 亲和与负载均衡路由：任何 prefill 机器都能命中任何 prefix，路由只需考虑负载（本地分层会造成"数据在哪、请求就得去哪"）；② 存储与计算独立扩缩容；③ 计算节点故障率比存储节点高一个数量级以上，KV 在存储侧不受计算节点宕机影响。配套设计：10+2 EC 冗余、GC Worker LRU、TTL 3/7 天、命中路径三段 overlap（匹配不可隐藏 / fetch 异步 / swap-in 用 GDR 与前向重叠）。SWA KV 是例外——它留本机内存（1.4MB/seq 很小但读频率极高），跨机命中走 400G VLAN 回拉。

**Q4：prefill 的 chunk 大小是怎么定的？**

A：不是固定值。每个 prefill step 有时间预算 `compute_quota_limit`（ns），用拟合的二次成本模型 `cost(q, c)`（q=本 chunk token 数，c=已算 prefix 长度）预测耗时，在 32768-token 预算内解二次方程取满足剩余时间预算的最大 q；预算再经 DP 组 all-reduce MIN 对齐到最慢的 rank。这本质是把 Week 3 学过的 chunked prefill 从"固定 max_num_batched_tokens"升级成"时间驱动"。

**Q5：多厂商硬件支持的难点在哪，DeepInfer 怎么抽象？**

A：难点是 kernel 生态、数值格式、通信库三线作战。DeepInfer 的做法：`platform.py` 用 `Platform` 基类收口平台差异（`tilelang_ops_backend` 选 kernel 目录、`fp8_sf_type` 定量化格式、`pin_host_memory`/`get_distributed_options` 封装内存注册与通信初始化），`deep_cluster.hardware.accelerator_kind()` 用文件系统探测而非环境变量定平台，vendor 库双份维护（DeepGEMM/DeepEP 各有 -Ascend submodule）。NPU 侧最有趣的 hack 是 `transfer_to_npu` 让 `torch.cuda.is_available()` 在 NPU 机器上返回 True，导致 TileLang 目标探测误判，必须 `TILELANG_DEFAULT_TARGET=ascend` 显式钉死。

**Q6（开放题，Day 7 复盘预热）：如果让你给 DeepInfer 补一个 KV cache 路由层，接口怎么设计？**

A：思路提示（今天先立框架，Day 7 给完整答案）——入口在 Entrypoint 做 metadata-only 的 hash 链匹配（对应 CHECK 角色），输出 `(hit_len_blocks, swa_hit: bool)`；引擎侧需要在 `cache_engine.py` 之上暴露 `fetch(prefix_key) -> Future[SharedMemoryHandle]` 与 `swap_in(handles, layers) -> Stream` 两个异步接口，swap-in 逐层与 forward 重叠；还需要一个跨机 SWA 目录（记录 host → sequence 的映射）与 3FS 客户端的读缓存。约束：命中长度必须在路由决策**之前**算出（官方明确说这步无法 overlap）。

---

## 九、今日总结（三分钟电梯版）

> DeepInfer 是 DeepSeek 支撑 V4.1-Flash 线上服务的推理框架（日服务 155T token，$38/GPU-hour），基于 vLLM 深度改造。它围绕三个生产目标构建：**多厂商硬件**（NVIDIA SM100 + Ascend 950，靠 platform.py + 双份 vendor 库）、**prefill/decode 分级部署**（prefill 用 EP2–8 的普通 8 卡机，decode 用 EP64–128 Wide-EP 超节点，P:D=1:1）、**KV cache 72h+ 保留**（100PiB 3FS 集群，TTL 3/7 天，14.4TB/GPU）。系统形态是 E/P/D 分离 + 专用存储 + 四个负载均衡器，其中引擎与 kernel 完整开源（C++ 调度器、FlashMLA/DeepGEMM/DeepEP/TileKernels/DeepSelect、30+ TileLang kernel），KV cache 系统和三个 DP 侧负载均衡器未开源。与 vLLM 的本质分歧是"专用 vs 通用"：单模型、双平台、生产参数硬编码进代码默认值。

---

## 十、今日自测题

1. DeepInfer 的三大设计目标是什么？每个目标给一个量化数字。
2. decode 生产并行度是多少？为什么是"GPU 允许的最大 EP/DP"？prefill 为什么反过来取最小值？
3. 最小高效部署单元是多少卡？P:D 配比是多少？
4. SWA KV cache 每序列存多少字节？由哪三个数相乘得到？为什么不上 3FS？
5. KV 命中路径的三段是什么？哪一段官方明确说无法 overlap？
6. 命中长度上限公式 `L − 134` 中，134 由哪两部分组成？
7. `DEEPINFER_ACCELERATOR` 在构建期和运行期分别由什么机制决定平台？`get_platform()` 为什么不读这个环境变量？
8. 五个 vendor 库分别管什么？其中哪些是 git submodule、哪些来自 pip？

<details>
<summary>参考答案（先自测再展开）</summary>

1. 多厂商硬件（SM100 + Ascend 950）；prefill 普通 8 卡机 / decode 64–128 卡超节点（EP64–128）；KV cache 保留 72h+（TTL：agent 3 天 / chatbot 7 天，约 100 PiB、14.4TB 逻辑/GPU）。规模数字：155T token/天，$38/GPU-hour。
2. EP64–128 / DP64–128（Wide-EP）。decode 是 memory-bound，切到 64–128 卡后每卡每 step 读的 expert 权重降到 1/64–1/128，且聚合 batch 让 GEMM 高效；prefill 是 compute-bound，卡数只需 HBM 放得下（EP2–8），多了只是浪费机器。
3. 64 卡 prefill + 64 卡 decode；P:D ≈ 1:1（1P+1D 即可部署，但 decode 推荐 EP64 起步）。
4. 1,415,040 B = 20 层 × 134 token × 528 B（MXFP8 格式，stored layers 为前 20 层）。不上 3FS 因为它小（~1.4MB/seq）但读取频率极高，放本机内存 + 内存目录 + 400G VLAN 跨机回拉更划算；且它只在 128-token 整数倍位置存储，与 block KV 的生命周期不同。
5. 匹配（Entrypoint hash 链前缀匹配，仅 metadata）→ 取数（独立进程异步拉 3FS → 主机共享内存）→ 换入（引擎内线程共享内存 → HBM，GDR 走 RDMA NIC）。匹配必须发生在路由决策之前，无法 overlap。
6. 128-token 重算窗口 + 6 token 投机回退余量。
7. 构建期由 `DEEPINFER_ACCELERATOR` 显式指定（build.sh 校验、setup.py 选 NpuExtension/CUDAExtension、build_vendor.sh 选 vendor 组合）；运行期由 `deep_cluster.hardware.accelerator_kind()` 文件系统探测（检查 /dev 设备节点）。`get_platform()` 不读环境变量是为了以真实硬件为准，避免环境变量与实际平台不符时静默跑错 kernel 目录。
8. FlashMLA=MLA attention，DeepGEMM=FP8/FP4 GEMM（含 grouped MoE），DeepEP=MoE all-to-all 通信，DeepSelect=选择/路由算子，TileKernels+TileLang=自研 kernel 框架与算子库。前四个（DeepGEMM/DeepEP 各含 -Ascend 变体，共六个）是 submodule；TileKernels 和 TileLang 来自 pip（tile-kernels 2.1.0 / tilelang 0.1.15）。

</details>

---

## 十一、今日产出物

1. **两张架构图**：本文第三节 SVG（生产系统总架构）与第七节 SVG（开源代码一页纸架构）——今日产出的核心，Day 7 复盘时迭代成终稿。
2. **"DeepInfer vs vLLM V1"差异清单 v1.0**（本周持续补充，Day 7 收口）：

| # | 维度 | vLLM V1 | DeepInfer | 源码证据 |
|---|---|---|---|---|
| 1 | 调度器实现 | Python（`v1/core/sched/scheduler.py`） | **C++**（pybind 导出） | `csrc/engine/pybind_scheduler.cpp`、`llm_engine.py` 的 `from deepinfer._C.engine import Scheduler` |
| 2 | 服务形态 | API server + EngineCore（ZMQ 分离） | 仅离线入口，无 online wrapper | `engine/llm_engine.py:140` docstring |
| 3 | 模型支持 | registry + 数百模型 | 仅 V4.1-Flash 一个家族 | `model_executor/models/` 只有 deepseek / deepseek_vl |
| 4 | Attention 后端 | FlashAttention/FlashInfer/Triton 多后端 | FlashMLA 单后端 | `layers/attention_backend.py`、`vendor/FlashMLA` |
| 5 | KV cache | kv_cache_manager + prefix caching 完整开源 | 单机 `cache_engine.py`；分布式 KV 系统（3FS）未开源 | `README.md:38-42`、Q&A |
| 6 | 并行策略 | TP 为主，DP/EP/PP 可选 | Attention DP + balanced CP + MoE EP + EPLB，生产不用 TP | Q&A "TP and CP are not used for now"、`parallel_utils/`、`eplb/` |
| 7 | 投机解码 | ngram / EAGLE / MTP | DSpark + confidence-scheduled varlen 验证 | `models/mtp/dflash.py`、`tilelang_ops/schedule_mtp_varlen.py` |
| 8 | 平台抽象 | 十余平台 + check_and_update_config | 仅 GPU/NPU，接口极小 | `platform.py`（227 行） |
| 9 | tokenizer/协议 | 内置（transformers） | 外置 deepseek-recipe（Rust，独立仓库） | Q&A "DeepSeek Recipe" 节 |
| 10 | kernel 生态 | 内置 CUDA/Triton/CK | vendor submodule + pip TileKernels + 30+ TileLang 自研 | `vendor/`、`layers/tilelang_ops/` |
| 11 | 量化 | FP8（GPTQ/AWQ 等算法量化） | FP4 三处：main KV per-16 e4m3、indexer KV per-32 ue8m0、MoE expert per-32 ue8m0 | Q&A "Low-precision computation" |
| 12 | chunked prefill | 固定 max_num_batched_tokens | 时间预算 + 二次成本模型解 q | Q&A Prefill LB；（开源引擎侧对应 `DEEPINFER_PREFILL_MAX_SUM_SEQLEN_K`） |

3. **环境变量笔记**：1.4 节表格 + 7 节实验 3 的分组索引。

**明日预告（Day 2）**：引擎与请求生命周期——异步调度如何做到 zero-overhead（GPU 跑 `forward(N)→sample(N)` 时 CPU 并发跑 `schedule(N+1)→launch_forward(N+1)→post-process(N−1)→launch_sample(N+1)`），`shm_broadcast` 在请求下发中的角色，以及引擎进程模型与 vLLM API server/EngineCore 分离的对照。


