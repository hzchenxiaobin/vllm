# 第 6 周：项目 A —— vLLM-Ascend / 源码贡献（上）

> **本周定位**：项目 A 是简历级产出（贯穿 W6-W7）。本周完成"选题 → 环境 → 剖析 → 第一轮优化"，下周完成"迭代 → 验证 → PR"。
> **核心优势**：你在昇腾上做过 WeightQuantBatchMatmulV2 的 tiling / 流水线 / 量化优化——这正是 vllm-ascend 最缺的能力，本周的叙事主线就是"跨平台方法论迁移"。
> **主计划**：见根目录 README.md 第 6 周部分（Day 36-42），本文件是它的展开执行手册。

---

## 0. 开工前自检（半小时）

- [ ] W1-W5 产出在手：性能手算公式、V1 架构图、4 份 A4 专题（量化/投机/P-D/分布式）——剖析阶段要反复用到
- [ ] 硬件就绪：Atlas A2（910B）服务器或 DevKit 远程环境，`npu-smi info` 能看到卡
- [ ] 时间预算：每天 2-4 小时；环境问题设硬性止损线（单问题 >1 天未解决 → 求助 issue / 启动降级方案）
- [ ] 心态设定：**宁可小而完整**。一个 shape 特化 + 前后数据 + 分析报告，比一个宏大但没闭环的方向值钱得多

---

## Day 36：选型与调研

**目标**：从"我想贡献"变成"我锁定这一个点，预计 X 天，数据可量化"。

### 36.1 vllm-ascend 仓库速览

先建立地图感，重点目录（以 main 分支实际结构为准）：

| 目录 | 作用 | 与你的关联 |
|---|---|---|
| `vllm_ascend/platform.py` | vLLM Platform 接口的 NPU 实现（配置检查、默认参数、capability 上报） | 了解 NPU 默认参数从哪来 |
| `vllm_ascend/attention/` | Attention 后端：FlashAttention(ACL)、AscendAttention（自研 paged）、torchair 图模式 | 方向 2 / 3 的主战场 |
| `vllm_ascend/quantization/` | w8a8 / w4a16 / FP8 量化实现，最终调到 CANN 侧量化融合算子（quant batch matmul 类） | **与 WeightQuantBatchMatmulV2 直接对口** |
| `vllm_ascend/linear/` | Row/ColumnParallel 线性层 NPU 实现（TP 切分、量化下沉） | 方向 1 的调用链上游 |
| `vllm_ascend/worker/` | model_runner / graph runner（capture、replay） | host 开销分析入口 |
| `vllm_ascend/torchair/` | NPU 整图下沉（torchair graph） | host-bound 优化方向 |
| `vllm_ascend/distributed/` | HCCL 通信封装 | TP 通信开销分析 |
| `tests/`、`ut/` | e2e 与单测 | Day 37 回归验证用 |

### 36.2 找贡献点的五条渠道

1. **issues 列表**：标签过滤 `good first issue` / `performance` / `help wanted`；关键词搜 `slow`、`fallback`、`OOM`、`regression`。优先"有人报、没人领、能复现"的。
2. **官方性能对比数据**：vllm-ascend 博客/文档里 GPU vs NPU 的 benchmark 差距表——差距大的 case 就是天然的优化空间和选题论证。
3. **支持矩阵缺口**：docs 里的 feature 支持表（量化方案 × 模型 × 精度），未勾选项看 issue 是否有人在推进。
4. **自己跑出来的坑**：Day 37 冒烟 + 基线测试中遇到的报错/慢路径（自己复现的问题最可靠）。
5. **近期 merged PR 趋势**：看维护者最近在合并什么，**避开正在大重构的区域**（改了也合不进去）。

### 36.3 候选方向评估（结合背景的优先级排序）

**方向 1：量化 GEMM（w8a8/w4a16）性能优化 —— 强烈推荐**
- 背景：vllm-ascend 的量化线性层最终调 CANN 的量化融合 matmul 算子，decode 阶段 M 极小（≤ 256）、N 窄的 shape 与你做过的场景高度重合
- 你能带来：ASW 蛇形滑窗/L1 全载/tiling 搜优（CalRebalanceBlock 的 balanceRate 思想）可以直接迁移论证
- 切入点：特定 shape 走了慢路径、tiling 参数不适配、反量化没融合、per-channel/group scale 处理开销
- 工作量：中；风险：若瓶颈在 CANN 算子库内部，vllm-ascend 侧只能做调用方式/布局优化（先确认可改的边界！）

**方向 2：Attention 后端（paged attention / MLA）**
- 背景：AscendAttention / DeepSeek MLA 在 NPU 上的 kernel 是热点也是难点
- 切入点：量化 KV 的 attention、特定 seq len 的性能塌陷、MLA absorb 路径的算子适配
- 工作量：大；风险：常需 CANN 算子库配合，PR 周期长。适合作为观察项，若 W6 就有明确小切口再进

**方向 3：torchair 图模式 / host 开销**
- 背景：图下沉失败点、fallback 到 eager、动态 shape 的切图开销
- 对接你"无 Queue 手工流水线"的直觉：host 侧每 step 开销是 NPU 推理的隐性大头
- 工作量：中；纯软改动，迭代快

**方向 4：平台默认参数 / 调度小改进**
- 背景：block_size、max_num_seqs、graph capture bucket 等 NPU 侧默认值调优；prefix caching 相关小 bug
- 工作量：小；影响面可用 benchmark 直接证明，适合作为保底选项或练手第一个 PR

**方向 5：算子补齐 / fallback 消除**
- 方法：跑 e2e 时 grep 日志里的 `fallback` / `not support` / `not implemented`，挑高频的补实现
- 工作量：小-中；好处是链路清晰、容易测

### 36.4 选题评估矩阵（每项 1-5 分，加权求和）

| 维度 | 权重 | 说明 |
|---|---|---|
| 可量化性 | ×3 | 能否拿到干净的前后 benchmark 数据 |
| 影响面 | ×2 | 多少模型/负载受益（主链路 GEMM/Attention > 冷门算子） |
| 经验匹配度 | ×2 | 你昇腾 kernel 功底的复用程度（讲故事的独特性） |
| 工作量可控 | ×2 | 两周内能走完"改-测-PR"闭环 |
| 环境依赖 | ×1 | 是否依赖 CANN 版本/特定卡型（越少越好） |
| 合并概率 | ×1 | 是否命中维护者关注方向、代码区域是否稳定 |

### 36.5 产出：选题备忘录（模板）

```markdown
# 选题备忘录 <日期>
## 主选：<一句话问题描述>
- 证据：issue #xxx / benchmark 差距 X% / 日志现象
- 影响面：<哪些模型/负载/路径>
- 初步假设：<瓶颈可能在哪，为什么>
- 验证方式：<microbenchmark / e2e 指标>
- 预计工作量：<天数>；里程碑：Day38 剖析报告 → Day42 第一轮数据
- 边界确认：<改动落在 vllm-ascend 侧还是 CANN 侧？能改吗？>
- 风险与备选：<若 X 卡住则降级为方向 Y>
```

---

## Day 37：环境打通

**目标**：源码开发环境 + 冒烟通过 + 基线数据落表。**改之前先量化**，否则所有优化都无法自证。

### 37.1 版本匹配（先查官方 Version Compatibility 表，以下仅示例）

| 组件 | 示例版本（以官方文档为准） |
|---|---|
| vllm-ascend | main（开发对齐 main） |
| vllm | 与 vllm-ascend 声明的配套版本 |
| torch / torch-npu | 配套版本（如 2.6.0 系列） |
| CANN Toolkit | 配套版本（如 8.1.RC1） |
| 驱动/固件 | 与 CANN 匹配，`npu-smi info` 确认 |

**原则：一切以 vllm-ascend 文档的版本矩阵为准，不自创组合。** 版本错配是环境问题的第一大来源。

### 37.2 安装步骤（源码开发环境）

```bash
# 0) 基础镜像（推荐直接用官方镜像起容器，省去驱动/CANN 纠错）
# 参考 docs 中的镜像地址（quay.io/ascend/vllm-ascend 或 ascendhub）

# 1) Python 环境
conda create -n vllm-ascend python=3.10 -y && conda activate vllm-ascend
pip install torch==<配套版本> torch-npu==<配套版本> vllm==<配套版本>

# 2) 源码
git clone https://github.com/vllm-project/vllm-ascend.git   # 或 gitee 镜像
cd vllm-ascend && git checkout main
pip install -e .

# 3) 每个新 shell 都要（写进 bashrc）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 4) 冒烟
python -c "import torch, torch_npu; print(torch.npu.is_available())"
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048
curl http://localhost:8000/v1/completions -d '{"model":"...","prompt":"你好","max_tokens":16}'
```

### 37.3 回归入口确认

- 找到与选题相关的最小测试集（`ut/` 单测 + `tests/e2e/` 中对应场景），本地能跑通——这是后续每步可测试的基础
- e2e 若依赖 docker runner，先跑通一条最小链路即可，不追求全绿

### 37.4 基线测试（当天核心产出）

```bash
# 压测（V1 引擎，数据集用 ShareGPT；国内网络注意提前下好模型）
vllm bench serve --model <model> --dataset-name sharegpt \
  --dataset-path <path> --max-concurrency <梯度: 1/4/16/64>

# 辅助：吞吐与时延
vllm bench throughput --model <model> ...
vllm bench latency --model <model> ...
```

**基线记录表**（每个配置一行，存 `week6/baseline.md`）：

| 项 | 值 |
|---|---|
| 模型 / 精度 / 量化 | |
| server 参数（max-model-len, max-num-seqs, block-size…） | |
| 并发梯度 | |
| TTFT p50/p99 | |
| TPOT p50/p99 | |
| 输出吞吐 | |
| HBM 占用 / AICore util（`npu-smi info`） | |
| 环境版本五元组 | |

### 37.5 常见坑速查

- 每个 shell 忘了 `set_env.sh` → 找不到 NPU/算子库
- 驱动-固件-CANN 版本错配 → 各种诡异报错（先对齐版本再排查）
- `ASCEND_RT_VISIBLE_DEVICES=0` 选卡；多卡测试用 `ASCEND_RT_VISIBLE_DEVICES=0,1`
- FlashAttention 相关性能：关注官方 known-issues 中 `TASK_QUEUE_ENABLE` 等环境变量建议（以文档为准，别盲调）
- 调试同步执行用 `ASCEND_LAUNCH_BLOCKING=1`（会拖慢，只用于定位）
- daemon 日志：`~/ascend/log/`（plog）出问题先看这里
- 模型下载：`VLLM_USE_MODELSCOPE=true`

**止损线：环境问题超过 1 天 → 在仓库提 issue / 切换官方镜像方案 / 启动降级方案（见文末）。**

---

## Day 38-40：性能剖析定位瓶颈

**目标**：产出《瓶颈分析报告》——现状数据 → 理论上限 → 优化空间，量化到"差多少、差在哪"。

### 38 三层剖析法（自顶向下，别一上来就掏 msprof）

**第一层：服务级（Day 38 上午）**
- vLLM `/metrics`：TTFT/TPOT 分位数、num_running/num_waiting、preemption 计数、prefix cache hit rate、queue duration
- `npu-smi info`（可 `watch -n 0.5`）：AICore 利用率、HBM 占用——利用率低 ≠ 没 kernel 问题，利用率高 ≠ 健康（可能在空转搬运）
- 先回答：差距在 **TTFT（prefill/排队）** 还是 **TPOT（decode step）**？是 **时延** 问题还是 **吞吐** 问题？——这一步直接决定后面往哪层深挖

**第二层：step 级（Day 38 下午）**
- vLLM profiler 接口抓 chrome trace：
  ```bash
  export VLLM_TORCH_PROFILER_DIR=/tmp/vllm_prof
  # 服务起来后，跑压测中途：
  curl -X POST http://localhost:8000/start_profile
  # 稳定运行 30-60s
  curl -X POST http://localhost:8000/stop_profile
  ```
- trace 里看三件事：① 单个 decode step 的 wall time 构成；② kernel 之间的 gap（host 开销 / 同步等待）；③ 通信（HCCL）占比
- 对比 graph replay vs eager 段落，确认图模式是否真正生效
- 产出：一张"step 时间分解饼图"（kernel / host / 通信 / gap）

**第三层：kernel 级（Day 39）**
- msprof（对标 nsys 的角色）：
  ```bash
  msprof --application="python3 my_step.py" --output=./prof_out
  # 用 MindStudio Insight 打开：timeline + kernel details
  ```
- 提取：**Top-N 热点 kernel 表**（名字/耗时/调用次数）、cube 与 vector 利用率、带宽估计、多核负载是否均衡（对接你的 balanceRate 直觉）
- 对目标 kernel 记录：shape（M/N/K）、dtype、数据布局（ND/NZ）、调用栈（aclnn 接口名）
- **画数据流图**：输入 → L2 → L1 → L0A/L0B → Cube/L0C → 写回，标注每段数据量与理论搬运/计算耗时——用你画惯了的那套达芬奇数据流图

### 39 bound 建模（Day 40，把你的昇腾分界模型写成通用形式）

核心公式组（与 W1 的手算公式闭环）：

```
访存量   Bytes  = 权重读取 + 激活读写 + KV 读写
计算量   FLOPs  = 2 × M × N × K × 迭代次数
机器平衡点 AI* = PeakFLOPS / BW_hbm        # 峰值与带宽以所配 910B SKU 手册为准
算子 AI        = FLOPs / Bytes
AI < AI* → memory-bound（decode 的宿命）；AI > AI* → compute-bound

理论时延 t_theory = max( t_compute, t_mem, t_comm )
效率 η = t_theory / t_measured            # 差距 1-η 就是优化空间
decode 单 token 下界 ≈ (参数字节 + KV 字节) / BW_hbm   # 复用 W1 推导
```

差距分解清单（逐项归因）：带宽利用不足（布局/搬运）｜算力利用不足（指令配比/尾块）｜多核不均（tiling）｜host/launch 开销｜同步与通信串行。

**产出模板：瓶颈分析报告**

```markdown
# 瓶颈分析报告 <日期>
## 1. 现状（数据）
服务级：TTFT/TPOT/吞吐（引用 baseline.md）
step 级：<step 时间分解图>
kernel 级：<Top-N 表 + 目标 kernel 详情>
## 2. 理论上限（推导）
<公式 + 代入数值 + t_theory>
## 3. 差距分解（η = x%，归因：a/b/c）
## 4. 优化空间与假设（按 ROI 排序）
假设1：<改动> 预期收益 <x%>，验证方式 <microbenchmark>
假设2：...
## 5. 风险
<改不动的地方、依赖 CANN 的部分、回归风险>
```

---

## Day 41-42：实施优化（第一轮）

### 41 开发规范

```bash
# fork → clone → 分支开发，小步提交，每个 commit 可运行
git checkout -b perf/<topic>
git commit -s    # 注意按 CONTRIBUTING 要求签署 DCO（sign-off）
# 代码风格：跟随仓库 lint 脚本（ruff），提交前本地过一遍
```

- **先写 microbenchmark**：绕开 vLLM 直接调目标 op，最小复现 + 秒级迭代 + 前后数据——这是你论证效率的尺子
- 每完成一步：microbench → 相关单测 → 小规模 e2e，三层都绿才算数

### 42 优化手段索引（按 bound 类型，全部来自你已有的昇腾经验）

| bound 类型 | 手段（对应你的既有产出） |
|---|---|
| memory-bound | 融合消搬运（quant/matmul/反量化边界再收）、tiling 提升 L2 命中（ASW 蛇形滑窗思想）、NZ 布局优化、小 M 场景 L1 全载（A/权重驻留）、乒乓流水掩盖搬运 |
| compute-bound | Scalar/Vector 指令配比、多精度路径选择（INT8/FP8 快路径）、尾块多核再切分消长尾 |
| 多核不均 | 分块搜优（CalRebalanceBlock 三条件：带宽/算力分界 + balanceRate 剪枝） |
| host-bound | 图模式/减少每 step python 开销、消除不必要 H2D 同步与 `.item()` 类调用 |
| 框架层 | 调用方式/布局/参数（block_size、capture bucket）等不碰 kernel 的软优化 |

**注意**：若瓶颈最终在 CANN 算子库内部而 vllm-ascend 侧改不到，立即降级框架层手段（布局转换避免、shape 归并、调用量裁剪），并把"kernel 内优化建议 + 数据"作为 issue/PR 附带产出——同样是贡献。

### 周末复盘（Day 42）

- [ ] 中途记录整理成时间线：每个决策点"现象 → 假设 → 验证 → 结论"（面试讲素材就是它）
- [ ] 第一轮优化有 microbench + e2e 两级数据了吗？没有的话周日补齐，别带进 W7
- [ ] 自测三问（口头 3 分钟）：
  1. 为什么选这个点而不是别的？（ROI 论证链）
  2. 理论上限怎么算的？实测差多少？差距分解到哪三项？
  3. 如果不让改 kernel，你还有什么手段？

---

## 本周产出物清单

| # | 产出物 | 交付日 |
|---|---|---|
| 1 | 选题备忘录 | Day 36 |
| 2 | 环境与基线记录表（baseline.md） | Day 37 |
| 3 | 瓶颈分析报告（含 step 分解图 + Top-N kernel 表 + bound 推导） | Day 40 |
| 4 | 第一轮优化代码 + microbench/e2e 前后数据 | Day 42 |
| 5 | 决策时间线笔记（面试讲稿素材） | Day 42 |

## 风险与应对

| 风险 | 应对 |
|---|---|
| NPU 环境拿不到/长期不稳定 | 降级方案（主 README）：GPU + Triton 做 paged attention decode kernel + ncu roofline 分析（项目 D），方法论叙事等价 |
| 选题过大做不完 | 砍范围：一个 shape 特化/一个融合/一组默认参数也是完整故事 |
| 瓶颈在 CANN 内部改不动 | 转框架层软优化；kernel 分析报告作为 issue 素材提交 |
| PR 无人 review | 不阻塞：PR + 前后数据本身即作品集闭环（W7 继续跟进） |

## 与第 7 周的衔接

- Day 43-45：完整 benchmark 前后对比（固定配置、多次取均值）、边界 case 与精度验证、整理成 PR 提交
- 本周必须留给 W7 的东西：**干净可复现的基线数据**、**已验证的 microbenchmark 脚本**、**决策时间线**

## 附：面试视角——本周工作的 STAR 骨架（先埋好，W8 打磨）

- **S**：vllm-ascend serving 量化模型时，TPOT/吞吐与理论值（或 GPU 基线）存在 X% 差距
- **T**：定位并缩小差距，形成可合并的贡献
- **A**：三层剖析（metrics → trace → msprof）→ 目标 kernel → 迁移昇腾 bound 分界模型算理论上限 → 差距分解 → tiling/融合/布局优化（明确点出哪些是从 WeightQuantBatchMatmulV2 经验迁移的方法）
- **R**：microbench 提升 X%，e2e TPOT/吞吐提升 Y%，PR #N（含分析报告）
