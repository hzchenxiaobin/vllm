#!/usr/bin/env bash
#
# generate_tutorials.sh — 按天调用 cannbot 生成 vLLM 推理系统教程（Day 1 ~ Day 56）
#
# 行为：
#   - 每次只向 cannbot 发送「一天」的教程生成任务；
#   - 默认单线程：前一天生成完成（产出 dayNN_*.md）后才发送下一天；
#     需要加速时可用 JOBS=2 等开启并发；
#   - 支持断点续跑：已生成的 Day 自动跳过；
#   - 不重试：某天失败则记录并继续生成下一天，失败的 Day 在结束时汇总，
#     重新运行本脚本即可补跑；
#   - 每天之间冷却 COOLDOWN 秒（默认 600，即 10 分钟）：缓解网关限流；
#   - 模型思考强度 CANNBOT_VARIANT（默认 low）：GLM-5.3 默认思考过长，
#     复盘类任务会烧 3 万+ reasoning token 撞输出上限（finish=length）导致
#     空退，low 可避免该失败模式；
#   - 末尾自动补跑：第一轮结束后，对缺失的 Day 最多再补跑 MAX_PASSES-1 轮
#     （默认共 3 轮），轮间同样冷却。
#
# 用法：
#   ./generate_tutorials.sh                 # 从第一个未完成的 Day 开始，直到 Day 56
#   START_DAY=5 END_DAY=10 ./generate_tutorials.sh
#   JOBS=2 ./generate_tutorials.sh          # 双线程并发（默认 1，单线程串行）
#   COOLDOWN=600 ./generate_tutorials.sh    # 自定义冷却秒数（默认 600，即 10 分钟）
#   CANNBOT_VARIANT=medium ./generate_tutorials.sh   # 调整思考强度（默认 low）
#   MAX_PASSES=5 ./generate_tutorials.sh    # 最多补跑轮数（默认 3）
#
set -u

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

START_DAY="${START_DAY:-1}"
END_DAY="${END_DAY:-56}"
JOBS="${JOBS:-1}"
COOLDOWN="${COOLDOWN:-600}"
MAX_PASSES="${MAX_PASSES:-3}"
LOG_DIR="$WORKDIR/logs"
PROGRESS_FILE="$WORKDIR/.tutorial_progress"
mkdir -p "$LOG_DIR"

# cannbot 需要自动写入文件，必须跳过权限确认才能无人值守运行
# 默认模型：cannbot/glm-5.3（可用 `cannbot models` 查看全部，用 CANNBOT_MODEL 覆盖）
CANNBOT_MODEL="${CANNBOT_MODEL:-cannbot/glm-5.3}"
CANNBOT_VARIANT="${CANNBOT_VARIANT:-low}"
CANNBOT_OPTS=(--dir "$WORKDIR" --dangerously-skip-permissions -m "$CANNBOT_MODEL" --variant "$CANNBOT_VARIANT")

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run.log" >&2
}

# 判断某一天是否已有产出文件（day01_*.md / day1_*.md 等，任意子目录）
day_done() {
  local day="$1"
  local padded
  padded="$(printf '%02d' "$day")"
  find "$WORKDIR" -type f \
    \( -iname "day${padded}_*.md" -o -iname "day${day}_*.md" \) \
    ! -path '*/.git/*' | grep -q .
}

# 找出该 Day 产出文件的存放目录：按周组织（week1..week8），目录不存在则用根目录
day_dir() {
  local day="$1"
  local week=$(( (day - 1) / 7 + 1 ))
  if [[ -d "$WORKDIR/week${week}" ]]; then
    echo "week${week}"
  else
    echo "."
  fi

}

# 构造发给 cannbot 的当天任务提示
build_prompt() {
  local day="$1"
  local padded
  padded="$(printf '%02d' "$day")"
  local outdir
  outdir="$(day_dir "$day")"
  cat <<EOF
请阅读当前目录下的 prompt.md 和 README.md。

README.md 中是 Day 1 ~ Day 56 的学习计划，prompt.md 中是教程的格式与深度要求（一日一文、章节结构、SVG 图、vLLM V1 主线、代码与源码调用链、面试问题等），请严格遵守。

本次任务：只生成 Day ${day} 这一天的教程，严格对应 README.md 中 Day ${day} 的主题、学习目标、实验和产出，不要生成其他天的内容。

要求：
- 输出一个 Markdown 文件到 ${outdir}/ 目录下，文件名形如 day${padded}_<主题英文小写下划线>.md。
- SVG 图保存到 ${outdir}/assets/ 并引用，或直接内嵌在 Markdown 中。
- 写作前可以浏览已存在的 day*.md 文件（如果存在），在内容上主动与前几天衔接（例如「回顾 Day X」），但不要修改已有文件。
- 完成后请确认文件已写入，并简要说明文件路径和主要内容。
EOF
}

# 生成某一天的教程：只尝试一次，失败返回 1（由调用方决定是否继续）
run_day() {
  local day="$1"
  local prompt
  prompt="$(build_prompt "$day")"

  log "Day ${day}: 发送给 cannbot ..."
  if cannbot run "${CANNBOT_OPTS[@]}" \
      --title "vLLM 教程 Day ${day}" \
      "$prompt" > "$LOG_DIR/day$(printf '%02d' "$day").log" 2>&1; then
    if day_done "$day"; then
      log "Day ${day}: 完成，产出文件已确认。"
      echo "$day" > "$PROGRESS_FILE"
      return 0
    fi
    log "Day ${day}: cannbot 正常退出但未找到 day 产出文件，视为失败（详见 logs/day$(printf '%02d' "$day").log）。"
  else
    log "Day ${day}: cannbot 退出码非零（详见 logs/day$(printf '%02d' "$day").log）。"
  fi
  return 1
}

# 单轮扫描：对范围内所有缺失的 Day 各生成一次，返回缺失列表（echo，逗号分隔）
run_pass() {
  local day
  local launched=0
  for (( day = START_DAY; day <= END_DAY; day++ )); do
    if day_done "$day"; then
      log "Day ${day}: 已存在产出文件，跳过。"
      continue
    fi
    # 并发控制：运行中的任务达到 JOBS 上限时，等待任意一个结束
    while (( $(jobs -rp | wc -l) >= JOBS )); do
      wait -n
    done
    # 冷却：仅在本轮已经跑过任务后生效，避免紧跟大生成触发网关限流
    if (( launched )) && (( COOLDOWN > 0 )); then
      log "冷却 ${COOLDOWN}s 后生成 Day ${day} ..."
      sleep "$COOLDOWN"
    fi
    # 失败不中断：run_day 返回非零也继续派发下一天
    run_day "$day" &
    launched=1
    sleep 2  # 错峰启动，避免两个 cannbot 同时请求
  done
  wait

  # 汇总本轮结束后仍缺失的 Day
  local missing=()
  for (( day = START_DAY; day <= END_DAY; day++ )); do
    day_done "$day" || missing+=("$day")
  done
  echo "${missing[*]}"
}

main() {
  log "===== 教程生成开始：Day ${START_DAY} ~ Day ${END_DAY}（并发 ${JOBS}，冷却 ${COOLDOWN}s，variant ${CANNBOT_VARIANT}，最多 ${MAX_PASSES} 轮）====="
  local pass=1
  local missing=""
  while (( pass <= MAX_PASSES )); do
    log "----- 第 ${pass}/${MAX_PASSES} 轮 -----"
    missing="$(run_pass)"
    if [[ -z "$missing" ]]; then
      log "===== 全部完成：Day ${START_DAY} ~ Day ${END_DAY} ====="
      return 0
    fi
    log "第 ${pass} 轮结束，以下 Day 未产出：${missing}"
    pass=$(( pass + 1 ))
    if (( pass <= MAX_PASSES )) && (( COOLDOWN > 0 )); then
      log "冷却 ${COOLDOWN}s 后开始补跑轮 ..."
      sleep "$COOLDOWN"
    fi
  done
  log "===== ${MAX_PASSES} 轮后仍缺失：${missing}（重跑本脚本可继续补）====="
  return 1
}

main "$@"
