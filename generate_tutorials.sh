#!/usr/bin/env bash
#
# generate_tutorials.sh — 按天调用 cannbot 生成 vLLM 推理系统教程（Day 1 ~ Day 56）
#
# 行为：
#   - 每次只向 cannbot 发送「一天」的教程生成任务；
#   - 默认单线程：前一天生成完成（产出 dayNN_*.md）后才发送下一天；
#     需要加速时可用 JOBS=2 等开启并发；
#   - 支持断点续跑：已生成的 Day 自动跳过；
#   - 失败自动重试，多次失败后停止派发新任务并保留现场。
#
# 用法：
#   ./generate_tutorials.sh                 # 从第一个未完成的 Day 开始，直到 Day 56
#   START_DAY=5 END_DAY=10 ./generate_tutorials.sh
#   MAX_RETRIES=5 ./generate_tutorials.sh
#   JOBS=2 ./generate_tutorials.sh          # 双线程并发（默认 1，单线程串行）
#
set -u

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

START_DAY="${START_DAY:-1}"
END_DAY="${END_DAY:-56}"
MAX_RETRIES="${MAX_RETRIES:-3}"
JOBS="${JOBS:-1}"
LOG_DIR="$WORKDIR/logs"
PROGRESS_FILE="$WORKDIR/.tutorial_progress"
mkdir -p "$LOG_DIR"

# cannbot 需要自动写入文件，必须跳过权限确认才能无人值守运行
# 默认模型：cannbot/glm-5.3（可用 `cannbot models` 查看全部，用 CANNBOT_MODEL 覆盖）
CANNBOT_MODEL="${CANNBOT_MODEL:-cannbot/glm-5.3}"
CANNBOT_OPTS=(--dir "$WORKDIR" --dangerously-skip-permissions -m "$CANNBOT_MODEL")

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run.log"
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

run_day() {
  local day="$1"
  local attempt=1
  local prompt
  prompt="$(build_prompt "$day")"

  while (( attempt <= MAX_RETRIES )); do
    log "Day ${day}: 第 ${attempt}/${MAX_RETRIES} 次尝试，发送给 cannbot ..."
    if cannbot run "${CANNBOT_OPTS[@]}" \
        --title "vLLM 教程 Day ${day}" \
        "$prompt" > "$LOG_DIR/day$(printf '%02d' "$day").log" 2>&1; then
      if day_done "$day"; then
        log "Day ${day}: 完成，产出文件已确认。"
        echo "$day" > "$PROGRESS_FILE"
        return 0
      fi
      log "Day ${day}: cannbot 正常退出但未找到 day 产出文件，视为失败。"
    else
      log "Day ${day}: cannbot 退出码非零（详见 logs/day$(printf '%02d' "$day").log）。"
    fi
    attempt=$(( attempt + 1 ))
    sleep 5
  done

  log "Day ${day}: 重试 ${MAX_RETRIES} 次仍失败，停止。修复后可重新运行本脚本续跑。"
  return 1
}

main() {
  log "===== 教程生成开始：Day ${START_DAY} ~ Day ${END_DAY}（并发数 ${JOBS}）====="
  local day
  local failed=0
  for (( day = START_DAY; day <= END_DAY; day++ )); do
    if day_done "$day"; then
      log "Day ${day}: 已存在产出文件，跳过。"
      continue
    fi
    # 并发控制：运行中的任务达到 JOBS 上限时，等待任意一个结束
    while (( $(jobs -rp | wc -l) >= JOBS )); do
      wait -n || failed=1
    done
    if (( failed )); then
      log "检测到失败任务，停止派发新 Day（已在运行的会跑完）。"
      break
    fi
    run_day "$day" &
    sleep 2  # 错峰启动，避免两个 cannbot 同时请求
  done
  # 等所有在跑的任务结束
  while (( $(jobs -rp | wc -l) > 0 )); do
    wait -n || failed=1
  done
  if (( failed )); then
    log "===== 存在失败的 Day，请查看 logs/ 后重跑本脚本续跑 ====="
    exit 1
  fi
  log "===== 全部完成：Day ${START_DAY} ~ Day ${END_DAY} ====="
}

main "$@"
