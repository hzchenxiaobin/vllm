"""Landing page (public/index.html) builder — a designed, sidebar-less home page."""

import html
from pathlib import Path

from .common import GITHUB_REPO_URL
from .weeks import WEEK_GOALS

PHASES = [
    ("阶段一", "地基与源码", "第一性原理 → vLLM V1 全链路源码精读", range(1, 4)),
    ("阶段二", "进阶专题", "量化 / 投机解码 / P·D 分离 / 分布式", range(4, 6)),
    ("阶段三", "项目与冲刺", "开源贡献 + 消融实验 + 面试冲刺", range(6, 9)),
]

_RESOURCES = [
    ("📄", "完整学习计划", "八周按天打卡总计划（README）", "plan.html"),
    ("🤖", "教程生成 Prompt", "逐天教程的生成提示词", "prompt.html"),
    ("💻", "GitHub 仓库", "本站的全部源码与 Markdown 原文", GITHUB_REPO_URL),
]


def _build_week_cards_html(weeks: list) -> str:
    by_num = {w["num"]: w for w in weeks}
    out = []
    for phase_no, phase_name, phase_desc, nums in PHASES:
        cards = []
        for num in nums:
            week = by_num.get(num)
            if week is None:
                continue
            title = html.escape(week["title"])
            goal = html.escape(week.get("goal") or WEEK_GOALS.get(num, ""))
            cards.append(f'''        <a class="week-card" href="week{num}/index.html">
          <div class="week-card-top">
            <span class="week-card-badge">W{num}</span>
            <span class="week-card-arrow">→</span>
          </div>
          <div class="week-card-title">{title}</div>
          <div class="week-card-goal">{goal}</div>
        </a>''')
        if not cards:
            continue
        cards_html = "\n".join(cards)
        out.append(f'''      <div class="phase-group">
        <div class="phase-header">
          <span class="phase-no">{phase_no}</span>
          <span class="phase-name">{phase_name}</span>
          <span class="phase-desc">{phase_desc}</span>
        </div>
        <div class="week-grid">
{cards_html}
        </div>
      </div>''')
    return "\n".join(out)


def _build_resource_cards_html() -> str:
    cards = []
    for icon, name, desc, url in _RESOURCES:
        cards.append(f'''        <a class="resource-card" href="{url}">
          <span class="resource-card-icon">{icon}</span>
          <span class="resource-card-body">
            <span class="resource-card-name">{html.escape(name)}</span>
            <span class="resource-card-desc">{html.escape(desc)}</span>
          </span>
        </a>''')
    return "\n".join(cards)


def build_home(public_dir: Path, weeks: list) -> None:
    """Generate the landing page at public_dir/index.html."""
    week_cards_html = _build_week_cards_html(weeks)
    resource_cards_html = _build_resource_cards_html()
    note_count = sum(len(w["days"]) for w in weeks) + len(weeks)

    page = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM 推理系统优化 · 八周学习计划</title>
    <meta name="description" content="LLM 推理系统优化（vLLM V1 方向）八周按天学习计划与教程笔记：第一性原理、V1 源码精读、量化与投机解码、P/D 分离、分布式推理与面试冲刺。">
    <link rel="stylesheet" href="css/style.css?v=8">
</head>
<body class="landing">
    <header class="landing-nav">
        <a class="landing-nav-brand" href="index.html">vLLM <span>Notes</span></a>
        <nav class="landing-nav-links">
            <a href="plan.html">8 周计划</a>
            <a class="landing-nav-github" href="{GITHUB_REPO_URL}">GitHub ↗</a>
        </nav>
    </header>

    <section class="hero">
        <div class="hero-inner">
            <div class="hero-eyebrow">工程实战 · 8 周递进式路线</div>
            <h1 class="hero-title">vLLM 推理系统优化 <span class="hero-title-accent">八周学习计划</span></h1>
            <p class="hero-subtitle">从昇腾算子优化经验出发，吃透 vLLM V1 推理系统</p>
            <p class="hero-meta">适合具备算子优化 / 高并发分布式基础、目标 LLM 推理系统优化岗位的工程师 · 每天 2～4 小时</p>
            <div class="hero-actions">
                <a class="btn btn-primary" href="week1/index.html">🚀 开始 Week 1</a>
                <a class="btn btn-secondary" href="plan.html">📋 查看完整计划</a>
            </div>
        </div>
    </section>

    <section class="stats-strip">
        <div class="stat-item"><span class="stat-value">8</span><span class="stat-label">周学习路线</span></div>
        <div class="stat-item"><span class="stat-value">56</span><span class="stat-label">每日打卡</span></div>
        <div class="stat-item"><span class="stat-value">{note_count}</span><span class="stat-label">篇笔记教程</span></div>
        <div class="stat-item"><span class="stat-value">∞</span><span class="stat-label">持续更新</span></div>
    </section>

    <main class="landing-main">
        <section class="landing-section" id="roadmap">
            <h2 class="section-title">学习路线</h2>
            <p class="section-subtitle">三个阶段、八个主题，从推理第一性原理一路走到开源贡献与面试冲刺。</p>
{week_cards_html}
        </section>

        <section class="landing-section">
            <h2 class="section-title">更多资源</h2>
            <div class="resource-grid">
{resource_cards_html}
            </div>
        </section>
    </main>

    <footer class="landing-footer">
        <span>vLLM Notes · 由 <a href="{GITHUB_REPO_URL}">GitHub</a> 驱动 · Deployed on GitHub Pages</span>
    </footer>
</body>
</html>
'''
    (public_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Generated: {public_dir / 'index.html'}")
