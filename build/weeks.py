"""Builders for week overview/day pages, the plan page and the prompt page."""

import re
import shutil
from pathlib import Path

from .common import (
    GITHUB_REPO_URL,
    PLAN_SOURCE,
    PROMPT_SOURCE,
    REPO_ROOT,
    build_day_cards_html,
    discover_days,
    parse_week_readme,
    rewrite_md_links_to_html,
    solution_page_template,
)

WEEK_TITLES = {
    0: "推理系统优化专家岗 · 7 天冲刺计划",
    1: "推理基础与性能建模（第一性原理）",
    2: "vLLM V1 源码精读（上）· 调度链路",
    3: "vLLM V1 源码精读（下）· KV 管理与执行",
    4: "进阶专题（一）· 量化与投机解码",
    5: "进阶专题（二）· P/D 分离与分布式",
    6: "项目 A · vLLM-Ascend / 源码贡献（上）",
    7: "项目 A（下）+ 项目 C 消融实验",
    8: "面试冲刺",
}

WEEK_GOALS = {
    0: "距面试仅 1 周的应急路线：放弃全面深入，追求面试产出最大化",
    1: "把 LLM 推理变成能手算的数学题，建立 Roofline 思维",
    2: "讲清请求从进入引擎到吐出 token 的全生命周期",
    3: "PagedAttention 与执行器的源码级细节",
    4: "W8A8 / AWQ / KV Cache 量化与投机解码",
    5: "P/D 分离架构与 TP/PP/EP 分布式推理",
    6: "向 vLLM-Ascend 提交第一个 PR",
    7: "完成开源贡献与消融实验报告",
    8: "八大专题手撕与全真模拟面试",
}

WEEK_HEADING_ANCHOR_PATTERN = re.compile(
    r"^(##\s*(?:第\s*(\d+)\s*周|Week\s*(\d+))[：:].*)$", re.MULTILINE
)


def _week_dir(week_num: int) -> Path:
    return REPO_ROOT / f"week{week_num}"


def _day_pills(days: list, current_day=None, overview_active: bool = False, prefix: str = "") -> list:
    """Pill strip items for the week pages: 概览 + one pill per day."""
    pills = [{"label": "📌 概览", "href": f"{prefix}index.html", "active": overview_active}]
    for day in days:
        pills.append({
            "label": f"Day {day['num']}",
            "href": f"{prefix}day{day['num']}.html",
            "active": current_day == day["num"],
        })
    return pills


def _day_prev_next(week_num: int, days: list, index: int, week_numbers: list) -> tuple:
    """(prev_link, next_link) for a day page; each link is (href, label) or None."""
    if index > 0:
        prev_day = days[index - 1]
        prev_link = (f"day{prev_day['num']}.html", f"Day {prev_day['num']}：{prev_day['title']}")
    else:
        prev_link = ("index.html", "本周概览")

    if index + 1 < len(days):
        next_day = days[index + 1]
        next_link = (f"day{next_day['num']}.html", f"Day {next_day['num']}：{next_day['title']}")
    else:
        later = [n for n in week_numbers if n > week_num]
        if later:
            next_num = later[0]
            next_link = (
                f"../week{next_num}/index.html",
                f"Week {next_num}：{WEEK_TITLES.get(next_num, '')}",
            )
        else:
            next_link = ("../index.html", "返回首页")
    return prev_link, next_link


def _copy_assets(week_dir: Path, output_dir: Path) -> None:
    """Copy weekN/assets/ images into the website output."""
    assets_src = week_dir / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, output_dir / "assets", dirs_exist_ok=True)


def build_week(week_num: int, public_dir: Path, week_numbers: list) -> dict:
    """Build one week's website: public/weekN/index.html + dayN.html pages.

    Returns the week metadata used by the landing page.
    """
    week_dir = _week_dir(week_num)
    output_dir = public_dir / f"week{week_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = parse_week_readme(week_dir, week_num, WEEK_TITLES.get(week_num, f"Week {week_num}"))
    days = discover_days(week_dir)

    root_prefix = "../"
    overview_body = rewrite_md_links_to_html(meta["body"])
    for day in days:
        day["markdown"] = rewrite_md_links_to_html(day["markdown"])

    if days:
        overview_with_cards = (
            overview_body + "\n\n## 🚀 进入每日学习\n\n" + build_day_cards_html(days)
        )
    else:
        overview_with_cards = (
            overview_body + "\n\n> 📝 本周教程将按天陆续生成，敬请期待。\n"
        )

    overview_html = solution_page_template(
        title=meta["title"],
        eyebrow=meta["eyebrow"],
        markdown=overview_with_cards,
        back_link=(f"{root_prefix}index.html", "返回首页"),
        root_prefix=root_prefix,
        page_title=f"Week {week_num} 概览",
        day_pills=_day_pills(days, overview_active=True),
    )
    (output_dir / "index.html").write_text(overview_html, encoding="utf-8")
    print(f"Generated: {output_dir / 'index.html'}")

    for index, day in enumerate(days):
        prev_link, next_link = _day_prev_next(week_num, days, index, week_numbers)
        html = solution_page_template(
            title=day["title"],
            markdown=day["markdown"],
            root_prefix=root_prefix,
            page_title=f"Week {week_num} - Day {day['num']}：{day['title']}",
            day_pills=_day_pills(days, current_day=day["num"]),
            prev_link=prev_link,
            next_link=next_link,
        )
        filename = f"day{day['num']}.html"
        (output_dir / filename).write_text(html, encoding="utf-8")
        print(f"Generated: {output_dir / filename}")

    _copy_assets(week_dir, output_dir)

    return {
        "num": week_num,
        "eyebrow": meta["eyebrow"],
        "title": meta["title"],
        "goal": WEEK_GOALS.get(week_num, ""),
        "days": days,
    }


def build_plan_page(public_dir: Path, week_numbers: list) -> None:
    """Build the full 8-week plan page from the repo README."""
    if not PLAN_SOURCE.exists():
        print(f"Warning: plan source not found: {PLAN_SOURCE}")
        return

    markdown_text = PLAN_SOURCE.read_text(encoding="utf-8")
    # Strip the plan's own H1 — the page template shows the title instead.
    markdown_text = re.sub(r"^\s*#\s+.+\n", "", markdown_text, count=1)
    markdown_text = rewrite_md_links_to_html(markdown_text)

    def add_week_anchor(match: re.Match) -> str:
        return f'<a id="week-{match.group(2) or match.group(3)}"></a>\n{match.group(0)}'

    markdown_text = WEEK_HEADING_ANCHOR_PATTERN.sub(add_week_anchor, markdown_text)

    week_pills = [
        {"label": f"W{num}", "href": f"week{num}/index.html"}
        for num in week_numbers
    ]

    html = solution_page_template(
        title="vLLM 推理系统优化 · 八周学习计划",
        eyebrow="📋 完整学习计划",
        markdown=markdown_text,
        back_link=("index.html", "返回首页"),
        page_title="八周学习计划",
        day_pills=week_pills,
    )
    (public_dir / "plan.html").write_text(html, encoding="utf-8")
    print(f"Generated: {public_dir / 'plan.html'}")


def build_prompt_page(public_dir: Path) -> None:
    """Build the tutorial-generation prompt page from prompt.md."""
    if not PROMPT_SOURCE.exists():
        print(f"Warning: prompt source not found: {PROMPT_SOURCE}")
        return

    markdown_text = PROMPT_SOURCE.read_text(encoding="utf-8")
    markdown_text = rewrite_md_links_to_html(markdown_text)

    html = solution_page_template(
        title="教程生成 Prompt",
        eyebrow="🤖 站点工具",
        markdown=markdown_text,
        back_link=("index.html", "返回首页"),
        page_title="教程生成 Prompt",
    )
    (public_dir / "prompt.html").write_text(html, encoding="utf-8")
    print(f"Generated: {public_dir / 'prompt.html'}")
