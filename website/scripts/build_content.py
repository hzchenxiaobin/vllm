#!/usr/bin/env python3
"""Parse hzchenxiaobin/vllm study notes into website content bundles."""
import json, os, re, shutil, sys

# 在本仓库内运行：python3 website/scripts/build_content.py
# SRC = 仓库根目录（week0~week9 所在）；OUT = website/public/content
import pathlib
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = str(_REPO_ROOT)
OUT = str(pathlib.Path(__file__).resolve().parents[1] / "public" / "content")
ASSET_OUT = os.path.join(OUT, "assets")
DAY_OUT = os.path.join(OUT, "days")

os.makedirs(ASSET_OUT, exist_ok=True)
os.makedirs(DAY_OUT, exist_ok=True)

def slug(s):
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()

def parse_week_meta(week_dir, widx):
    readme = os.path.join(week_dir, "README.md")
    title, goal = f"第 {widx} 周", ""
    if os.path.exists(readme):
        txt = open(readme, encoding="utf-8").read()
        m = re.search(r"^#\s*(.+)$", txt, re.M)
        if m:
            t = m.group(1).strip()
            t = re.sub(r"[·\-—]\s*(展开手册|源码精读.*)$", "", t).strip()
            title = t
        g = re.search(r"\*\*本周目标\*\*[：:]\s*(.+)", txt)
        if g:
            goal = g.group(1).strip()
    return title, goal

def extract_quiz(md):
    """Extract quiz: details-format Q&A (with answers) + numbered open questions."""
    quiz, open_q = [], []
    pattern = re.compile(
        r"\*\*Q(\d+)\*\*[：:]\s*(.+?)\s*<details>\s*<summary>(.*?)</summary>(.*?)</details>",
        re.S,
    )
    covered = []
    for m in pattern.finditer(md):
        q = m.group(2).strip()
        a = m.group(4).strip()
        if len(q) > 5 and a:
            quiz.append({"q": q, "a": a})
            covered.append((m.start(), m.end()))
    # self-test sections: numbered open questions
    for sm in re.finditer(r"^#{2,3}[^\n]*自测[^\n]*\n(.*?)(?=^#{1,2} |\Z)", md, re.M | re.S):
        body = sm.group(1)
        # remove covered detail blocks to avoid duplicates
        for m in re.finditer(r"^\s*(?:\d+[.、)]|\*\*Q\d+\*\*[：:])\s*(.+)$", body, re.M):
            line = m.group(1).strip()
            line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            if 10 < len(line) < 600 and not any(q["q"][:20] in line for q in quiz):
                open_q.append(line)
    return quiz, open_q[:12]

def extract_cards(md):
    """Extract concept tables under 核心概念速览-like sections: | **term** | def | ... |"""
    cards = []
    # find section containing 核心概念 or 速览 table
    for m in re.finditer(r"^#{2,3}[^\n]*(核心概念|速览|概念速查|术语)[^\n]*\n(.*?)(?=^#{1,3} |\Z)", md, re.M | re.S):
        body = m.group(2)
        lines = [l for l in body.splitlines() if l.strip().startswith("|")]
        if len(lines) < 3:
            continue
        # skip header + separator
        for row in lines[2:]:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and cells[1] and "---" not in cells[0]:
                front = re.sub(r"\*\*(.+?)\*\*", r"\1", cells[0]).strip()
                back = cells[1].strip()
                if 1 < len(front) < 60 and 4 < len(back) < 400:
                    cards.append({"front": front, "back": back})
    # dedupe
    seen, out = set(), []
    for c in cards:
        if c["front"] not in seen:
            seen.add(c["front"])
            out.append(c)
    return out

def rewrite_assets(md, week_tag, day_stem):
    def repl(m):
        alt, src = m.group(1), m.group(2)
        if src.startswith("http") or src.startswith("/"):
            return m.group(0)
        fname = f"{week_tag}_{day_stem}_{os.path.basename(src)}"
        src_path = os.path.join(cur_dir, src)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(ASSET_OUT, fname))
            return f"![{alt}](/content/assets/{fname})"
        return m.group(0)
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)

weeks = []
total_q = total_c = 0
for widx in range(0, 10):
    week_dir = os.path.join(SRC, f"week{widx}")
    if not os.path.isdir(week_dir):
        continue
    cur_dir = week_dir
    wtitle, wgoal = parse_week_meta(week_dir, widx)
    day_files = sorted(
        f for f in os.listdir(week_dir)
        if f.endswith(".md") and f != "README.md" and re.match(r"day", f)
    )
    days = []
    for i, fn in enumerate(day_files):
        stem = fn[:-3]
        md = open(os.path.join(week_dir, fn), encoding="utf-8").read()
        tm = re.search(r"^#\s*(.+)$", md, re.M)
        title = tm.group(1).strip() if tm else stem
        title = re.sub(r"^Day\s*\d+[a-z]?\s*[·\-：:]\s*", "", title)
        quiz, open_q = extract_quiz(md)
        cards = extract_cards(md)
        total_q += len(quiz) + len(open_q); total_c += len(cards)
        key = f"w{widx}d{i+1:02d}"
        proc = rewrite_assets(md, f"w{widx}", slug(stem)[:40])
        with open(os.path.join(DAY_OUT, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump({"key": key, "title": title, "markdown": proc, "quiz": quiz, "openQuiz": open_q, "cards": cards}, f, ensure_ascii=False)
        days.append({"key": key, "title": title, "quizCount": len(quiz) + len(open_q), "cardCount": len(cards)})
    weeks.append({"week": widx, "title": wtitle, "goal": wgoal, "days": days})

with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({
        "siteTitle": "vLLM 推理系统优化 · 学习训练营",
        "source": "https://github.com/hzchenxiaobin/vllm",
        "weeks": weeks,
    }, f, ensure_ascii=False, indent=1)

print(f"weeks={len(weeks)} days={sum(len(w['days']) for w in weeks)} quizzes={total_q} cards={total_c}")
print(f"assets={len(os.listdir(ASSET_OUT))}")
for w in weeks:
    print(f"  w{w['week']}: {len(w['days'])} days — {w['title'][:40]}")
