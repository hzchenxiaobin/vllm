#!/usr/bin/env python3
"""
Build the website for GitHub Pages.

Generates public/ (deployment root):
  - Shared css/js (copied from static/)
  - Landing page index.html (built by build.home)
  - Plan page plan.html + prompt page prompt.html (built by build.weeks)
  - week1~weekN overview + day pages (built by build.weeks)
"""

import shutil
from pathlib import Path

from build.common import (
    WEEK_DIR_PATTERN,
    copy_static_assets,
    discover_week_dirs,
)
from build.home import build_home
from build.weeks import (
    build_plan_page,
    build_prompt_page,
    build_week,
)


def main() -> None:
    repo_root = Path(__file__).parent
    public_dir = repo_root / "public"

    if public_dir.exists():
        shutil.rmtree(public_dir)
    public_dir.mkdir()
    (public_dir / ".nojekyll").write_text("", encoding="utf-8")

    print("Copying static assets (css/js)...")
    copy_static_assets(public_dir)

    week_dirs = discover_week_dirs()
    week_numbers = [int(WEEK_DIR_PATTERN.match(d.name).group(1)) for d in week_dirs]

    weeks = []
    for week_dir, week_num in zip(week_dirs, week_numbers):
        print(f"Building Week {week_num} website...")
        weeks.append(build_week(week_num, public_dir, week_numbers))

    print("Building plan + prompt pages...")
    build_plan_page(public_dir, week_numbers)
    build_prompt_page(public_dir)

    print("Building landing page (index.html)...")
    build_home(public_dir, weeks)

    print("Website built successfully in public/")


if __name__ == "__main__":
    main()
