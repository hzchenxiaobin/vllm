# vLLM 推理系统优化 · 学习训练营（website/）

基于本仓库 `week0` ~ `week9` 学习笔记构建的闭环学习网站：读笔记 → 打卡 → 自测 → 闪卡间隔复习 → 每周复盘 → AI 模拟面试，进度云端同步、多端一致。

## 功能

- **仪表盘**：总进度、连续打卡天数、待复习闪卡数、周进度分段条、"继续学习"直达下一个未完成的一天
- **每日页**：笔记渲染（LaTeX 公式 / GFM 表格 / SVG 配图）、自测（解析题 + 开放题，自评计分）、闪卡（翻面 + 四档评分）
- **闪卡复习**：简化 SM-2 间隔重复，到期卡片自动进入复习队列
- **每周复盘**：带四问模板的编辑器，保存后同步
- **AI 模拟面试**：按周选题，AI 面试官基于该周笔记与题库逐题追问，「结束面试」输出四维度评分并存档

## 内容管线

笔记内容经脚本离线提取为网站数据，不依赖运行时爬取：

```bash
python3 website/scripts/build_content.py
```

从仓库根目录读取 `week*/day*.md`，生成 `website/public/content/`：

- `manifest.json` — 周/天清单（标题、目标、题目数）
- `days/{key}.json` — 每日笔记原文 + 自测题（`**Qn**：…<details>参考答案</details>` 解析题与编号开放题）+ 闪卡（"核心概念速览"表格）
- `assets/` — 笔记引用的 SVG 等配图（路径已重写）

重新生成后需重启/重新构建前端生效。

## 技术栈

- 前端：React 19 + TypeScript + Vite + Tailwind CSS v3 + shadcn/ui；react-markdown + remark-math + KaTeX 渲染笔记
- 后端：Hono + tRPC 11 + Drizzle ORM + MySQL；Kimi OAuth 登录
- AI：平台网关 OpenAI 兼容接口（`KIMI_AGENTGW_*` 环境变量注入，仅服务端）

## 目录结构

```
website/
├── scripts/build_content.py   # 内容管线（仓库笔记 → 网站数据）
├── public/content/            # 生成的内容包（manifest + days + assets）
├── src/
│   ├── pages/                 # Home 仪表盘 / Week 周页 / Day 日页 / Review 复习 / Interview 面试
│   ├── components/            # AuthLayout（侧栏+周导航）、MarkdownView、ui/
│   └── lib/content.ts         # 内容加载与类型
├── api/
│   ├── study-router.ts        # tRPC：打卡、成绩、闪卡 SRS、复盘、面试存档
│   ├── routes/interview.ts    # AI 面试流式接口（Hono REST）
│   └── ai-client.ts           # 模型清单 + 错误归类
├── db/schema.ts               # checkins / quiz_attempts / card_states / week_reviews / interview_sessions
└── contracts/                 # 前后端共享常量
```

## 本地运行

```bash
cd website
npm install
npm run db:push   # 同步数据库表结构（需要 DATABASE_URL，见 .env.example）
npm run dev       # http://localhost:3000
```

`.env` 不要提交；本地开发复制 `.env.example` 并按需填写。本项目依赖 Kimi 网站的托管能力（OAuth、数据库、AI 网关），完整功能需在 Kimi 网站环境运行；纯前端浏览笔记可在构建后用任意静态服务器查看。
