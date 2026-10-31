# vLLM 推理系统优化 · 学习训练营（website/）

基于本仓库 `content/week0` ~ `content/week9` 学习笔记构建的闭环学习网站：读笔记 → 打卡 → 自测 → 闪卡间隔复习 → 每周复盘 → AI 模拟面试，进度云端同步、多端一致。

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

从 `content/` 读取 `week*/day*.md`，生成 `website/public/content/`：

- `manifest.json` — 周/天清单（标题、目标、题目数）
- `days/{key}.json` — 每日笔记原文 + 自测题（`**Qn**：…<details>参考答案</details>` 解析题与编号开放题）+ 闪卡（"核心概念速览"表格）
- `assets/` — 笔记引用的 SVG 等配图（路径已重写）

重新生成后需重启/重新构建前端生效。

## 技术栈

- 前端：React 19 + TypeScript + Vite + Tailwind CSS v3 + shadcn/ui；react-markdown + remark-math + KaTeX 渲染笔记
- 后端：Hono + tRPC 11 + Drizzle ORM + MySQL；独立邮箱账号体系（邮箱 + 密码注册，邮件验证码校验）
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
│   ├── routes/auth.ts         # 邮箱注册/验证/登录/找回密码（Hono REST）
│   ├── auth/                  # scrypt 密码哈希、邮件发送（Resend）、限流、会话签发
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

`.env` 不要提交；本地开发复制 `.env.example` 并按需填写。本项目依赖 Kimi 网站的托管能力（数据库、AI 网关），完整功能需在 Kimi 网站环境运行；纯前端浏览笔记可在构建后用任意静态服务器查看。

## 账号体系（邮箱注册）

不依赖第三方登录，邮箱 + 密码即可开户：

- **注册**：邮箱 + 密码（≥8 位，含字母和数字）→ 6 位邮件验证码（15 分钟有效，错 5 次作废）→ 验证通过即登录
- **登录/找回密码**：验证码重置新密码，成功后 `sessionVersion` +1，该用户全部旧会话立即失效
- **密码存储**：Node 内置 scrypt（memory-hard，N=16384/r=8/p=1），哈希串自带参数与盐；验证码只存 sha256
- **会话**：httpOnly JWT cookie（HS256，`APP_SECRET` 签名），与业务表通过 `users.unionId = "email:<邮箱>"` 关联
- **限流**：注册/登录/发码按 IP + 邮箱双维度滑动窗口限流

邮件投递走 [Resend](https://resend.com)（HTTPS API，免费档 100 封/天）：

```bash
RESEND_API_KEY=re_xxx       # 留空则降级：注册自动验证、找回密码提示未配置
EMAIL_FROM=学习训练营 <onboarding@resend.dev>   # 正式使用请在 Resend 验证自有域名
```

> 若 `npm run db:push` 询问"是否为已有表重建唯一索引/截断数据"，一律选不截断（No）。
