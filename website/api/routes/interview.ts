import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { Hono } from "hono";
import { streamText } from "ai";
import { kimiGw } from "../ai/provider";
import { classifyAiError, listModels } from "../ai-client";
import { authenticateRequest } from "../kimi/auth";
import { log } from "../lib/logger";

export const interviewRoutes = new Hono();

interface DayDoc {
  key: string;
  title: string;
  quiz: { q: string; a: string }[];
  openQuiz: string[];
}
interface Manifest {
  weeks: {
    week: number;
    title: string;
    goal: string;
    days: { key: string; title: string }[];
  }[];
}

function contentDir(): string {
  const candidates = [
    path.resolve(process.cwd(), "public/content"),
    path.resolve(process.cwd(), "dist/public/content"),
  ];
  for (const c of candidates) if (existsSync(c)) return c;
  return candidates[0];
}

/** 组装面试官 system prompt：本周目标 + 每日标题 + 自测题库 */
function buildSystemPrompt(week: number): string {
  const dir = contentDir();
  const manifest: Manifest = JSON.parse(
    readFileSync(path.join(dir, "manifest.json"), "utf-8"),
  );
  const wk = manifest.weeks.find((w) => w.week === week);
  if (!wk) throw new Error("week not found");
  const questionPool: string[] = [];
  const dayTitles: string[] = [];
  for (const d of wk.days) {
    dayTitles.push(d.title);
    try {
      const doc: DayDoc = JSON.parse(
        readFileSync(path.join(dir, "days", `${d.key}.json`), "utf-8"),
      );
      for (const q of doc.quiz.slice(0, 3)) questionPool.push(q.q);
      for (const q of doc.openQuiz.slice(0, 4)) questionPool.push(q);
    } catch {
      /* skip */
    }
  }
  const pool = questionPool.slice(0, 30).map((q, i) => `${i + 1}. ${q}`).join("\n");
  return `你是一位资深 LLM 推理系统优化方向的面试官（曾负责 vLLM 推理引擎团队的技术面试）。候选人正在按「${wk.title}」的内容备考，本周目标：${wk.goal || "掌握本周主题"}。

本周覆盖的主题：
${dayTitles.map((t, i) => `- ${t}`).join("\n")}

可选考题池（优先从中选题，也可按回答深挖）：
${pool}

面试规则（严格遵守）：
1. 一次只问一个问题，等候选人回答后再继续；不要一次性抛出多个问题。
2. 候选人回答后：先简短点评（对/错/不完整，指出关键缺口），再决定追问（同一问题最多追问 2 层）或换下一题。
3. 重点考察：第一性原理推导（显存/时延手算）、机制理解（PagedAttention、调度、KV cache）、trade-off 权衡、失效模式。对背诵式回答要追问"为什么"。
4. 回答涉及数字时，要求候选人给出量级估算而非泛泛而谈。
5. 全程使用中文，语气专业但友善，每次回复控制在 200 字以内。
6. 候选人输入「结束面试」时，输出面试总结：按 概念掌握/推导能力/工程权衡/表达结构 四项各打分（10 分制），给出最强项、最弱项和 3 条针对性改进建议。

现在，请以面试官身份开始：简短自我介绍（一句话），然后提出第一个问题。`;
}

interviewRoutes.post("/chat", async (c) => {
  let user;
  try {
    user = await authenticateRequest(c.req.raw.headers);
  } catch {
    return c.json({ error: "请先登录后再开始模拟面试" }, 401);
  }
  void user;
  let body: { week?: number; messages?: { role: string; content: string }[] };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "请求格式错误" }, 400);
  }
  const week = body.week;
  const messages = body.messages;
  if (
    typeof week !== "number" ||
    !Array.isArray(messages) ||
    messages.length === 0 ||
    messages.some((m) => typeof m.content !== "string" || m.content.length > 8000)
  ) {
    return c.json({ error: "参数不完整" }, 400);
  }
  try {
    const { defaultModelId } = await listModels();
    const system = buildSystemPrompt(week);
    const result = streamText({
      model: kimiGw(defaultModelId),
      messages: [
        { role: "system", content: system },
        ...messages.slice(-20).map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
      ],
      providerOptions: { "kimi-gw": { max_completion_tokens: 2048 } },
    });
    return result.toTextStreamResponse();
  } catch (err) {
    const mapped = classifyAiError(err);
    log.error("interview chat failed", { error: mapped.message, name: mapped.name });
    const status = mapped.name === "AiTransient" ? 503 : 403;
    return c.json(
      {
        error:
          mapped.name === "AiUnavailable"
            ? "AI 额度耗尽，模拟面试暂不可用，请联系站点管理员充值"
            : mapped.name === "AiTransient"
              ? "服务暂时繁忙，请稍后重试"
              : "模拟面试服务异常：" + mapped.message,
      },
      status,
    );
  }
});
