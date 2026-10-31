import { Hono, type Context } from "hono";
import { setCookie } from "hono/cookie";
import { z } from "zod";
import { and, desc, eq, isNull } from "drizzle-orm";
import { createHash, randomInt } from "node:crypto";
import * as schema from "@db/schema";
import { getDb } from "../queries/connection";
import { findUserByUnionId } from "../queries/users";
import { getSessionCookieOptions } from "../lib/cookies";
import { Session } from "@contracts/constants";
import { log } from "../lib/logger";
import { hashPassword, verifyPassword } from "../auth/password";
import { emailProviderEnabled, sendAuthCodeEmail } from "../auth/email";
import { rateLimitOk } from "../auth/ratelimit";
import {
  emailToUnionId,
  mintEmailSessionToken,
} from "../auth/session";

/**
 * 邮箱账号体系 REST 接口（/api/auth/*）。
 * 会话复用全站统一的 JWT cookie，登录成功后前端调 trpc.auth.me 即可拿到用户。
 */
export const authRoutes = new Hono();

const emailSchema = z
  .string()
  .trim()
  .toLowerCase()
  .pipe(z.email("邮箱格式不正确"))
  .pipe(z.string().max(320));
const passwordSchema = z
  .string()
  .min(8, "密码至少 8 位")
  .max(72, "密码最长 72 位")
  .regex(/[a-zA-Z]/, "密码需包含字母")
  .regex(/[0-9]/, "密码需包含数字");
const codeSchema = z.string().trim().regex(/^\d{6}$/, "验证码为 6 位数字");

const TOKEN_TTL_MS = 15 * 60 * 1000;
const MAX_CODE_ATTEMPTS = 5;

function clientIp(headers: Headers): string {
  return (
    headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    headers.get("x-real-ip") ||
    "local"
  );
}

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

/** 生成 6 位验证码并落库（只存哈希），返回明文码用于发邮件 */
async function issueCode(
  userId: number,
  purpose: "verify" | "reset",
): Promise<string> {
  const code = String(randomInt(0, 1_000_000)).padStart(6, "0");
  await getDb().insert(schema.emailTokens).values({
    userId,
    tokenHash: sha256(code),
    purpose,
    expiresAt: new Date(Date.now() + TOKEN_TTL_MS),
  });
  return code;
}

/** 校验验证码：取该用户该用途最新一条未用令牌，错 5 次作废，成功即标记已用 */
async function checkCode(
  userId: number,
  purpose: "verify" | "reset",
  code: string,
): Promise<boolean> {
  const db = getDb();
  const rows = await db
    .select()
    .from(schema.emailTokens)
    .where(
      and(
        eq(schema.emailTokens.userId, userId),
        eq(schema.emailTokens.purpose, purpose),
        isNull(schema.emailTokens.usedAt),
      ),
    )
    .orderBy(desc(schema.emailTokens.createdAt))
    .limit(1);
  const token = rows.at(0);
  if (!token) return false;
  if (token.expiresAt.getTime() < Date.now()) return false;
  if (token.attempts >= MAX_CODE_ATTEMPTS) return false;
  if (token.tokenHash !== sha256(code)) {
    await db
      .update(schema.emailTokens)
      .set({ attempts: token.attempts + 1 })
      .where(eq(schema.emailTokens.id, token.id));
    return false;
  }
  await db
    .update(schema.emailTokens)
    .set({ usedAt: new Date() })
    .where(eq(schema.emailTokens.id, token.id));
  return true;
}

async function establishSession(
  c: Context,
  user: { unionId: string; sessionVersion: number },
) {
  const token = await mintEmailSessionToken(user.unionId, user.sessionVersion);
  setCookie(c, Session.cookieName, token, {
    ...getSessionCookieOptions(c.req.raw.headers),
    maxAge: Session.maxAgeMs / 1000,
  });
}

function zodMessage(err: z.ZodError): string {
  return err.issues[0]?.message ?? "参数不合法";
}

// ---------- 注册 ----------
authRoutes.post("/register", async (c) => {
  const ip = clientIp(c.req.raw.headers);
  if (!rateLimitOk(`rg:ip:${ip}`, 20, 3_600_000)) {
    return c.json({ error: "操作过于频繁，请稍后再试" }, 429);
  }
  const parsed = z
    .object({
      email: emailSchema,
      password: passwordSchema,
      name: z.string().trim().max(64).optional(),
    })
    .safeParse(await c.req.json().catch(() => ({})));
  if (!parsed.success) {
    return c.json({ error: zodMessage(parsed.error) }, 400);
  }
  const { email, password, name } = parsed.data;
  if (!rateLimitOk(`rg:em:${email}`, 5, 3_600_000)) {
    return c.json({ error: "该邮箱操作过于频繁，请稍后再试" }, 429);
  }

  const unionId = emailToUnionId(email);
  const existing = await findUserByUnionId(unionId);
  if (existing?.passwordHash && existing.emailVerifiedAt) {
    return c.json({ error: "该邮箱已注册，请直接登录或使用找回密码" }, 409);
  }

  const passwordHash = await hashPassword(password);
  const displayName = name || email.split("@")[0];
  const canSend = emailProviderEnabled();

  if (!canSend) {
    // 未配置邮件服务：个人部署场景下降级为注册即验证通过，直接登录
    const verifiedAt = new Date();
    if (existing) {
      await getDb()
        .update(schema.users)
        .set({ passwordHash, name: displayName, email, emailVerifiedAt: verifiedAt })
        .where(eq(schema.users.id, existing.id));
    } else {
      await getDb().insert(schema.users).values({
        unionId,
        email,
        name: displayName,
        passwordHash,
        emailVerifiedAt: verifiedAt,
        lastSignInAt: new Date(),
      });
    }
    const user = await findUserByUnionId(unionId);
    if (!user) return c.json({ error: "注册失败，请重试" }, 500);
    await establishSession(c, user);
    log.warn("email provider not configured, auto-verified registration", {
      email,
    });
    return c.json({ ok: true, sent: false, autoVerified: true });
  }

  if (existing) {
    await getDb()
      .update(schema.users)
      .set({ passwordHash, name: displayName, email })
      .where(eq(schema.users.id, existing.id));
  } else {
    await getDb().insert(schema.users).values({
      unionId,
      email,
      name: displayName,
      passwordHash,
      lastSignInAt: new Date(),
    });
  }
  const user = await findUserByUnionId(unionId);
  if (!user) return c.json({ error: "注册失败，请重试" }, 500);
  const code = await issueCode(user.id, "verify");
  const delivered = await sendAuthCodeEmail(email, code, "verify");
  if (!delivered) {
    return c.json({ error: "验证码邮件发送失败，请稍后重试" }, 502);
  }
  return c.json({ ok: true, sent: true });
});

// ---------- 验证邮箱 ----------
authRoutes.post("/verify", async (c) => {
  const parsed = z
    .object({ email: emailSchema, code: codeSchema })
    .safeParse(await c.req.json().catch(() => ({})));
  if (!parsed.success) {
    return c.json({ error: zodMessage(parsed.error) }, 400);
  }
  const { email, code } = parsed.data;
  if (!rateLimitOk(`vf:${email}`, 20, 900_000)) {
    return c.json({ error: "尝试次数过多，请稍后再试" }, 429);
  }
  const user = await findUserByUnionId(emailToUnionId(email));
  if (!user) return c.json({ error: "验证码无效或已过期" }, 400);
  if (!user.emailVerifiedAt) {
    const ok = await checkCode(user.id, "verify", code);
    if (!ok) return c.json({ error: "验证码无效或已过期" }, 400);
    await getDb()
      .update(schema.users)
      .set({ emailVerifiedAt: new Date() })
      .where(eq(schema.users.id, user.id));
  }
  const fresh = await findUserByUnionId(user.unionId);
  if (!fresh) return c.json({ error: "验证失败，请重试" }, 500);
  await establishSession(c, fresh);
  return c.json({ ok: true });
});

// ---------- 登录 ----------
authRoutes.post("/login", async (c) => {
  const ip = clientIp(c.req.raw.headers);
  const parsed = z
    .object({ email: emailSchema, password: z.string().min(1, "请输入密码") })
    .safeParse(await c.req.json().catch(() => ({})));
  if (!parsed.success) {
    return c.json({ error: zodMessage(parsed.error) }, 400);
  }
  const { email, password } = parsed.data;
  if (
    !rateLimitOk(`li:ip:${ip}`, 30, 900_000) ||
    !rateLimitOk(`li:em:${email}`, 10, 900_000)
  ) {
    return c.json({ error: "尝试过于频繁，请 15 分钟后再试" }, 429);
  }

  const user = await findUserByUnionId(emailToUnionId(email));
  if (!user?.passwordHash || !(await verifyPassword(password, user.passwordHash))) {
    return c.json({ error: "邮箱或密码错误" }, 400);
  }
  if (!user.emailVerifiedAt) {
    if (emailProviderEnabled()) {
      const code = await issueCode(user.id, "verify");
      await sendAuthCodeEmail(email, code, "verify");
      return c.json(
        { error: "EMAIL_NOT_VERIFIED", message: "邮箱尚未验证，验证码已重新发送" },
        403,
      );
    }
    await getDb()
      .update(schema.users)
      .set({ emailVerifiedAt: new Date() })
      .where(eq(schema.users.id, user.id));
  }
  await getDb()
    .update(schema.users)
    .set({ lastSignInAt: new Date() })
    .where(eq(schema.users.id, user.id));
  await establishSession(c, user);
  return c.json({
    ok: true,
    user: { id: user.id, name: user.name, email: user.email },
  });
});

// ---------- 忘记密码 ----------
authRoutes.post("/forgot-password", async (c) => {
  const parsed = z
    .object({ email: emailSchema })
    .safeParse(await c.req.json().catch(() => ({})));
  if (!parsed.success) {
    return c.json({ error: zodMessage(parsed.error) }, 400);
  }
  const { email } = parsed.data;
  if (!emailProviderEnabled()) {
    return c.json(
      { error: "邮件服务未配置，无法发送重置邮件，请联系站点管理员" },
      503,
    );
  }
  if (!rateLimitOk(`fp:${email}`, 5, 3_600_000)) {
    return c.json({ error: "操作过于频繁，请稍后再试" }, 429);
  }
  const user = await findUserByUnionId(emailToUnionId(email));
  if (user?.passwordHash) {
    const code = await issueCode(user.id, "reset");
    const delivered = await sendAuthCodeEmail(email, code, "reset");
    if (!delivered) {
      log.error("reset code delivery failed", { email });
    }
  }
  // 无论邮箱是否注册都返回成功，防账号枚举
  return c.json({ ok: true, sent: true });
});

// ---------- 重置密码 ----------
authRoutes.post("/reset-password", async (c) => {
  const parsed = z
    .object({ email: emailSchema, code: codeSchema, password: passwordSchema })
    .safeParse(await c.req.json().catch(() => ({})));
  if (!parsed.success) {
    return c.json({ error: zodMessage(parsed.error) }, 400);
  }
  const { email, code, password } = parsed.data;
  if (!rateLimitOk(`rs:${email}`, 20, 900_000)) {
    return c.json({ error: "尝试次数过多，请稍后再试" }, 429);
  }
  const user = await findUserByUnionId(emailToUnionId(email));
  if (!user) return c.json({ error: "验证码无效或已过期" }, 400);
  const ok = await checkCode(user.id, "reset", code);
  if (!ok) return c.json({ error: "验证码无效或已过期" }, 400);
  await getDb()
    .update(schema.users)
    .set({
      passwordHash: await hashPassword(password),
      // 会话版本 +1：所有旧 JWT 立即失效
      sessionVersion: user.sessionVersion + 1,
    })
    .where(eq(schema.users.id, user.id));
  return c.json({ ok: true });
});
