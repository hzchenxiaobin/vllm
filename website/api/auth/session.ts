import * as jose from "jose";
import * as cookie from "cookie";
import { env } from "../lib/env";
import { Session } from "@contracts/constants";
import { Errors } from "@contracts/errors";
import { authenticateRequest } from "../kimi/auth";
import { findUserByUnionId } from "../queries/users";

/**
 * 邮箱账号的会话与 Kimi OAuth 共用同一枚 JWT cookie（Session.cookieName），
 * 区别只在 payload：邮箱用户的 unionId 形如 "email:<邮箱>"，并多一个 sv
 * （sessionVersion）声明——改密/重置密码时 users.sessionVersion +1，
 * 旧 JWT 立即失效，等价于「吊销该用户全部旧会话」。
 */
export const EMAIL_UNION_PREFIX = "email:";

const JWT_ALG = "HS256";

export function emailToUnionId(email: string): string {
  return EMAIL_UNION_PREFIX + email.trim().toLowerCase();
}

export async function mintEmailSessionToken(
  unionId: string,
  sessionVersion: number,
): Promise<string> {
  const secret = new TextEncoder().encode(env.appSecret);
  return new jose.SignJWT({ unionId, clientId: env.appId, sv: sessionVersion })
    .setProtectedHeader({ alg: JWT_ALG })
    .setIssuedAt()
    .setExpirationTime("1 year")
    .sign(secret);
}

/**
 * 请求认证入口（api/context.ts 使用）：先按 JWT 里的 unionId 前缀分流——
 * 邮箱用户走本地校验（含 sv 会话版本检查），其余交给 Kimi OAuth 的原逻辑。
 */
export async function authenticateAny(headers: Headers) {
  const cookies = cookie.parse(headers.get("cookie") || "");
  const token = cookies[Session.cookieName];
  if (!token) {
    throw Errors.forbidden("Invalid authentication token.");
  }
  // 只看前缀，不验签——验签在各自分支里做
  const claim = jose.decodeJwt(token);
  const unionId = typeof claim.unionId === "string" ? claim.unionId : "";
  if (!unionId.startsWith(EMAIL_UNION_PREFIX)) {
    return authenticateRequest(headers);
  }
  try {
    const secret = new TextEncoder().encode(env.appSecret);
    const { payload } = await jose.jwtVerify(token, secret, {
      algorithms: [JWT_ALG],
    });
    const user = await findUserByUnionId(unionId);
    if (!user) {
      throw Errors.forbidden("User not found. Please re-login.");
    }
    if (typeof payload.sv !== "number" || payload.sv !== user.sessionVersion) {
      throw Errors.forbidden("Session expired. Please re-login.");
    }
    return user;
  } catch (err) {
    if (err && typeof err === "object" && "tag" in err) throw err;
    throw Errors.forbidden("Invalid authentication token.");
  }
}
