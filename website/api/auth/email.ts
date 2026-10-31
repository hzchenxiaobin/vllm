import { log } from "../lib/logger";

/**
 * 邮件投递：目前支持 Resend（HTTPS API，出站 443 即可，无需 SMTP 端口）。
 * 环境变量：
 *   RESEND_API_KEY — Resend 的 API key；未配置时邮件功能整体降级：
 *                    注册自动跳过邮箱验证，找回密码返回「未配置」错误。
 *   EMAIL_FROM     — 发件人，默认 onboarding@resend.dev（Resend 测试发件人，
 *                    只能发到自己账号邮箱；正式使用需在 Resend 验证自有域名）。
 */
export function emailProviderEnabled(): boolean {
  return Boolean(process.env.RESEND_API_KEY);
}

export async function sendAuthCodeEmail(
  to: string,
  code: string,
  purpose: "verify" | "reset",
): Promise<boolean> {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    log.warn("email provider not configured, code not delivered", {
      to,
      purpose,
    });
    return false;
  }
  const from =
    process.env.EMAIL_FROM || "学习训练营 <onboarding@resend.dev>";
  const subject =
    purpose === "verify" ? "注册验证码 - 学习训练营" : "重置密码验证码 - 学习训练营";
  const action = purpose === "verify" ? "完成注册" : "重置密码";
  const text = [
    `你的${subject}是：${code}`,
    "",
    `15 分钟内有效，用于${action}。如果这不是你的操作，请忽略本邮件。`,
  ].join("\n");

  try {
    const resp = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ from, to: [to], subject, text }),
    });
    if (!resp.ok) {
      const body = await resp.text();
      log.error("resend api rejected", { status: resp.status, body });
      return false;
    }
    return true;
  } catch (err) {
    log.error("resend api request failed", {
      error: err instanceof Error ? err.message : String(err),
    });
    return false;
  }
}
