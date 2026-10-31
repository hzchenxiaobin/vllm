import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type Mode = "login" | "register" | "forgot";
/** 表单步骤：entry=填邮箱密码；code=填验证码；reset=填验证码+新密码 */
type Step = "entry" | "code" | "reset";

async function postAuth(
  path: string,
  body: Record<string, string>,
): Promise<{ ok: boolean; status: number; data: Record<string, unknown> }> {
  try {
    const resp = await fetch(`/api/auth${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(body),
    });
    const data = (await resp.json().catch(() => ({}))) as Record<string, unknown>;
    return { ok: resp.ok, status: resp.status, data };
  } catch {
    return { ok: false, status: 0, data: { error: "网络错误，请稍后重试" } };
  }
}

export default function Login() {
  const [mode, setMode] = useState<Mode>("login");
  const [step, setStep] = useState<Step>("entry");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [code, setCode] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  const goHome = () => {
    window.location.href = "/";
  };

  const reset = (m: Mode) => {
    setMode(m);
    setStep("entry");
    setCode("");
    setError("");
    setMessage("");
  };

  async function run(fn: () => Promise<void>) {
    setPending(true);
    setError("");
    setMessage("");
    try {
      await fn();
    } finally {
      setPending(false);
    }
  }

  const submit = () =>
    run(async () => {
      if (mode === "register" && step === "entry") {
        const r = await postAuth("/register", { email, password, name });
        if (!r.ok) return setError(String(r.data.error ?? "注册失败"));
        if (r.data.autoVerified) return goHome();
        setStep("code");
        setMessage("验证码已发送到邮箱，15 分钟内有效");
        return;
      }
      if (step === "code") {
        const r = await postAuth("/verify", { email, code });
        if (!r.ok) return setError(String(r.data.error ?? "验证失败"));
        return goHome();
      }
      if (mode === "login") {
        const r = await postAuth("/login", { email, password });
        if (!r.ok) {
          if (r.data.error === "EMAIL_NOT_VERIFIED") {
            setMode("register");
            setStep("code");
            setMessage("邮箱尚未验证，验证码已重新发送，请查收");
            return;
          }
          return setError(String(r.data.error ?? "登录失败"));
        }
        return goHome();
      }
      if (mode === "forgot" && step === "entry") {
        const r = await postAuth("/forgot-password", { email });
        if (!r.ok) return setError(String(r.data.error ?? "发送失败"));
        setStep("reset");
        setMessage("如果该邮箱已注册，重置验证码已发送，15 分钟内有效");
        return;
      }
      if (step === "reset") {
        const r = await postAuth("/reset-password", { email, code, password });
        if (!r.ok) return setError(String(r.data.error ?? "重置失败"));
        setMessage("密码已重置，请用新密码登录");
        setMode("login");
        setStep("entry");
        setPassword("");
        setCode("");
      }
    });

  const title =
    mode === "login" ? "登录" : mode === "register" ? "注册" : "找回密码";
  const buttonText = pending
    ? "处理中…"
    : step === "code"
      ? "完成验证并登录"
      : step === "reset"
        ? "重置密码"
        : mode === "login"
          ? "登录"
          : mode === "register"
            ? "注册"
            : "发送重置验证码";

  return (
    <div className="flex min-h-dvh items-center justify-center bg-background p-4">
      <Card className="w-full max-w-sm border-border bg-card">
        <CardHeader className="space-y-3 text-center">
          <div className="font-tech text-xs uppercase tracking-[0.3em] text-primary">
            vLLM TRAINING CAMP
          </div>
          <CardTitle className="text-xl">{title}</CardTitle>
          <div className="flex justify-center gap-2 text-xs">
            {(["login", "register", "forgot"] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => reset(m)}
                className={`rounded px-2 py-1 transition-colors ${
                  mode === m
                    ? "bg-primary/15 text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {m === "login" ? "登录" : m === "register" ? "注册" : "忘记密码"}
              </button>
            ))}
          </div>
        </CardHeader>
        <CardContent>
          <form
            className="flex flex-col gap-4"
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="email">邮箱</Label>
              <Input
                id="email"
                type="email"
                required
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={step !== "entry"}
              />
            </div>

            {mode === "register" && step === "entry" && (
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="name">昵称（可选）</Label>
                <Input
                  id="name"
                  type="text"
                  maxLength={64}
                  placeholder="默认取邮箱前缀"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
            )}

            {(step === "code" || step === "reset") && (
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="code">邮箱验证码</Label>
                <Input
                  id="code"
                  inputMode="numeric"
                  required
                  maxLength={6}
                  placeholder="6 位数字"
                  value={code}
                  onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
                />
              </div>
            )}

            {((mode !== "forgot" && step === "entry") || step === "reset") && (
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="password">
                  {step === "reset" ? "新密码" : "密码"}
                </Label>
                <Input
                  id="password"
                  type="password"
                  required
                  autoComplete={
                    mode === "login" ? "current-password" : "new-password"
                  }
                  placeholder="至少 8 位，含字母和数字"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            )}

            {message && (
              <p className="text-sm text-success">{message}</p>
            )}
            {error && <p className="text-sm text-destructive">{error}</p>}

            <Button type="submit" size="lg" disabled={pending} className="w-full">
              {buttonText}
            </Button>

            {mode === "login" && (
              <p className="text-center text-xs text-muted-foreground">
                还没有账号？点上方「注册」，邮箱即可开户。
              </p>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
