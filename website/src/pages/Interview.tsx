import AuthLayout from "@/components/AuthLayout";
import MarkdownView from "@/components/MarkdownView";
import { Button } from "@/components/ui/button";
import { loadManifest, type WeekMeta } from "@/lib/content";
import { trpc } from "@/providers/trpc";
import { Loader2, SendHorizontal, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface Msg {
  role: "user" | "assistant";
  content: string;
}

export default function Interview() {
  const [weeks, setWeeks] = useState<WeekMeta[]>([]);
  const [week, setWeek] = useState(1);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [started, setStarted] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const saveInterview = trpc.study.saveInterview.useMutation();
  const history = trpc.study.listInterviews.useQuery();

  useEffect(() => {
    loadManifest().then((m) => setWeeks(m.weeks)).catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const send = async (text: string, base: Msg[]) => {
    setError(null);
    setStreaming(true);
    const nextMsgs: Msg[] = [...base, { role: "user", content: text }];
    setMessages([...nextMsgs, { role: "assistant", content: "" }]);
    try {
      const res = await fetch("/api/interview/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          week,
          messages: nextMsgs.map((m) => ({ role: m.role, content: m.content })),
        }),
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.error ?? `请求失败（${res.status}）`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      let doneSignal = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        const snapshot = acc;
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: snapshot };
          return copy;
        });
      }
      // 流正常结束
      doneSignal = acc.length > 0;
      if (!doneSignal) throw new Error("响应中断，请重试");
      const finalMsgs: Msg[] = [...nextMsgs, { role: "assistant", content: acc }];
      setMessages(finalMsgs);
      // 结束面试时保存会话
      if (text.includes("结束面试")) {
        saveInterview.mutate(
          {
            week,
            title: `${weeks.find((w) => w.week === week)?.title ?? `W${week}`} · ${new Date().toLocaleDateString("zh-CN")}`,
            messages: JSON.stringify(finalMsgs),
          },
          { onSuccess: () => history.refetch() },
        );
      }
    } catch (e) {
      setMessages(nextMsgs);
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setStreaming(false);
    }
  };

  const start = () => {
    setStarted(true);
    setMessages([]);
    // 触发面试官开场
    send("你好，我准备好了，请开始面试。", []);
  };

  return (
    <AuthLayout>
      <div className="mx-auto flex h-full max-w-3xl flex-col px-2 py-6 md:px-6">
        <h1 className="text-2xl font-bold tracking-tight">AI 模拟面试</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          基于所选周的笔记与自测题库，AI 面试官逐题提问并追问。输入「结束面试」获取四维度评分总结，会话自动存档。
        </p>

        {!started ? (
          <div className="mt-8">
            <div className="rounded-xl border border-border bg-card p-6">
              <div className="font-tech text-[11px] uppercase tracking-widest text-muted-foreground">
                选择考察范围
              </div>
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {weeks.map((w) => (
                  <button
                    key={w.week}
                    onClick={() => setWeek(w.week)}
                    className={`rounded-lg border px-4 py-3 text-left text-sm transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                      week === w.week
                        ? "border-primary bg-primary/10"
                        : "border-border hover:border-primary/40"
                    }`}
                  >
                    <span className="font-tech text-xs text-primary">
                      W{String(w.week).padStart(2, "0")}
                    </span>
                    <div className="mt-0.5 line-clamp-1">{w.title.replace(/^(第\s*\d+\s*周|Week\s*\d+)\s*[：:]\s*/, "")}</div>
                  </button>
                ))}
              </div>
              <Button className="mt-5 w-full" size="lg" onClick={start} disabled={streaming}>
                {streaming ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                开始面试
              </Button>
            </div>

            {history.data && history.data.length > 0 && (
              <div className="mt-8">
                <div className="font-tech text-[11px] uppercase tracking-widest text-muted-foreground">
                  历史面试记录
                </div>
                <div className="mt-3 space-y-2">
                  {history.data.map((h) => (
                    <div
                      key={h.id}
                      className="flex items-center justify-between rounded-lg border border-border bg-card px-4 py-3 text-sm"
                    >
                      <span className="truncate">{h.title}</span>
                      <span className="font-tech text-xs text-muted-foreground">
                        {new Date(h.createdAt).toLocaleString("zh-CN")}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <>
            <div className="mt-4 flex items-center justify-between rounded-lg border border-border bg-card px-4 py-2.5">
              <span className="font-tech text-xs text-muted-foreground">
                考察范围：W{String(week).padStart(2, "0")}
              </span>
              <Button
                variant="secondary"
                size="sm"
                disabled={streaming}
                onClick={() => send("结束面试", messages)}
              >
                <Square className="mr-1.5 h-3.5 w-3.5" />
                结束面试并评分
              </Button>
            </div>

            <div className="mt-4 flex-1 space-y-4 overflow-y-auto pb-4">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-xl px-4 py-3 text-sm ${
                      m.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "border border-border bg-card"
                    }`}
                  >
                    {m.role === "assistant" ? (
                      m.content ? (
                        <MarkdownView markdown={m.content} />
                      ) : (
                        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                      )
                    ) : (
                      <div className="whitespace-pre-wrap">{m.content}</div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>

            {error && (
              <div className="mb-3 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-2.5 text-sm text-destructive">
                {error}
              </div>
            )}
            <form
              className="flex gap-2 pb-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (input.trim() && !streaming) {
                  const text = input.trim();
                  setInput("");
                  send(text, messages);
                }
              }}
            >
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="输入你的回答…"
                disabled={streaming}
                className="flex-1 rounded-lg border border-input bg-card px-4 py-2.5 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
              />
              <Button type="submit" disabled={streaming || !input.trim()}>
                <SendHorizontal className="h-4 w-4" />
              </Button>
            </form>
          </>
        )}
      </div>
    </AuthLayout>
  );
}
