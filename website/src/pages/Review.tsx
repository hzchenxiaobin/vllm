import AuthLayout from "@/components/AuthLayout";
import MarkdownView from "@/components/MarkdownView";
import { Button } from "@/components/ui/button";
import { loadDay, type CardItem } from "@/lib/content";
import { trpc } from "@/providers/trpc";
import { Flame, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

interface DueCard {
  cardId: string;
  front: string;
  back: string;
}

export default function Review() {
  const states = trpc.study.cardStates.useQuery();
  const reviewCard = trpc.study.reviewCard.useMutation();
  const utils = trpc.useUtils();
  const [queue, setQueue] = useState<DueCard[]>([]);
  const [loading, setLoading] = useState(false);
  const [idx, setIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);

  const dueStates = useMemo(() => {
    if (!states.data) return [];
    const now = Date.now();
    return states.data.filter((s) => new Date(s.dueAt).getTime() <= now);
  }, [states.data]);

  useEffect(() => {
    if (!states.data) return;
    if (dueStates.length === 0) {
      setQueue([]);
      return;
    }
    setLoading(true);
    // 按 dayKey 分组加载对应的天内容
    const byDay = new Map<string, string[]>();
    for (const s of dueStates) {
      const dayKey = s.cardId.slice(0, 5);
      const list = byDay.get(dayKey) ?? [];
      list.push(s.cardId);
      byDay.set(dayKey, list);
    }
    Promise.all(
      [...byDay.entries()].map(async ([dayKey, ids]) => {
        const doc = await loadDay(dayKey);
        return ids
          .map((id) => {
            const i = Number(id.slice(5));
            const c: CardItem | undefined = doc.cards[i];
            return c ? { cardId: id, front: c.front, back: c.back } : null;
          })
          .filter((x): x is DueCard => !!x);
      }),
    )
      .then((groups) => setQueue(groups.flat()))
      .finally(() => setLoading(false));
  }, [dueStates.length, states.data]);

  const grade = (quality: 1 | 3 | 4 | 5) => {
    const card = queue[idx];
    if (!card) return;
    reviewCard.mutate(
      { cardId: card.cardId, quality },
      { onSuccess: () => utils.study.cardStates.invalidate() },
    );
    setFlipped(false);
    setIdx((i) => i + 1);
  };

  const restart = () => {
    setIdx(0);
    setFlipped(false);
    utils.study.cardStates.invalidate();
  };

  return (
    <AuthLayout>
      <div className="mx-auto max-w-2xl px-2 py-6 md:px-6">
        <div className="flex items-center gap-3">
          <Flame className="h-5 w-5 text-[hsl(var(--warning))]" />
          <h1 className="text-2xl font-bold tracking-tight">闪卡复习</h1>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          基于简化 SM-2 间隔重复：评分决定下次出现时间（重来 10 分钟 / 良好按倍率拉长）。在各天页面给卡片评分后，到期卡片会进入这里的队列。
        </p>

        <div className="mt-8">
          {loading || states.isLoading ? (
            <div className="rounded-xl border border-border bg-card p-10 text-center text-muted-foreground">
              正在加载到期卡片…
            </div>
          ) : dueStates.length === 0 ? (
            <div className="rounded-xl border border-border bg-card p-10 text-center">
              <div className="text-lg font-semibold">当前没有到期卡片</div>
              <p className="mt-2 text-sm text-muted-foreground">
                去每日页面的「闪卡」标签学习新卡片，或等已学卡片到期。
              </p>
              <Link to="/">
                <Button variant="secondary" className="mt-5">
                  返回仪表盘
                </Button>
              </Link>
            </div>
          ) : idx >= queue.length ? (
            <div className="rounded-xl border border-border bg-card p-10 text-center">
              <div className="text-lg font-semibold">
                本轮 {queue.length} 张到期卡片复习完成
              </div>
              <Button variant="secondary" className="mt-5" onClick={restart}>
                <RotateCcw className="mr-1.5 h-4 w-4" />
                检查是否还有到期卡片
              </Button>
            </div>
          ) : (
            <div>
              <div className="mb-3 flex items-center justify-between">
                <span className="font-tech text-xs text-muted-foreground">
                  DUE {idx + 1} / {queue.length}
                </span>
                <span className="font-tech text-[10px] uppercase tracking-widest text-muted-foreground">
                  {queue[idx].cardId.slice(0, 5)}
                </span>
              </div>
              <button
                className="flip-card block w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                onClick={() => setFlipped((f) => !f)}
                aria-label="翻面查看答案"
              >
                <div
                  className={`flip-inner relative min-h-56 w-full ${flipped ? "flipped" : ""}`}
                >
                  <div className="flip-face absolute inset-0 flex flex-col items-center justify-center rounded-xl border border-border bg-card p-8">
                    <div className="font-tech text-[10px] uppercase tracking-widest text-muted-foreground">
                      概念
                    </div>
                    <div className="mt-3 text-center text-xl font-semibold">
                      {queue[idx].front}
                    </div>
                  </div>
                  <div className="flip-face flip-back absolute inset-0 flex flex-col items-center justify-center overflow-y-auto rounded-xl border border-primary/40 bg-primary/5 p-8">
                    <div className="font-tech text-[10px] uppercase tracking-widest text-primary">
                      释义
                    </div>
                    <div className="mt-3 max-w-lg text-center text-sm leading-relaxed">
                      <MarkdownView markdown={queue[idx].back} />
                    </div>
                  </div>
                  <div className="invisible flex min-h-56 flex-col items-center justify-center p-8">
                    <div className="mt-3 text-xl font-semibold">{queue[idx].front}</div>
                    <div className="mt-3 max-w-lg">{queue[idx].back}</div>
                  </div>
                </div>
              </button>
              <div className="mt-4 grid grid-cols-4 gap-2">
                {(
                  [
                    ["重来", 1, "border-destructive/50 text-destructive hover:bg-destructive/10"],
                    ["困难", 3, "border-[hsl(var(--warning))]/50 text-[hsl(var(--warning))] hover:bg-[hsl(var(--warning))]/10"],
                    ["良好", 4, "border-primary/50 text-primary hover:bg-primary/10"],
                    ["简单", 5, "border-[hsl(var(--success))]/50 text-[hsl(var(--success))] hover:bg-[hsl(var(--success))]/10"],
                  ] as const
                ).map(([label, q, cls]) => (
                  <button
                    key={label}
                    disabled={!flipped}
                    onClick={() => grade(q)}
                    className={`rounded-lg border px-2 py-2.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring ${cls}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {!flipped && (
                <p className="mt-2 text-center text-xs text-muted-foreground">
                  点击卡片翻面，然后按掌握程度评分
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </AuthLayout>
  );
}
