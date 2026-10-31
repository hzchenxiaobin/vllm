import AuthLayout from "@/components/AuthLayout";
import MarkdownView from "@/components/MarkdownView";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  cardId,
  dayLabel,
  loadDay,
  loadManifest,
  type DayDoc,
} from "@/lib/content";
import { trpc } from "@/providers/trpc";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Eye,
  RotateCcw,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";

export default function Day() {
  const { dayKey } = useParams<{ dayKey: string }>();
  const [doc, setDoc] = useState<DayDoc | null>(null);
  const [nav, setNav] = useState<{ prev?: string; next?: string }>({});
  const navigate = useNavigate();
  const progress = trpc.study.progress.useQuery();
  const checkin = trpc.study.checkin.useMutation({
    onSuccess: () => progress.refetch(),
  });
  const uncheckin = trpc.study.uncheckin.useMutation({
    onSuccess: () => progress.refetch(),
  });

  useEffect(() => {
    if (!dayKey) return;
    setDoc(null);
    loadDay(dayKey).then(setDoc).catch(() => {});
    loadManifest().then((m) => {
      const all = m.weeks.flatMap((w) => w.days.map((d) => d.key));
      const idx = all.indexOf(dayKey);
      setNav({
        prev: idx > 0 ? all[idx - 1] : undefined,
        next: idx >= 0 && idx < all.length - 1 ? all[idx + 1] : undefined,
      });
    });
  }, [dayKey]);

  const checked = progress.data?.checkedDays.includes(dayKey ?? "") ?? false;

  if (!doc || !dayKey) {
    return (
      <AuthLayout>
        <div className="p-10 text-muted-foreground">加载中…</div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout>
      <div className="mx-auto max-w-4xl px-2 py-6 md:px-6">
        {/* 头部 */}
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="font-tech text-[11px] uppercase tracking-[0.3em] text-primary">
              {dayLabel(dayKey)}
            </div>
            <h1 className="mt-1.5 text-xl font-bold leading-snug tracking-tight md:text-2xl">
              {doc.title}
            </h1>
          </div>
          <Button
            variant={checked ? "secondary" : "default"}
            onClick={() =>
              checked
                ? uncheckin.mutate({ dayKey })
                : checkin.mutate({ dayKey })
            }
            disabled={checkin.isPending || uncheckin.isPending}
          >
            <Check className="mr-1.5 h-4 w-4" />
            {checked ? "已打卡（点击取消）" : "完成打卡"}
          </Button>
        </div>

        {/* 学习闭环三件套 */}
        <Tabs defaultValue="notes" className="mt-6">
          <TabsList className="w-full justify-start md:w-auto">
            <TabsTrigger value="notes">读笔记</TabsTrigger>
            <TabsTrigger value="quiz">
              自测{doc.quiz.length + doc.openQuiz.length > 0 && ` · ${doc.quiz.length + doc.openQuiz.length}`}
            </TabsTrigger>
            <TabsTrigger value="cards">
              闪卡{doc.cards.length > 0 && ` · ${doc.cards.length}`}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="notes" className="mt-6">
            <MarkdownView markdown={doc.markdown} />
          </TabsContent>

          <TabsContent value="quiz" className="mt-6">
            <QuizPanel doc={doc} />
          </TabsContent>

          <TabsContent value="cards" className="mt-6">
            <CardPanel doc={doc} />
          </TabsContent>
        </Tabs>

        {/* 上/下一天 */}
        <div className="mt-12 flex items-center justify-between border-t border-border pt-5">
          {nav.prev ? (
            <button
              onClick={() => navigate(`/day/${nav.prev}`)}
              className="flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <ChevronLeft className="h-4 w-4" />
              {dayLabel(nav.prev)}
            </button>
          ) : (
            <span />
          )}
          {nav.next ? (
            <button
              onClick={() => navigate(`/day/${nav.next}`)}
              className="flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {dayLabel(nav.next)}
              <ChevronRight className="h-4 w-4" />
            </button>
          ) : (
            <span />
          )}
        </div>
      </div>
    </AuthLayout>
  );
}

/* ---------------- 自测面板 ---------------- */

function QuizPanel({ doc }: { doc: DayDoc }) {
  const recordQuiz = trpc.study.recordQuiz.useMutation();
  const [idx, setIdx] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [results, setResults] = useState<boolean[]>([]);
  const [finished, setFinished] = useState(false);

  // 题目队列：解析题（带答案） + 开放题（自评）
  const queue = useMemo(
    () => [
      ...doc.quiz.map((q) => ({ kind: "qa" as const, q: q.q, a: q.a })),
      ...doc.openQuiz.map((q) => ({ kind: "open" as const, q, a: "" })),
    ],
    [doc],
  );

  useEffect(() => {
    setIdx(0);
    setRevealed(false);
    setResults([]);
    setFinished(false);
  }, [doc.key]);

  if (queue.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center text-sm text-muted-foreground">
        本日没有内置自测题，请按笔记中的「产出物」要求自查。
      </div>
    );
  }

  if (finished) {
    const correct = results.filter(Boolean).length;
    const ratio = correct / results.length;
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center">
        <div className="font-tech text-[11px] uppercase tracking-widest text-muted-foreground">
          自测完成
        </div>
        <div className="mt-3 text-4xl font-bold">
          {correct}
          <span className="text-xl text-muted-foreground">/{results.length}</span>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          {ratio >= 0.8
            ? "掌握良好，可以进入下一天。"
            : ratio >= 0.5
              ? "基本掌握，建议把错题加入闪卡复习。"
              : "掌握度偏低，建议重读笔记后再测一次。"}
        </p>
        <Button
          variant="secondary"
          className="mt-5"
          onClick={() => {
            setIdx(0);
            setRevealed(false);
            setResults([]);
            setFinished(false);
          }}
        >
          <RotateCcw className="mr-1.5 h-4 w-4" />
          再测一次
        </Button>
      </div>
    );
  }

  const item = queue[idx];
  const answer = (ok: boolean) => {
    const next = [...results, ok];
    setResults(next);
    if (idx + 1 >= queue.length) {
      recordQuiz.mutate({
        dayKey: doc.key,
        correct: next.filter(Boolean).length,
        total: next.length,
      });
      setFinished(true);
    } else {
      setIdx(idx + 1);
      setRevealed(false);
    }
  };

  return (
    <div className="rounded-xl border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-5 py-3">
        <span className="font-tech text-xs text-muted-foreground">
          Q{idx + 1} / {queue.length}
          {item.kind === "open" && (
            <span className="ml-2 rounded bg-secondary px-1.5 py-0.5 text-[10px]">
              开放题·自评
            </span>
          )}
        </span>
        <div className="flex gap-1">
          {queue.map((_, i) => (
            <div
              key={i}
              className={`h-1 w-4 rounded-full ${
                i < results.length
                  ? results[i]
                    ? "bg-[hsl(var(--success))]"
                    : "bg-destructive"
                  : i === idx
                    ? "bg-primary"
                    : "bg-secondary"
              }`}
            />
          ))}
        </div>
      </div>
      <div className="px-5 py-5">
        <MarkdownView markdown={item.q} />
        {item.kind === "qa" && !revealed && (
          <Button variant="secondary" className="mt-4" onClick={() => setRevealed(true)}>
            <Eye className="mr-1.5 h-4 w-4" />
            先自己作答，再查看参考答案
          </Button>
        )}
        {(item.kind === "open" || revealed) && (
          <>
            {item.a && (
              <div className="mt-4 rounded-lg border border-primary/30 bg-primary/5 p-4">
                <div className="mb-2 font-tech text-[10px] uppercase tracking-widest text-primary">
                  参考答案
                </div>
                <MarkdownView markdown={item.a} />
              </div>
            )}
            <div className="mt-5 flex gap-3">
              <Button
                className="flex-1 bg-[hsl(var(--success))] hover:bg-[hsl(var(--success))]/90"
                onClick={() => answer(true)}
              >
                答对了
              </Button>
              <Button
                variant="destructive"
                className="flex-1"
                onClick={() => answer(false)}
              >
                {item.kind === "open" ? "没答上来" : "答错了"}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ---------------- 闪卡面板 ---------------- */

function CardPanel({ doc }: { doc: DayDoc }) {
  const reviewCard = trpc.study.reviewCard.useMutation();
  const [idx, setIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [, setDone] = useState(0);

  useEffect(() => {
    setIdx(0);
    setFlipped(false);
    setDone(0);
  }, [doc.key]);

  if (doc.cards.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center text-sm text-muted-foreground">
        本日没有可提取的概念卡。
      </div>
    );
  }

  const finishedAll = idx >= doc.cards.length;
  const card = doc.cards[Math.min(idx, doc.cards.length - 1)];

  const grade = (quality: 1 | 3 | 4 | 5) => {
    reviewCard.mutate({ cardId: cardId(doc.key, idx), quality });
    setDone((d) => d + 1);
    setFlipped(false);
    setTimeout(() => setIdx((i) => i + 1), 120);
  };

  if (finishedAll) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center">
        <div className="text-lg font-semibold">本日 {doc.cards.length} 张卡片已过完</div>
        <p className="mt-2 text-sm text-muted-foreground">
          评分已记入间隔重复队列，到期后会在「闪卡复习」中出现。
        </p>
        <Button
          variant="secondary"
          className="mt-5"
          onClick={() => {
            setIdx(0);
            setFlipped(false);
            setDone(0);
          }}
        >
          <RotateCcw className="mr-1.5 h-4 w-4" />
          重新过一遍
        </Button>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <span className="font-tech text-xs text-muted-foreground">
          CARD {idx + 1} / {doc.cards.length}
        </span>
        <span className="text-xs text-muted-foreground">点击卡片翻面</span>
      </div>
      <button
        className="flip-card block w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={() => setFlipped((f) => !f)}
        aria-label="翻面查看答案"
      >
        <div className={`flip-inner relative min-h-56 w-full ${flipped ? "flipped" : ""}`}>
          <div className="flip-face absolute inset-0 flex flex-col items-center justify-center rounded-xl border border-border bg-card p-8">
            <div className="font-tech text-[10px] uppercase tracking-widest text-muted-foreground">
              概念
            </div>
            <div className="mt-3 text-center text-xl font-semibold">{card.front}</div>
          </div>
          <div className="flip-face flip-back absolute inset-0 flex flex-col items-center justify-center overflow-y-auto rounded-xl border border-primary/40 bg-primary/5 p-8">
            <div className="font-tech text-[10px] uppercase tracking-widest text-primary">
              释义
            </div>
            <div className="mt-3 max-w-lg text-center text-sm leading-relaxed [&_.md-body]:text-sm">
              <MarkdownView markdown={card.back} />
            </div>
          </div>
          {/* 占位保持高度 */}
          <div className="invisible flex min-h-56 flex-col items-center justify-center p-8">
            <div className="mt-3 text-xl font-semibold">{card.front}</div>
            <div className="mt-3 max-w-lg">{card.back}</div>
          </div>
        </div>
      </button>
      <div className="mt-4 grid grid-cols-4 gap-2">
        <GradeBtn label="重来" hint="10分钟后" onClick={() => grade(1)} disabled={!flipped} tone="destructive" />
        <GradeBtn label="困难" hint="短期" onClick={() => grade(3)} disabled={!flipped} tone="warning" />
        <GradeBtn label="良好" hint="正常" onClick={() => grade(4)} disabled={!flipped} tone="primary" />
        <GradeBtn label="简单" hint="拉长间隔" onClick={() => grade(5)} disabled={!flipped} tone="success" />
      </div>
      {!flipped && (
        <p className="mt-2 text-center text-xs text-muted-foreground">
          先翻面看释义，再按掌握程度评分
        </p>
      )}
    </div>
  );
}

function GradeBtn({
  label,
  hint,
  onClick,
  disabled,
  tone,
}: {
  label: string;
  hint: string;
  onClick: () => void;
  disabled: boolean;
  tone: "destructive" | "warning" | "primary" | "success";
}) {
  const color =
    tone === "destructive"
      ? "border-destructive/50 text-destructive hover:bg-destructive/10"
      : tone === "warning"
        ? "border-[hsl(var(--warning))]/50 text-[hsl(var(--warning))] hover:bg-[hsl(var(--warning))]/10"
        : tone === "success"
          ? "border-[hsl(var(--success))]/50 text-[hsl(var(--success))] hover:bg-[hsl(var(--success))]/10"
          : "border-primary/50 text-primary hover:bg-primary/10";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg border px-2 py-2.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring ${color}`}
    >
      <div>{label}</div>
      <div className="mt-0.5 text-[10px] font-normal opacity-70">{hint}</div>
    </button>
  );
}
