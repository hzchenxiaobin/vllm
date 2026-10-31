import AuthLayout from "@/components/AuthLayout";
import MarkdownView from "@/components/MarkdownView";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { dayLabel, loadManifest, type WeekMeta } from "@/lib/content";
import { trpc } from "@/providers/trpc";
import { CheckCircle2, ChevronRight, Circle, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router";

export default function Week() {
  const { week: weekParam } = useParams<{ week: string }>();
  const weekNum = Number(weekParam ?? 0);
  const [week, setWeek] = useState<WeekMeta | null>(null);
  const navigate = useNavigate();
  const progress = trpc.study.progress.useQuery();
  const review = trpc.study.getReview.useQuery({ week: weekNum });
  const saveReview = trpc.study.saveReview.useMutation({
    onSuccess: () => {
      review.refetch();
      progress.refetch();
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
  });
  const [draft, setDraft] = useState("");
  const [saved, setSaved] = useState(false);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    loadManifest().then((m) => {
      setWeek(m.weeks.find((w) => w.week === weekNum) ?? null);
    });
  }, [weekNum]);

  useEffect(() => {
    if (review.data && !dirty) setDraft(review.data.content);
  }, [review.data, dirty]);

  const checked = useMemo(
    () => new Set(progress.data?.checkedDays ?? []),
    [progress.data],
  );
  const quizBest = progress.data?.quizBest ?? {};

  if (!week) {
    return (
      <AuthLayout>
        <div className="p-10 text-muted-foreground">加载中…</div>
      </AuthLayout>
    );
  }

  const doneCount = week.days.filter((d) => checked.has(d.key)).length;

  return (
    <AuthLayout>
      <div className="mx-auto max-w-4xl px-2 py-6 md:px-6">
        <div className="font-tech text-[11px] uppercase tracking-[0.3em] text-primary">
          Week {String(week.week).padStart(2, "0")}
        </div>
        <h1 className="mt-2 text-2xl font-bold tracking-tight">{week.title}</h1>
        {week.goal && (
          <p className="mt-2 text-sm text-muted-foreground">
            <span className="font-medium text-foreground/80">本周目标：</span>
            {week.goal}
          </p>
        )}
        <div className="mt-4 flex items-center gap-3">
          <div className="flex flex-1 gap-1">
            {week.days.map((d) => (
              <div
                key={d.key}
                className={`h-1.5 flex-1 rounded-full ${checked.has(d.key) ? "bg-primary" : "bg-secondary"}`}
              />
            ))}
          </div>
          <span className="font-tech text-xs text-muted-foreground">
            {doneCount}/{week.days.length}
          </span>
        </div>

        {/* 天数列表 */}
        <div className="mt-8 space-y-2">
          {week.days.map((d) => {
            const isDone = checked.has(d.key);
            const best = quizBest[d.key];
            return (
              <button
                key={d.key}
                onClick={() => navigate(`/day/${d.key}`)}
                className="group flex w-full items-center gap-4 rounded-xl border border-border bg-card px-4 py-3.5 text-left transition-colors hover:border-primary/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {isDone ? (
                  <CheckCircle2 className="h-5 w-5 shrink-0 text-[hsl(var(--success))]" />
                ) : (
                  <Circle className="h-5 w-5 shrink-0 text-muted-foreground/50" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="font-tech text-[10px] uppercase tracking-widest text-muted-foreground">
                    {dayLabel(d.key)}
                  </div>
                  <div className="truncate text-sm font-medium">{d.title}</div>
                </div>
                <div className="flex shrink-0 items-center gap-3 text-[11px] text-muted-foreground">
                  {d.cardCount > 0 && <span>{d.cardCount} 卡</span>}
                  {best && (
                    <span
                      className={
                        best.correct / best.total >= 0.8
                          ? "text-[hsl(var(--success))]"
                          : "text-[hsl(var(--warning))]"
                      }
                    >
                      自测 {best.correct}/{best.total}
                    </span>
                  )}
                  <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                </div>
              </button>
            );
          })}
        </div>

        {/* 周复盘 */}
        <section className="mt-12">
          <h2 className="font-tech text-xs uppercase tracking-[0.25em] text-muted-foreground">
            本周复盘
          </h2>
          <div className="mt-3 rounded-xl border border-border bg-card p-5">
            <MarkdownView
              markdown={
                "**复盘模板**：① 本周最核心的 3 个收获是什么？② 自测/闪卡中反复错的点是什么，根因是什么？③ 哪些内容能用自己的话完整讲出来（白板验证）？④ 下周要补的短板？"
              }
            />
            <Textarea
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value);
                setDirty(true);
              }}
              placeholder="写下本周复盘……（保存后云端同步）"
              className="mt-3 min-h-40 bg-background/60"
            />
            <div className="mt-3 flex items-center gap-3">
              <Button
                onClick={() => saveReview.mutate({ week: weekNum, content: draft })}
                disabled={saveReview.isPending || !draft.trim()}
              >
                <Save className="mr-1.5 h-4 w-4" />
                {saved ? "已保存" : "保存复盘"}
              </Button>
              {review.data && (
                <span className="text-xs text-muted-foreground">
                  上次保存：
                  {new Date(review.data.updatedAt).toLocaleString("zh-CN")}
                </span>
              )}
            </div>
          </div>
        </section>
      </div>
    </AuthLayout>
  );
}
