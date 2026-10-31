import AuthLayout from "@/components/AuthLayout";
import { dayLabel, loadManifest, type Manifest } from "@/lib/content";
import { trpc } from "@/providers/trpc";
import {
  ArrowRight,
  CalendarCheck2,
  Flame,
  Layers,
  MessageSquareText,
  PenLine,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";

function weekShortTitle(t: string) {
  return t
    .replace(/^(第\s*\d+\s*周|Week\s*\d+)\s*[：:]\s*/, "")
    .replace(/（([^）]{5,})）/g, ""); // 只去长括号说明，保留（一）（二）（上）等序号
}

export default function Home() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const progress = trpc.study.progress.useQuery();
  const navigate = useNavigate();

  useEffect(() => {
    loadManifest().then(setManifest).catch(() => {});
  }, []);

  const stats = useMemo(() => {
    if (!manifest || !progress.data) return null;
    const totalDays = manifest.weeks.reduce((s, w) => s + w.days.length, 0);
    const checked = new Set(progress.data.checkedDays);
    return {
      totalDays,
      doneDays: manifest.weeks
        .flatMap((w) => w.days)
        .filter((d) => checked.has(d.key)).length,
      streak: progress.data.streak,
      dueCards: progress.data.dueCards,
      reviewedWeeks: progress.data.reviewedWeeks.length,
      totalWeeks: manifest.weeks.length,
    };
  }, [manifest, progress.data]);

  // 下一个未完成的一天
  const nextDay = useMemo(() => {
    if (!manifest || !progress.data) return null;
    const checked = new Set(progress.data.checkedDays);
    for (const w of manifest.weeks) {
      for (const d of w.days) {
        if (!checked.has(d.key)) return { ...d, week: w.week };
      }
    }
    return null;
  }, [manifest, progress.data]);

  const pct = stats ? Math.round((stats.doneDays / stats.totalDays) * 100) : 0;

  return (
    <AuthLayout>
      <div className="mx-auto max-w-5xl px-2 py-6 md:px-6">
        {/* 顶部概览 */}
        <header className="mb-8">
          <div className="font-tech text-[11px] uppercase tracking-[0.3em] text-primary">
            LLM Inference System · 8-Week Camp
          </div>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">
            vLLM 推理系统优化 · 学习训练营
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            学习闭环：读笔记 → 打卡 → 自测 → 闪卡间隔复习 → 每周复盘 → AI 模拟面试。进度云端同步，多端一致。
          </p>
        </header>

        {/* 指标条 */}
        <div className="mb-10 grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatBlock
            icon={<CalendarCheck2 className="h-4 w-4" />}
            label="总进度"
            value={stats ? `${stats.doneDays}/${stats.totalDays}` : "—"}
            sub={`${pct}% 已打卡`}
          />
          <StatBlock
            icon={<Flame className="h-4 w-4" />}
            label="连续学习"
            value={stats ? `${stats.streak} 天` : "—"}
            sub="保持打卡节奏"
            accent="text-[hsl(var(--warning))]"
          />
          <StatBlock
            icon={<Layers className="h-4 w-4" />}
            label="待复习闪卡"
            value={stats ? String(stats.dueCards) : "—"}
            sub="到期卡片，点击复习"
            onClick={() => navigate("/review")}
          />
          <StatBlock
            icon={<PenLine className="h-4 w-4" />}
            label="周复盘"
            value={stats ? `${stats.reviewedWeeks}/${stats.totalWeeks}` : "—"}
            sub="已完成的周复盘"
          />
        </div>

        {/* 继续学习 */}
        {nextDay && (
          <button
            onClick={() => navigate(`/day/${nextDay.key}`)}
            className="group mb-10 flex w-full items-center justify-between rounded-xl border border-primary/40 bg-primary/10 px-5 py-4 text-left transition-colors hover:bg-primary/15 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <div>
              <div className="font-tech text-[11px] uppercase tracking-widest text-primary">
                继续学习 · {dayLabel(nextDay.key)}
              </div>
              <div className="mt-1 font-medium">{nextDay.title}</div>
            </div>
            <ArrowRight className="h-5 w-5 text-primary transition-transform group-hover:translate-x-1" />
          </button>
        )}
        {stats && stats.doneDays === stats.totalDays && (
          <div className="mb-10 rounded-xl border border-[hsl(var(--success))]/40 bg-[hsl(var(--success))]/10 px-5 py-4 text-sm">
            全部 {stats.totalDays} 天已完成打卡——去「模拟面试」做最后冲刺。
          </div>
        )}

        {/* 周路线图 */}
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-tech text-xs uppercase tracking-[0.25em] text-muted-foreground">
            学习路线 · {manifest?.weeks.length ?? 10} 周
          </h2>
          <Link
            to="/interview"
            className="flex items-center gap-1.5 text-xs text-primary hover:underline"
          >
            <MessageSquareText className="h-3.5 w-3.5" />
            直接开始模拟面试
          </Link>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {manifest?.weeks.map((w) => (
            <WeekCard
              key={w.week}
              week={w.week}
              title={weekShortTitle(w.title)}
              goal={w.goal}
              days={w.days.map((d) => d.key)}
              checkedDays={progress.data?.checkedDays ?? []}
              quizBest={progress.data?.quizBest ?? {}}
            />
          ))}
        </div>

        <footer className="mt-10 border-t border-border pt-4 text-xs text-muted-foreground">
          内容来源：
          <a
            href={manifest?.source ?? "https://github.com/hzchenxiaobin/vllm"}
            target="_blank"
            rel="noreferrer"
            className="text-primary hover:underline"
          >
            github.com/hzchenxiaobin/vllm
          </a>
          · 内置 {manifest?.weeks.reduce((s, w) => s + w.days.length, 0) ?? 71} 篇笔记 /
          539 道自测题 / 459 张闪卡
        </footer>
      </div>
    </AuthLayout>
  );
}

function StatBlock({
  icon,
  label,
  value,
  sub,
  accent,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  accent?: string;
  onClick?: () => void;
}) {
  const inner = (
    <>
      <div className="flex items-center gap-2 text-muted-foreground">
        {icon}
        <span className="font-tech text-[10px] uppercase tracking-widest">{label}</span>
      </div>
      <div className={`mt-2 text-2xl font-bold tracking-tight ${accent ?? ""}`}>
        {value}
      </div>
      <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>
    </>
  );
  const cls =
    "rounded-xl border border-border bg-card px-4 py-4 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring";
  return onClick ? (
    <button onClick={onClick} className={`${cls} hover:border-primary/50 cursor-pointer`}>
      {inner}
    </button>
  ) : (
    <div className={cls}>{inner}</div>
  );
}

function WeekCard({
  week,
  title,
  goal,
  days,
  checkedDays,
  quizBest,
}: {
  week: number;
  title: string;
  goal: string;
  days: string[];
  checkedDays: string[];
  quizBest: Record<string, { correct: number; total: number }>;
}) {
  const navigate = useNavigate();
  const checked = new Set(checkedDays);
  const done = days.filter((d) => checked.has(d)).length;
  const quizDays = days.filter((d) => quizBest[d]);
  return (
    <button
      onClick={() => navigate(`/week/${week}`)}
      className="group rounded-xl border border-border bg-card p-5 text-left transition-colors hover:border-primary/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-tech text-xs text-primary">
          W{String(week).padStart(2, "0")}
        </span>
        <span className="font-tech text-[11px] text-muted-foreground">
          {done}/{days.length} 天
        </span>
      </div>
      <h3 className="mt-1.5 font-semibold leading-snug">{title}</h3>
      {goal && (
        <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{goal}</p>
      )}
      {/* 分段进度条 */}
      <div className="mt-4 flex gap-1">
        {days.map((d) => (
          <div
            key={d}
            className={`h-1.5 flex-1 rounded-full transition-colors ${
              checked.has(d) ? "bg-primary" : "bg-secondary"
            }`}
          />
        ))}
      </div>
      <div className="mt-2 text-[11px] text-muted-foreground">
        {quizDays.length > 0
          ? `已完成 ${quizDays.length} 天自测`
          : "尚未开始自测"}
      </div>
    </button>
  );
}
