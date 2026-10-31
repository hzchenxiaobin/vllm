import { getDb } from "./queries/connection";
import { authedQuery, createRouter } from "./middleware";
import {
  cardStates,
  checkins,
  interviewSessions,
  quizAttempts,
  weekReviews,
} from "@db/schema";
import { and, desc, eq, lte } from "drizzle-orm";
import { z } from "zod";

const dayKeySchema = z.string().regex(/^w\d+d\d{2}$/);

/** 计算连续打卡天数（从今天或昨天往前数） */
function computeStreak(dates: string[]): number {
  const set = new Set(dates);
  const day = new Date();
  const fmt = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  // 今天没打卡则从昨天起算
  if (!set.has(fmt(day))) day.setDate(day.getDate() - 1);
  let streak = 0;
  while (set.has(fmt(day))) {
    streak += 1;
    day.setDate(day.getDate() - 1);
  }
  return streak;
}

export const studyRouter = createRouter({
  /** 总进度：打卡、最佳成绩、复盘完成周、连续天数、到期闪卡数 */
  progress: authedQuery.query(async ({ ctx }) => {
    const db = getDb();
    const uid = ctx.user.id;
    const [ci, qa, rv, due] = await Promise.all([
      db.select().from(checkins).where(eq(checkins.userId, uid)),
      db.select().from(quizAttempts).where(eq(quizAttempts.userId, uid)),
      db.select().from(weekReviews).where(eq(weekReviews.userId, uid)),
      db
        .select()
        .from(cardStates)
        .where(and(eq(cardStates.userId, uid), lte(cardStates.dueAt, new Date()))),
    ]);
    const quizBest: Record<string, { correct: number; total: number }> = {};
    for (const a of qa) {
      const prev = quizBest[a.dayKey];
      const ratio = a.total > 0 ? a.correct / a.total : 0;
      if (!prev || ratio > prev.correct / Math.max(prev.total, 1)) {
        quizBest[a.dayKey] = { correct: a.correct, total: a.total };
      }
    }
    const dates = ci.map((c) => {
      const d = new Date(c.createdAt);
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    });
    return {
      checkedDays: ci.map((c) => c.dayKey),
      checkinDates: [...new Set(dates)],
      streak: computeStreak([...new Set(dates)]),
      quizBest,
      reviewedWeeks: rv.map((r) => r.week),
      dueCards: due.length,
    };
  }),

  checkin: authedQuery
    .input(z.object({ dayKey: dayKeySchema }))
    .mutation(async ({ ctx, input }) => {
      const db = getDb();
      await db
        .insert(checkins)
        .values({ userId: ctx.user.id, dayKey: input.dayKey })
        .onDuplicateKeyUpdate({ set: { dayKey: input.dayKey } });
      return { ok: true };
    }),

  uncheckin: authedQuery
    .input(z.object({ dayKey: dayKeySchema }))
    .mutation(async ({ ctx, input }) => {
      const db = getDb();
      await db
        .delete(checkins)
        .where(
          and(eq(checkins.userId, ctx.user.id), eq(checkins.dayKey, input.dayKey)),
        );
      return { ok: true };
    }),

  recordQuiz: authedQuery
    .input(
      z.object({
        dayKey: dayKeySchema,
        correct: z.number().int().min(0),
        total: z.number().int().min(1),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const db = getDb();
      await db.insert(quizAttempts).values({
        userId: ctx.user.id,
        dayKey: input.dayKey,
        correct: input.correct,
        total: input.total,
      });
      return { ok: true };
    }),

  /** 全量闪卡状态（前端据此算出到期队列） */
  cardStates: authedQuery.query(async ({ ctx }) => {
    const db = getDb();
    return db
      .select()
      .from(cardStates)
      .where(eq(cardStates.userId, ctx.user.id));
  }),

  /** 闪卡评分：quality 1=重来 3=困难 4=良好 5=简单（简化 SM-2） */
  reviewCard: authedQuery
    .input(
      z.object({
        cardId: z.string().min(1).max(64),
        quality: z.union([
          z.literal(1),
          z.literal(3),
          z.literal(4),
          z.literal(5),
        ]),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const db = getDb();
      const uid = ctx.user.id;
      const existing = await db
        .select()
        .from(cardStates)
        .where(and(eq(cardStates.userId, uid), eq(cardStates.cardId, input.cardId)))
        .limit(1);
      const cur = existing[0];
      const q = input.quality;
      let reps = cur?.reps ?? 0;
      let lapses = cur?.lapses ?? 0;
      let ease = cur?.ease ?? 2.5;
      let interval = cur?.intervalDays ?? 0;
      let dueAt: Date;
      if (q < 3) {
        reps = 0;
        lapses += 1;
        ease = Math.max(1.3, ease - 0.2);
        interval = 0;
        dueAt = new Date(Date.now() + 10 * 60 * 1000); // 10 分钟后重来
      } else {
        reps += 1;
        interval = reps === 1 ? 1 : reps === 2 ? 6 : Math.round(interval * ease);
        ease = Math.max(
          1.3,
          ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)),
        );
        dueAt = new Date(Date.now() + interval * 24 * 60 * 60 * 1000);
      }
      const values = { reps, lapses, ease, intervalDays: interval, dueAt };
      if (cur) {
        await db
          .update(cardStates)
          .set(values)
          .where(eq(cardStates.id, cur.id));
      } else {
        await db.insert(cardStates).values({ userId: uid, cardId: input.cardId, ...values });
      }
      return { ok: true, ...values };
    }),

  getReview: authedQuery
    .input(z.object({ week: z.number().int().min(0).max(9) }))
    .query(async ({ ctx, input }) => {
      const db = getDb();
      const rows = await db
        .select()
        .from(weekReviews)
        .where(
          and(eq(weekReviews.userId, ctx.user.id), eq(weekReviews.week, input.week)),
        )
        .limit(1);
      return rows[0] ?? null;
    }),

  saveReview: authedQuery
    .input(
      z.object({
        week: z.number().int().min(0).max(9),
        content: z.string().max(20000),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const db = getDb();
      await db
        .insert(weekReviews)
        .values({ userId: ctx.user.id, week: input.week, content: input.content })
        .onDuplicateKeyUpdate({ set: { content: input.content } });
      return { ok: true };
    }),

  saveInterview: authedQuery
    .input(
      z.object({
        week: z.number().int().min(0).max(9),
        title: z.string().max(255),
        messages: z.string().max(100000),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const db = getDb();
      await db.insert(interviewSessions).values({
        userId: ctx.user.id,
        week: input.week,
        title: input.title,
        messages: input.messages,
      });
      return { ok: true };
    }),

  listInterviews: authedQuery.query(async ({ ctx }) => {
    const db = getDb();
    const rows = await db
      .select({
        id: interviewSessions.id,
        week: interviewSessions.week,
        title: interviewSessions.title,
        createdAt: interviewSessions.createdAt,
      })
      .from(interviewSessions)
      .where(eq(interviewSessions.userId, ctx.user.id))
      .orderBy(desc(interviewSessions.createdAt))
      .limit(20);
    return rows;
  }),
});
