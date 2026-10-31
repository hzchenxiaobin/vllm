import {
  mysqlTable,
  mysqlEnum,
  serial,
  varchar,
  text,
  timestamp,
  bigint,
  int,
  double,
  uniqueIndex,
} from "drizzle-orm/mysql-core";

export const users = mysqlTable("users", {
  id: serial("id").primaryKey(),
  unionId: varchar("unionId", { length: 255 }).notNull().unique(),
  name: varchar("name", { length: 255 }),
  email: varchar("email", { length: 320 }),
  avatar: text("avatar"),
  // ---- 邮箱账号体系 ----
  // 邮箱注册用户的 unionId 形如 "email:<邮箱小写>"，与 Kimi OAuth 用户共用一套会话
  passwordHash: varchar("passwordHash", { length: 255 }),
  emailVerifiedAt: timestamp("emailVerifiedAt"),
  // 改密/重置密码时 +1，旧 JWT 因 sv 不匹配立即失效
  sessionVersion: int("sessionVersion").notNull().default(0),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt")
    .defaultNow()
    .notNull()
    .$onUpdate(() => new Date()),
  lastSignInAt: timestamp("lastSignInAt").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

// ---------- 学习闭环数据表 ----------

/** 邮件验证码（注册验证 / 找回密码共用，只存 sha256 哈希） */
export const emailTokens = mysqlTable("email_tokens", {
  id: serial("id").primaryKey(),
  userId: bigint("userId", { mode: "number", unsigned: true })
    .notNull()
    .references(() => users.id),
  tokenHash: varchar("tokenHash", { length: 64 }).notNull(),
  purpose: mysqlEnum("purpose", ["verify", "reset"]).notNull(),
  /** 连续输错 5 次作废 */
  attempts: int("attempts").notNull().default(0),
  expiresAt: timestamp("expiresAt").notNull(),
  usedAt: timestamp("usedAt"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

/** 每日打卡 */
export const checkins = mysqlTable(
  "checkins",
  {
    id: serial("id").primaryKey(),
    userId: bigint("userId", { mode: "number", unsigned: true })
      .notNull()
      .references(() => users.id),
    dayKey: varchar("dayKey", { length: 32 }).notNull(),
    createdAt: timestamp("createdAt").defaultNow().notNull(),
  },
  (t) => [uniqueIndex("uq_checkin_user_day").on(t.userId, t.dayKey)],
);

/** 自测成绩记录 */
export const quizAttempts = mysqlTable("quiz_attempts", {
  id: serial("id").primaryKey(),
  userId: bigint("userId", { mode: "number", unsigned: true })
    .notNull()
    .references(() => users.id),
  dayKey: varchar("dayKey", { length: 32 }).notNull(),
  correct: int("correct").notNull(),
  total: int("total").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

/** 闪卡间隔重复状态（简化 SM-2） */
export const cardStates = mysqlTable(
  "card_states",
  {
    id: serial("id").primaryKey(),
    userId: bigint("userId", { mode: "number", unsigned: true })
      .notNull()
      .references(() => users.id),
    cardId: varchar("cardId", { length: 64 }).notNull(),
    reps: int("reps").notNull().default(0),
    lapses: int("lapses").notNull().default(0),
    ease: double("ease").notNull().default(2.5),
    intervalDays: double("intervalDays").notNull().default(0),
    dueAt: timestamp("dueAt").defaultNow().notNull(),
    updatedAt: timestamp("updatedAt")
      .defaultNow()
      .notNull()
      .$onUpdate(() => new Date()),
  },
  (t) => [uniqueIndex("uq_card_user").on(t.userId, t.cardId)],
);

/** 每周复盘 */
export const weekReviews = mysqlTable(
  "week_reviews",
  {
    id: serial("id").primaryKey(),
    userId: bigint("userId", { mode: "number", unsigned: true })
      .notNull()
      .references(() => users.id),
    week: int("week").notNull(),
    content: text("content").notNull(),
    updatedAt: timestamp("updatedAt")
      .defaultNow()
      .notNull()
      .$onUpdate(() => new Date()),
  },
  (t) => [uniqueIndex("uq_review_user_week").on(t.userId, t.week)],
);

/** 模拟面试会话 */
export const interviewSessions = mysqlTable("interview_sessions", {
  id: serial("id").primaryKey(),
  userId: bigint("userId", { mode: "number", unsigned: true })
    .notNull()
    .references(() => users.id),
  week: int("week").notNull(),
  title: varchar("title", { length: 255 }).notNull(),
  messages: text("messages").notNull(), // JSON string
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});
