export interface DayMeta {
  key: string;
  title: string;
  quizCount: number;
  cardCount: number;
}
export interface WeekMeta {
  week: number;
  title: string;
  goal: string;
  days: DayMeta[];
}
export interface Manifest {
  siteTitle: string;
  source: string;
  weeks: WeekMeta[];
}
export interface QuizItem {
  q: string;
  a: string;
}
export interface CardItem {
  front: string;
  back: string;
}
export interface DayDoc {
  key: string;
  title: string;
  markdown: string;
  quiz: QuizItem[];
  openQuiz: string[];
  cards: CardItem[];
}

let manifestCache: Promise<Manifest> | null = null;
const dayCache = new Map<string, Promise<DayDoc>>();

export function loadManifest(): Promise<Manifest> {
  if (!manifestCache) {
    manifestCache = fetch("/content/manifest.json").then((r) => {
      if (!r.ok) throw new Error("manifest load failed");
      return r.json();
    });
  }
  return manifestCache;
}

export function loadDay(key: string): Promise<DayDoc> {
  if (!dayCache.has(key)) {
    dayCache.set(
      key,
      fetch(`/content/days/${key}.json`).then((r) => {
        if (!r.ok) throw new Error(`day ${key} load failed`);
        return r.json();
      }),
    );
  }
  return dayCache.get(key)!;
}

/** 闪卡全局 ID：w1d02c3 */
export function cardId(dayKey: string, index: number): string {
  return `${dayKey}c${index}`;
}

export function dayKeyWeek(dayKey: string): number {
  return Number(dayKey.slice(1, dayKey.indexOf("d")));
}

/** 短标题：去掉过长的后缀 */
export function shortTitle(t: string, max = 34): string {
  return t.length > max ? t.slice(0, max) + "…" : t;
}

/** 从 dayKey 推出全局第几天（用于进度展示） */
export function dayLabel(key: string): string {
  const w = key.slice(1, key.indexOf("d"));
  const d = key.slice(key.indexOf("d") + 1);
  return `W${w.padStart(2, "0")} · D${d}`;
}
