/**
 * 进程内滑动窗口限流。单实例部署（本站形态）足够；
 * 多实例水平扩容时需换成 Redis 等共享存储。
 */
const buckets = new Map<string, number[]>();

export function rateLimitOk(
  key: string,
  limit: number,
  windowMs: number,
): boolean {
  const now = Date.now();
  const hits = (buckets.get(key) ?? []).filter((t) => now - t < windowMs);
  if (hits.length >= limit) {
    buckets.set(key, hits);
    return false;
  }
  hits.push(now);
  buckets.set(key, hits);
  return true;
}

// 定期清理空桶，避免 Map 无限增长
const sweeper = setInterval(() => {
  const now = Date.now();
  for (const [key, hits] of buckets) {
    const alive = hits.filter((t) => now - t < 60 * 60 * 1000);
    if (alive.length === 0) buckets.delete(key);
    else buckets.set(key, alive);
  }
}, 10 * 60 * 1000);
sweeper.unref?.();
