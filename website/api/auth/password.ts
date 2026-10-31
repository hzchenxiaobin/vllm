import { randomBytes, scrypt as _scrypt, timingSafeEqual } from "node:crypto";
import { promisify } from "node:util";

// @types/node 未给 promisify 后的 scrypt 声明 options 重载，这里显式补上
const scrypt = promisify(_scrypt) as (
  password: string,
  salt: Buffer,
  keylen: number,
  options: { N: number; r: number; p: number },
) => Promise<Buffer>;

// scrypt 是 memory-hard KDF：N=16384, r=8, p=1 约耗 16MB 内存，
// 对 GPU 并行爆破不友好。格式：s2$N$r$p$salt_b64$hash_b64，自带参数便于日后升级。
const PARAMS = { N: 16384, r: 8, p: 1 } as const;
const KEY_LEN = 64;

export async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16);
  const key = await scrypt(password, salt, KEY_LEN, PARAMS);
  return [
    "s2",
    String(PARAMS.N),
    String(PARAMS.r),
    String(PARAMS.p),
    salt.toString("base64"),
    key.toString("base64"),
  ].join("$");
}

export async function verifyPassword(
  password: string,
  stored: string,
): Promise<boolean> {
  const parts = stored.split("$");
  if (parts.length !== 6 || parts[0] !== "s2") return false;
  const [, n, r, p, saltB64, hashB64] = parts;
  const salt = Buffer.from(saltB64, "base64");
  const expected = Buffer.from(hashB64, "base64");
  if (expected.length === 0) return false;
  try {
    const key = (await scrypt(password, salt, expected.length, {
      N: Number(n),
      r: Number(r),
      p: Number(p),
    })) as Buffer;
    return timingSafeEqual(key, expected);
  } catch {
    return false;
  }
}
