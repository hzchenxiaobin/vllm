import { AsyncLocalStorage } from "async_hooks";

export const REQUEST_ID_HEADER = "x-request-id";

const storage = new AsyncLocalStorage<string>();

// runWithRequestId scopes `fn` to a request id. The Hono middleware wraps every
// request in this, so anything logged inside — including after awaits — picks
// the id up via currentRequestId().
export function runWithRequestId<T>(requestId: string, fn: () => T): T {
  return storage.run(requestId, fn);
}

// currentRequestId returns the id of the request in flight, or undefined outside
// any request scope (startup logging, a timer callback, a client-log write).
export function currentRequestId(): string | undefined {
  return storage.getStore();
}
