import { authRouter } from "./auth-router";
import { createRouter, publicQuery } from "./middleware";
import { studyRouter } from "./study-router";

export const appRouter = createRouter({
  ping: publicQuery.query(() => ({ ok: true, ts: Date.now() })),
  auth: authRouter,
  study: studyRouter,
});

export type AppRouter = typeof appRouter;
