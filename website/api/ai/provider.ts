import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

function readEnv(key: string): string {
  const v = process.env[key];
  if (!v) throw new Error(`missing env ${key}`);
  return v;
}

export const kimiGw = createOpenAICompatible({
  name: "kimi-gw",
  baseURL: readEnv("KIMI_AGENTGW_BASE_URL"),
  apiKey: readEnv("KIMI_AGENTGW_API_KEY"),
  includeUsage: true,
  supportsStructuredOutputs: true,
});
