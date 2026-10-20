/** Anything shaped like a chat message from OpenAI, Anthropic, the AI SDK, Gemini-style wrappers and so on. */
export interface ChatMessage {
  role: string;
  content?: unknown;
  [key: string]: unknown;
}

export interface CountOptions {
  /** Counts the tokens in a string. Default: a fast estimate (see `estimateTokens`). */
  countTokens?: (text: string) => number;
  /** Tokens added per message for role and formatting. Default `4`. */
  messageOverhead?: number;
  /** Tokens charged for each image or file part. Default `1000`. */
  imageTokens?: number;
}

const CJK = /[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]/g;

/**
 * Estimates tokens without a tokenizer: about 4 characters per token for
 * English and code, 1 per character for Chinese, Japanese and Korean.
 * Usually within 10-15% of real tokenizers; pass `countTokens` for exact counts.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  const cjk = text.match(CJK)?.length ?? 0;
  return Math.ceil((text.length - cjk) / 4 + cjk);
}

const MEDIA = new Set(["image", "image_url", "input_image", "file", "input_file", "document", "audio", "input_audio"]);

function partTokens(part: unknown, count: (t: string) => number, imageTokens: number): number {
  if (typeof part === "string") return count(part);
  if (!part || typeof part !== "object") return 0;
  const p = part as Record<string, unknown>;
  const type = typeof p.type === "string" ? p.type : "";
  if (MEDIA.has(type)) return imageTokens;
  if (typeof p.text === "string") return count(p.text);
  // Anthropic tool_result, AI SDK tool-result: content or output can be nested parts.
  if (type === "tool_result" || type === "tool-result") {
    const body = p.content ?? p.output ?? p.result;
    if (typeof body === "string") return count(body);
    if (Array.isArray(body)) return sumParts(body, count, imageTokens);
    return count(JSON.stringify(body ?? ""));
  }
  if (type === "tool_use" || type === "tool-call" || type === "function_call") {
    const args = p.input ?? p.args ?? p.arguments ?? {};
    return count(String(p.name ?? p.toolName ?? "")) + count(typeof args === "string" ? args : JSON.stringify(args));
  }
  return count(JSON.stringify(p));
}

function sumParts(parts: unknown[], count: (t: string) => number, imageTokens: number): number {
  let n = 0;
  for (const part of parts) n += partTokens(part, count, imageTokens);
  return n;
}

/** Tokens one message takes, including per-message overhead. */
export function countMessageTokens(message: ChatMessage, options: CountOptions = {}): number {
  const count = options.countTokens ?? estimateTokens;
  const imageTokens = options.imageTokens ?? 1000;
  let n = options.messageOverhead ?? 4;
  const { content } = message;
  if (typeof content === "string") n += count(content);
  else if (Array.isArray(content)) n += sumParts(content, count, imageTokens);
  else if (content != null) n += partTokens(content, count, imageTokens);
  // OpenAI tool calls on assistant messages.
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls as Array<{ function?: { name?: string; arguments?: string } }>) {
      n += count(call.function?.name ?? "") + count(call.function?.arguments ?? "") + 3;
    }
  }
  if (typeof message.name === "string") n += count(message.name);
  return n;
}

/** Tokens a list of messages takes. */
export function countTokens(messages: readonly ChatMessage[], options: CountOptions = {}): number {
  let n = 0;
  for (const m of messages) n += countMessageTokens(m, options);
  return n;
}
