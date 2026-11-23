import { countMessageTokens, estimateTokens, type ChatMessage, type CountOptions } from "./tokens.js";

export interface FitOptions<M extends ChatMessage = ChatMessage> extends CountOptions {
  /** The model's context window (or any budget) in tokens. */
  maxTokens: number;
  /** Tokens to leave free for the model's reply. Default `0`. */
  reserve?: number;
  /** A system prompt passed separately (like Anthropic's `system`), counted against the budget. */
  system?: unknown;
  /** Never drop `system` / `developer` messages. Default `true`. */
  keepSystem?: boolean;
  /** Always keep this many messages (turns) from the start of the conversation. Default `0`. */
  keepFirst?: number;
  /** Always keep this many of the latest messages (turns). Default `1`. */
  keepLast?: number;
  /**
   * When dropping whole messages isn't enough, cut the middle out of the
   * longest kept messages' text. Default `true`.
   */
  truncate?: boolean;
  /**
   * Drop an assistant message left at the start of the conversation after
   * trimming, since some APIs (Anthropic) require the first message to be
   * from the user. Default `true`.
   */
  ensureUserFirst?: boolean;
  /** Builds a message that stands in for the dropped ones, like a summary or "12 earlier messages omitted". */
  placeholder?: (dropped: M[]) => M | undefined;
}

export interface FitResult<M extends ChatMessage = ChatMessage> {
  /** The messages that fit, in their original order. Input messages are never mutated. */
  messages: M[];
  /** Tokens the result takes, including `system`. */
  tokens: number;
  /** The budget it had to fit: `maxTokens - reserve`. */
  budget: number;
  /** Messages that were removed, in their original order. */
  dropped: M[];
  /** Whether any message's text was shortened. */
  truncated: boolean;
  /** `false` when even the protected messages don't fit (nothing more could be removed). */
  fits: boolean;
}

interface Group {
  start: number;
  end: number;
  tokens: number;
  system: boolean;
}

interface TextTarget {
  key: string;
  text: string;
  set: (text: string) => ChatMessage;
}

const TRUNCATION_MARKER = "\n\n[…]\n\n";

function isSystem(m: ChatMessage): boolean {
  return m.role === "system" || m.role === "developer";
}

/** Tool results must stay with the assistant message that made the call, or APIs reject the request. */
function isToolResult(m: ChatMessage): boolean {
  if (m.role === "tool" || m.role === "function") return true;
  return (
    Array.isArray(m.content) &&
    m.content.some((p: { type?: unknown } | null) => {
      const type = p?.type;
      return type === "tool_result" || type === "tool-result" || type === "function_call_output";
    })
  );
}

/**
 * Trims a chat history to fit a token budget: drops the oldest turns first,
 * keeps system prompts, the first and latest turns, and tool call / result
 * pairs together, then shortens text only if it must.
 *
 * ```ts
 * const { messages } = fitChat(history, { maxTokens: 128_000, reserve: 4_000 });
 * ```
 */
export function fitChat<M extends ChatMessage>(messages: readonly M[], options: FitOptions<M>): FitResult<M> {
  const {
    maxTokens,
    reserve = 0,
    system,
    keepSystem = true,
    keepFirst = 0,
    keepLast = 1,
    truncate = true,
    ensureUserFirst = true,
    placeholder,
  } = options;
  const count = (m: ChatMessage) => countMessageTokens(m, options);
  const budget = maxTokens - reserve;
  const systemTokens = system == null ? 0 : count({ role: "system", content: system });

  const groups: Group[] = [];
  messages.forEach((m, i) => {
    const last = groups.at(-1);
    if (last && !last.system && isToolResult(m)) {
      last.end = i + 1;
      last.tokens += count(m);
    } else {
      groups.push({ start: i, end: i + 1, tokens: count(m), system: keepSystem && isSystem(m) });
    }
  });

  const conversation = groups.filter((g) => !g.system);
  const head = Math.min(Math.max(0, keepFirst), conversation.length);
  const tail = Math.max(head, conversation.length - Math.max(0, keepLast));
  const droppable = conversation.slice(head, tail);
  const total = systemTokens + groups.reduce((n, g) => n + g.tokens, 0);

  const plan = (room: number) => {
    const dropped = new Set<Group>();
    let tokens = total;
    for (const g of droppable) {
      if (tokens <= room) break;
      dropped.add(g);
      tokens -= g.tokens;
    }
    if (ensureUserFirst && dropped.size > 0) {
      for (const g of conversation) {
        if (dropped.has(g)) continue;
        if (messages[g.start]!.role !== "assistant" || !droppable.includes(g)) break;
        dropped.add(g);
        tokens -= g.tokens;
      }
    }
    return { dropped, tokens };
  };

  let { dropped, tokens } = plan(budget);
  const collect = (set: Set<Group>) =>
    groups.filter((g) => set.has(g)).flatMap((g) => messages.slice(g.start, g.end));

  let filler: M | undefined;
  if (placeholder && dropped.size > 0) {
    filler = placeholder(collect(dropped));
    if (filler) {
      const extra = count(filler);
      if (tokens + extra > budget) {
        ({ dropped, tokens } = plan(budget - extra));
        filler = placeholder(collect(dropped));
      }
      if (filler) tokens += count(filler);
    }
  }

  const out: M[] = [];
  let fillerAt = -1;
  for (const g of groups) {
    if (dropped.has(g)) {
      if (fillerAt === -1) fillerAt = out.length;
      continue;
    }
    out.push(...messages.slice(g.start, g.end));
  }
  if (filler && fillerAt !== -1) out.splice(fillerAt, 0, filler);

  let truncated = false;
  if (truncate && tokens > budget) {
    const result = shorten(out, tokens - budget, options);
    truncated = result.changed;
    tokens -= result.saved;
  }

  return { messages: out, tokens, budget, dropped: collect(dropped), truncated, fits: tokens <= budget };
}

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

/** Cuts the middle out of `text` to keep at most `keep` UTF-16 code units without splitting surrogate pairs. */
function cutMiddle(text: string, keep: number): string {
  if (keep >= text.length) return text;
  let headEnd = Math.ceil(keep / 2);
  let tailStart = text.length - Math.floor(keep / 2);
  if (headEnd > 0 && isHighSurrogate(text.charCodeAt(headEnd - 1)) && isLowSurrogate(text.charCodeAt(headEnd))) {
    headEnd -= 1;
  }
  if (
    tailStart > 0 &&
    tailStart < text.length &&
    isLowSurrogate(text.charCodeAt(tailStart)) &&
    isHighSurrogate(text.charCodeAt(tailStart - 1))
  ) {
    tailStart += 1;
  }
  if (headEnd === 0 && tailStart === text.length) return TRUNCATION_MARKER.trim();
  return text.slice(0, headEnd) + TRUNCATION_MARKER + text.slice(tailStart);
}

/** Finds all truncatable text fields in a message: string content, text parts, or string tool-result parts. */
function textTargets(m: ChatMessage): TextTarget[] {
  if (typeof m.content === "string") {
    return [{ key: "content", text: m.content, set: (text) => ({ ...m, content: text }) }];
  }
  if (!Array.isArray(m.content)) return [];
  const parts = m.content as unknown[];
  const targets: TextTarget[] = [];
  for (let i = 0; i < parts.length; i++) {
    const raw = parts[i];
    if (!raw || typeof raw !== "object") continue;
    const p = raw as Record<string, unknown>;
    if (typeof p.text === "string") {
      targets.push({
        key: `${i}:text`,
        text: p.text,
        set: (text) => ({ ...m, content: parts.map((item, j) => (j === i ? { ...p, text } : item)) }),
      });
      continue;
    }
    const type = typeof p.type === "string" ? p.type : "";
    if (type === "tool_result" || type === "tool-result" || type === "function_call_output") {
      const field =
        typeof p.content === "string"
          ? "content"
          : typeof p.output === "string"
            ? "output"
            : typeof p.result === "string"
              ? "result"
              : undefined;
      if (field) {
        targets.push({
          key: `${i}:${field}`,
          text: p[field] as string,
          set: (text) => ({ ...m, content: parts.map((item, j) => (j === i ? { ...p, [field]: text } : item)) }),
        });
      }
    }
  }
  return targets;
}

/** Cuts the middle out of the longest non-system text fields until `over` tokens are saved. */
function shorten(out: ChatMessage[], over: number, options: CountOptions): { changed: boolean; saved: number } {
  const count = (m: ChatMessage) => countMessageTokens(m, options);
  const countText = options.countTokens ?? estimateTokens;
  let saved = 0;
  let changed = false;
  const tried = new Set<string>();
  while (saved < over) {
    let pick = -1;
    let pickTarget: TextTarget | undefined;
    let pickScore = 0;
    for (let i = 0; i < out.length; i++) {
      const m = out[i]!;
      if (isSystem(m)) continue;
      for (const target of textTargets(m)) {
        const id = `${i}:${target.key}`;
        if (tried.has(id)) continue;
        const score = countText(target.text);
        if (score > pickScore) {
          pick = i;
          pickTarget = target;
          pickScore = score;
        }
      }
    }
    if (pick === -1 || !pickTarget) break;
    tried.add(`${pick}:${pickTarget.key}`);
    const m = out[pick]!;
    const pickTokens = count(m);
    const { text, set } = pickTarget;
    const targetTokens = pickTokens - (over - saved);
    // Binary search the most text (head + tail) that keeps the message within targetTokens.
    let lo = 0;
    let hi = text.length;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (count(set(cutMiddle(text, mid))) <= targetTokens) lo = mid;
      else hi = mid - 1;
    }
    const next = set(cutMiddle(text, lo));
    const nextTokens = count(next);
    if (nextTokens >= pickTokens) continue;
    out[pick] = next;
    saved += pickTokens - nextTokens;
    changed = true;
  }
  return { changed, saved };
}
