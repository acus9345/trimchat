import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { countMessageTokens, countTokens, estimateTokens, fitChat, type ChatMessage } from "../dist/index.js";

/** One token per word, no overhead: easy to reason about in tests. */
const words = { countTokens: (t: string) => (t.trim() ? t.trim().split(/\s+/).length : 0), messageOverhead: 0 };
const msg = (role: string, n: number, tag = role): ChatMessage => ({
  role,
  content: Array.from({ length: n }, (_, i) => `${tag}${i}`).join(" "),
});

describe("estimateTokens()", () => {
  it("estimates about 4 characters per token, 1 per CJK character", () => {
    assert.equal(estimateTokens(""), 0);
    assert.equal(estimateTokens("abcd"), 1);
    assert.equal(estimateTokens("abcde"), 2);
    assert.equal(estimateTokens("你好世界"), 4);
  });
});

describe("countMessageTokens()", () => {
  it("counts strings, parts, images, tool calls and overhead", () => {
    assert.equal(countMessageTokens({ role: "user", content: "a b c" }, words), 3);
    assert.equal(countMessageTokens({ role: "user", content: "a b" }, { ...words, messageOverhead: 4 }), 6);
    const parts = {
      role: "user",
      content: [
        { type: "text", text: "look at this" },
        { type: "image_url", image_url: { url: "data:..." } },
      ],
    };
    assert.equal(countMessageTokens(parts, { ...words, imageTokens: 100 }), 103);
    const call = {
      role: "assistant",
      content: null,
      tool_calls: [{ id: "1", type: "function", function: { name: "get_weather", arguments: '{"city": "Paris"}' } }],
    };
    assert.equal(countMessageTokens(call, words), 1 + 2 + 3);
    const objectArgsCall = {
      role: "assistant",
      content: null,
      tool_calls: [{ id: "2", type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }],
    };
    assert.equal(countMessageTokens(objectArgsCall, words), 1 + 1 + 3);
    const anthropic = {
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "x", content: [{ type: "text", text: "sunny and warm" }] }],
    };
    assert.equal(countMessageTokens(anthropic, words), 3);
    const fnOutput = {
      role: "user",
      content: [{ type: "function_call_output", call_id: "fc_1", output: "sunny and warm" }],
    };
    assert.equal(countMessageTokens(fnOutput, words), 3);
    assert.equal(countTokens([msg("user", 2), msg("assistant", 3)], words), 5);
  });
});

describe("fitChat()", () => {
  it("returns everything when it already fits", () => {
    const history = [msg("system", 5), msg("user", 5), msg("assistant", 5)];
    const r = fitChat(history, { ...words, maxTokens: 100 });
    assert.deepEqual(r.messages, history);
    assert.equal(r.tokens, 15);
    assert.deepEqual(r.dropped, []);
    assert.equal(r.fits, true);
    assert.equal(r.truncated, false);
  });

  it("drops the oldest turns first and keeps the system prompt", () => {
    const history = [msg("system", 10), msg("user", 10, "u1"), msg("assistant", 10, "a1"), msg("user", 10, "u2"), msg("assistant", 10, "a2"), msg("user", 10, "u3")];
    const r = fitChat(history, { ...words, maxTokens: 35 });
    assert.deepEqual(r.messages.map((m) => m.role), ["system", "user"]);
    assert.deepEqual(r.messages, [history[0], history[5]]);
    assert.equal(r.tokens, 20);
    assert.equal(r.dropped.length, 4);
  });

  it("never leaves an assistant message first after trimming", () => {
    const history = [msg("user", 10, "u1"), msg("assistant", 10, "a1"), msg("user", 10, "u2"), msg("assistant", 10, "a2"), msg("user", 10, "u3")];
    // Dropping only u1 would fit 40, but then a1 would come first.
    const r = fitChat(history, { ...words, maxTokens: 40 });
    assert.deepEqual(r.messages, history.slice(2));
    const loose = fitChat(history, { ...words, maxTokens: 40, ensureUserFirst: false });
    assert.deepEqual(loose.messages, history.slice(1));
  });

  it("does not drop a leading assistant greeting when nothing was trimmed", () => {
    const history = [msg("assistant", 5), msg("user", 5)];
    assert.deepEqual(fitChat(history, { ...words, maxTokens: 10 }).messages, history);
  });

  it("keeps the first turns with keepFirst and more recent turns with keepLast", () => {
    const history = [msg("user", 10, "task"), msg("assistant", 10, "a1"), msg("user", 10, "u2"), msg("assistant", 10, "a2"), msg("user", 10, "u3")];
    const r = fitChat(history, { ...words, maxTokens: 30, keepFirst: 1 });
    assert.deepEqual(r.messages, [history[0], history[3], history[4]]);
    const r2 = fitChat(history, { ...words, maxTokens: 25, keepLast: 3, truncate: false });
    assert.deepEqual(r2.messages, history.slice(2));
    assert.equal(r2.fits, false);
  });

  it("clamps negative keepFirst and keepLast to zero", () => {
    const history = [msg("user", 10, "u1"), msg("assistant", 10, "a1"), msg("user", 10, "u2")];
    const r = fitChat(history, { ...words, maxTokens: 15, keepFirst: -2, keepLast: -1 });
    assert.deepEqual(r.messages, [history[2]]);
    assert.equal(r.fits, true);
  });

  it("keeps OpenAI tool calls and their results together", () => {
    const history: ChatMessage[] = [
      msg("user", 10, "q1"),
      { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "search", arguments: "x" } }] },
      { role: "tool", tool_call_id: "c1", content: "result one two three" },
      msg("assistant", 5, "answer"),
      msg("user", 5, "q2"),
    ];
    const r = fitChat(history, { ...words, maxTokens: 12 });
    // The call + result pair is dropped as a unit, never leaving an orphaned tool message.
    assert.ok(!r.messages.some((m) => m.role === "tool"));
    assert.deepEqual(r.messages, [history[4]]);
    const r2 = fitChat(history, { ...words, maxTokens: 22, ensureUserFirst: false });
    assert.deepEqual(r2.messages, history.slice(1));
  });

  it("keeps Anthropic tool_use and tool_result together", () => {
    const history: ChatMessage[] = [
      msg("user", 10, "q1"),
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "calc", input: { a: 1 } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "42" }] },
      msg("assistant", 3, "answer"),
      msg("user", 3, "q2"),
    ];
    const r = fitChat(history, { ...words, maxTokens: 10 });
    assert.ok(r.messages.every((m) => !JSON.stringify(m).includes("tool_result")));
    assert.deepEqual(r.messages, [history[4]]);
  });

  it("counts a separate system prompt and the reserve against the budget", () => {
    const history = [msg("user", 10, "u1"), msg("assistant", 10, "a1"), msg("user", 10, "u2")];
    const r = fitChat(history, { ...words, maxTokens: 50, reserve: 20, system: "one two three four five" });
    assert.equal(r.budget, 30);
    assert.deepEqual(r.messages, [history[2]]);
    assert.equal(r.tokens, 15);
  });

  it("inserts a placeholder for dropped messages and makes room for it", () => {
    const history = [msg("system", 5), msg("user", 10, "u1"), msg("assistant", 10, "a1"), msg("user", 10, "u2"), msg("assistant", 10, "a2"), msg("user", 5, "u3")];
    const r = fitChat(history, {
      ...words,
      maxTokens: 30,
      ensureUserFirst: false,
      placeholder: (dropped) => ({ role: "system", content: `${dropped.length} earlier messages omitted` }),
    });
    assert.deepEqual(r.messages[0], history[0]);
    assert.equal(r.messages[1]!.content, "3 earlier messages omitted");
    assert.deepEqual(r.messages.slice(2), history.slice(4));
    assert.equal(r.tokens, 5 + 4 + 15);
    assert.ok(r.fits);
  });

  it("truncates the middle of a message that is too long on its own", () => {
    const long = Array.from({ length: 200 }, (_, i) => `w${i}`).join(" ");
    const history = [msg("system", 5), { role: "user", content: long }];
    const r = fitChat(history, { ...words, maxTokens: 50 });
    assert.equal(r.truncated, true);
    assert.ok(r.fits);
    assert.ok(r.tokens <= 50);
    const text = r.messages[1]!.content as string;
    assert.ok(text.startsWith("w0 w1"), text);
    assert.ok(text.endsWith("w198 w199"), text);
    assert.ok(text.includes("[…]"));
    // The input is untouched.
    assert.equal(history[1]!.content, long);
  });

  it("truncates text parts inside content arrays", () => {
    const long = Array.from({ length: 100 }, (_, i) => `w${i}`).join(" ");
    const history = [{ role: "user", content: [{ type: "image", source: {} }, { type: "text", text: long }] }];
    const r = fitChat(history, { ...words, imageTokens: 10, maxTokens: 40 });
    assert.ok(r.fits);
    const parts = r.messages[0]!.content as Array<{ type: string; text?: string }>;
    assert.equal(parts[0]!.type, "image");
    assert.ok(parts[1]!.text!.length < long.length);
  });

  it("truncates multiple text parts in the same message when one part isn't enough", () => {
    const long1 = Array.from({ length: 100 }, (_, i) => `a${i}`).join(" ");
    const long2 = Array.from({ length: 100 }, (_, i) => `b${i}`).join(" ");
    const history = [{ role: "user", content: [{ type: "text", text: long1 }, { type: "text", text: long2 }] }];
    const r = fitChat(history, { ...words, maxTokens: 40 });
    assert.equal(r.truncated, true);
    assert.equal(r.fits, true);
    assert.ok(r.tokens <= 40);
    const parts = r.messages[0]!.content as Array<{ type: string; text: string }>;
    assert.ok(parts[0]!.text.length < long1.length);
    assert.ok(parts[1]!.text.length < long2.length);
  });

  it("truncates string tool_result and function_call_output parts when a kept tool turn overflows", () => {
    const long = Array.from({ length: 100 }, (_, i) => `r${i}`).join(" ");
    const history: ChatMessage[] = [
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "fetch", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: long }] },
    ];
    const r = fitChat(history, { ...words, maxTokens: 30 });
    assert.equal(r.truncated, true);
    assert.equal(r.fits, true);
    assert.ok(r.tokens <= 30);
    const parts = r.messages[1]!.content as Array<{ type: string; content: string }>;
    assert.ok(parts[0]!.content.includes("[…]"));
    assert.equal((history[1]!.content as Array<{ content: string }>)[0]!.content, long);
  });

  it("does not split UTF-16 surrogate pairs when truncating", () => {
    const emojis = "😀".repeat(60);
    const history = [{ role: "user", content: emojis }];
    const r = fitChat(history, { countTokens: (t) => t.length, messageOverhead: 0, maxTokens: 25 });
    assert.equal(r.truncated, true);
    assert.equal(r.fits, true);
    const text = r.messages[0]!.content as string;
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      if (code >= 0xd800 && code <= 0xdbff) {
        const next = text.charCodeAt(i + 1);
        assert.ok(next >= 0xdc00 && next <= 0xdfff, "unpaired high surrogate");
        i++;
      } else {
        assert.ok(code < 0xdc00 || code > 0xdfff, "unpaired low surrogate");
      }
    }
  });

  it("reports fits: false when protected content can't shrink", () => {
    const history = [{ role: "user", content: [{ type: "image", source: {} }] }];
    const r = fitChat(history, { ...words, imageTokens: 100, maxTokens: 10 });
    assert.equal(r.fits, false);
    assert.deepEqual(r.messages, history);
  });

  it("works with the default estimator", () => {
    const history = Array.from({ length: 50 }, (_, i) => ({ role: i % 2 ? "assistant" : "user", content: "x".repeat(400) }));
    const r = fitChat(history, { maxTokens: 1000 });
    assert.ok(r.tokens <= 1000);
    assert.equal(r.messages.at(-1), history.at(-1));
    assert.equal(r.messages[0]!.role, "user");
  });
});
