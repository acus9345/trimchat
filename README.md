# fitchat

Trim a chat history to fit a token budget **without breaking the conversation**. Drops the oldest turns first, keeps the system prompt, keeps tool calls with their results, makes sure the first message is from the user, and only cuts text when it has to. Works with OpenAI, Anthropic and Vercel AI SDK messages. Zero dependencies.

[![npm](https://img.shields.io/npm/v/fitchat)](https://www.npmjs.com/package/fitchat)
[![CI](https://github.com/acus9345/fitchat/actions/workflows/ci.yml/badge.svg)](https://github.com/acus9345/fitchat/actions/workflows/ci.yml)
[![bundle size](https://img.shields.io/bundlejs/size/fitchat)](https://bundlejs.com/?q=fitchat)
[![license](https://img.shields.io/npm/l/fitchat)](LICENSE)

```ts
import { fitChat } from "fitchat";

const { messages } = fitChat(history, { maxTokens: 128_000, reserve: 8_000 });
await openai.chat.completions.create({ model: "gpt-5", messages });
```

## Why

`history.slice(-20)` is how most apps do it, and it breaks in ways that only show up in production:

| Problem | `slice(-N)` | fitchat |
| --- | --- | --- |
| Long messages still overflow the context | ❌ | budget in tokens, not messages |
| System prompt gets cut off | ❌ | always kept |
| `tool` result left without its tool call → **400 error** | ❌ | call and results drop together |
| Anthropic: first message is `assistant` → **400 error** | ❌ | trimmed to start with `user` |
| One huge pasted document is bigger than the window | ❌ | cuts the middle out of it |
| Room for the reply | ❌ | `reserve` |
| You want to know what was removed | ❌ | `dropped`, `tokens`, `fits` |

## Install

```sh
npm install fitchat
```

## Usage

### OpenAI

```ts
import { fitChat } from "fitchat";

const { messages, dropped } = fitChat(history, {
  maxTokens: 400_000, // the model's context window
  reserve: 16_000, // room for the reply
});
```

### Anthropic

Pass the separate `system` prompt so it counts against the budget:

```ts
const { messages } = fitChat(history, { maxTokens: 200_000, reserve: 8_000, system });
await anthropic.messages.create({ model: "claude-sonnet-5-5", max_tokens: 8_000, system, messages });
```

`tool_use` blocks and the `tool_result` blocks that answer them are always kept or dropped together, and an `assistant` message is never left first.

### Vercel AI SDK

```ts
import { streamText } from "ai";
import { fitChat } from "fitchat";

const { messages } = fitChat(modelMessages, { maxTokens: 128_000, reserve: 4_000 });
streamText({ model, messages });
```

### Exact token counts

By default fitchat estimates tokens (about 4 characters per token, 1 per CJK character), which is fast and usually within 10-15%. For exact counts, plug in a tokenizer:

```ts
import { encode } from "gpt-tokenizer";

fitChat(history, { maxTokens: 128_000, countTokens: (text) => encode(text).length });
```

### Keep the task, summarize the rest

```ts
fitChat(history, {
  maxTokens: 32_000,
  keepFirst: 1, // the user's original request
  keepLast: 6, // the latest three exchanges
  placeholder: (dropped) => ({
    role: "user",
    content: `(${dropped.length} earlier messages were removed to save space.)`,
  }),
});
```

The placeholder goes where the dropped messages were, and fitchat makes room for it in the budget. You can also return a real summary (computed earlier) from it.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `maxTokens` | required | The budget, usually the model's context window. |
| `reserve` | `0` | Tokens kept free for the reply. The budget is `maxTokens - reserve`. |
| `system` | none | A system prompt passed separately (Anthropic, Gemini), counted against the budget. |
| `keepSystem` | `true` | Never drop `system` / `developer` messages. |
| `keepFirst` | `0` | Always keep this many turns from the start. |
| `keepLast` | `1` | Always keep this many of the latest turns. |
| `truncate` | `true` | If dropping turns isn't enough, cut the middle out of the longest kept messages, keeping the start and the end with `[…]` between them. |
| `ensureUserFirst` | `true` | After trimming, drop assistant messages left at the start. |
| `placeholder` | none | `(dropped) => message`, inserted where messages were removed. |
| `countTokens` | estimate | `(text) => number` for exact counts. |
| `messageOverhead` | `4` | Tokens per message for role and formatting. |
| `imageTokens` | `1000` | Tokens charged per image, file or audio part. |

A "turn" here is one message, or an assistant tool call together with all of its results.

## Result

```ts
{
  messages,   // what fits, in order (inputs are never mutated)
  tokens,     // tokens used, including `system`
  budget,     // maxTokens - reserve
  dropped,    // removed messages, in order
  truncated,  // whether any text was cut
  fits,       // false if even the protected messages are over budget
}
```

## Also exported

- `countTokens(messages, options?)`: tokens a message list takes.
- `countMessageTokens(message, options?)`: tokens one message takes.
- `estimateTokens(text)`: the default estimator.

All of them understand string content, text/image/file parts, OpenAI `tool_calls` and `tool` messages, Anthropic `tool_use` / `tool_result`, and AI SDK `tool-call` / `tool-result` parts.

## License

[MIT](LICENSE)
