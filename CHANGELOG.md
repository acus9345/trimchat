# Changelog

All notable changes to this project are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-20

### Added

- `fitChat()` trims a message list to `maxTokens - reserve`, dropping the oldest turns first.
- Keeps `system` / `developer` messages, `keepFirst` and `keepLast` turns, and tool calls together with their results (OpenAI, Anthropic and AI SDK formats).
- `ensureUserFirst`, `placeholder`, a separate `system` prompt, and middle-out `truncate` for messages that are too long on their own.
- `countTokens()`, `countMessageTokens()` and `estimateTokens()`, with pluggable `countTokens`, `messageOverhead` and `imageTokens`.

[0.1.0]: https://github.com/acus9345/trimchat/releases/tag/v0.1.0
