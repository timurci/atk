# Changelog

## [1.0.0] - 2026-04-09

_If you are upgrading from 0.x: the `atk.providers.openai` module has been replaced by `atk.providers` using the any-llm SDK. Update imports from `AnyLanguageModel` (formerly `OpenAILanguageModel`) and configure providers via the `provider` parameter instead of `api_key`/`model` kwargs._

### Changed

- **Breaking:** replace OpenAI integration with any-llm providers ([`84195be`](https://github.com/timurci/atk/commit/84195be))

### Added

- Add `CallableToolset` and tool registration from callables with automatic parameter extraction ([`0ff3a3b`](https://github.com/timurci/atk/commit/0ff3a3b), [`4022a79`](https://github.com/timurci/atk/commit/4022a79))
- Add stream response generation for language models ([`dbf252a`](https://github.com/timurci/atk/commit/dbf252a))
- Add `ThinkingPart` and `ThinkingDelta` types for reasoning token support ([`75df460`](https://github.com/timurci/atk/commit/75df460))
- Add async factory method for model auto-detection ([`0f3d328`](https://github.com/timurci/atk/commit/0f3d328))
- Extend type annotations for array and dict parameters ([`c49124a`](https://github.com/timurci/atk/commit/c49124a))

### Fixed

- Fix message mapping for `ToolResultPart`, verbatim text, and invalid JSON in tool call arguments ([`71b27e6`](https://github.com/timurci/atk/commit/71b27e6), [`6ddec2b`](https://github.com/timurci/atk/commit/6ddec2b), [`7d616a1`](https://github.com/timurci/atk/commit/7d616a1), [`f955171`](https://github.com/timurci/atk/commit/f955171))
- Handle `typing.Union`/`Optional` and validate dict key types in tool parameters ([`a44b261`](https://github.com/timurci/atk/commit/a44b261))

## [0.2.0] - 2026-03-18
### Added

- Add tool support to language protocol (`0855659`, `1fe9ebf`)
- Support OpenAI SDK (`ae4e8f4`)
- Add language protocol and text support (`62e73fc`)
