# @mlxts/agent

OpenAI-compatible chat client and tool-loop primitives. The boundary with `@mlxts/serve` is one HTTP call to `/v1/chat/completions` (or another OpenAI-compatible endpoint).

This package is an experimental harness for local loop primitives until a phase plan promotes it. It is not the primary product-agent integration surface.

Owns: agent message types, tool registration, tool-call parsing from generated assistant text, tool execution scheduling, max-iteration discipline, conversation state, CLI presentation.

Non-interactive CLI commands and finite status/error output follow `.agents/skills/axi/SKILL.md`. The interactive REPL is a terminal conversation surface. Non-TTY paths do not prompt.

Out of scope: model execution. Imports of `@mlxts/core`, `@mlxts/nn`, `@mlxts/transformers`, and `@mlxts/serve` are forbidden. The package stays usable against any OpenAI-compatible server.

Reasoning-tag handling re-exports from `@mlxts/protocols`. Forking the tag-stream parser inside this package is forbidden.

`createOpenAIChatAgentModel` is the OpenAI-compatible adapter. New transports land as sibling adapters; the `AgentModel` contract is the line they cross.

`runAgentTurn` is the single iteration primitive. Multi-turn approval, sandboxing, and planning compose on top of `runAgentTurn`, not inside it.

Tool-call parsing accepts both the JSON envelope (`<tool_call>{...}</tool_call>`) and Qwen-style native function blocks (`<tool_call><function=name>...</function></tool_call>`). Malformed or incomplete envelopes remain visible content.

Max-iteration termination surfaces as an explicit `[agent]` notice. A silent stop is forbidden.
