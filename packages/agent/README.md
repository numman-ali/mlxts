# @mlxts/agent

First-class local agent loop primitives for mlxts.

`@mlxts/agent` owns the loop: conversation state, tool schemas, tool-call
parsing, tool execution, observations, and stop conditions. Model serving stays
in `@mlxts/serve`; examples should only demonstrate usage.

## CLI

Start a local model endpoint first:

```bash
mlxts-serve mlx-community/Qwen3.6-27B-4bit --model-id mlx-community/Qwen3.6-27B-4bit --port 8000
```

Then talk to it with read-only local tools:

```bash
mlxts-agent --model mlx-community/Qwen3.6-27B-4bit --endpoint http://127.0.0.1:8000 --cwd .
```

Agent turns use the served model's generation defaults unless you override them.
Use `--greedy` or `--deterministic` when you want `temperature: 0`, and use
`--thinking` / `--no-thinking` to pass Qwen-style thinking controls through the
chat template when the served model supports them. The CLI uses streaming chat
completions by default so reasoning and assistant text appear as the model
generates; use `--no-stream` to force the older whole-response path.

The CLI prints model reasoning, tool calls, tool results, and final answers as
separate sections (`[thinking]`, `[tool call]`, `[tool result]`, `[assistant]`)
so a local loop is inspectable without feeling like raw protocol logs.
If a turn exhausts `--max-iterations` before a final answer, the CLI prints an
`[agent]` notice instead of silently returning to the prompt.

The first CLI slice intentionally includes read-only file tools only. Shell,
write/edit tools, MCP, approvals, session compaction, and sandbox policy are
separate product layers, not hidden prompt glue.

## Programmatic Loop

```ts
import {
  createOpenAIChatAgentModel,
  createReadOnlyFileTools,
  runAgentTurn,
} from "@mlxts/agent";

const model = createOpenAIChatAgentModel({
  endpoint: "http://127.0.0.1:8000",
  model: "mlx-community/Qwen3.6-27B-4bit",
});

const result = await runAgentTurn({
  model,
  tools: createReadOnlyFileTools({ root: process.cwd() }),
  messages: [{ role: "user", content: "Read README.md and summarize the project." }],
});

console.log(result.finalText);
```

Tool calls use a conservative, model-agnostic envelope:

```xml
<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>
```

The parser also accepts Qwen-style function blocks such as
`<tool_call><function=list_files>{}</function></tool_call>`, because local
models may emit their native tool syntax even when prompted with the generic
envelope. That keeps the first loop easy to test while leaving room for richer
model-family tool-call parsers and future non-chat agent adapters later.
