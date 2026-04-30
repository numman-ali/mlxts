# Product Surfaces

This project will eventually expose multiple interfaces to different users. Each surface has its own design principles, quality bar, and user expectations. Even when we're only building one surface (the API), the others inform our decisions — a good API makes a good CLI possible; a good CLI makes a good TUI possible.

This document defines the standards for each surface. All agents must consider these guidelines when making design decisions, even in early phases.

## The Surfaces

```
┌─────────────────────────────────────────────────────┐
│                     GUI (future)                     │
│         Web dashboard, training playground           │
├─────────────────────────────────────────────────────┤
│                     TUI (future)                     │
│       Interactive terminal, live training view       │
├─────────────────────────────────────────────────────┤
│            CLI / Serve / Agent (Phase 4-9)           │
│   train, generate, serve, inspect, benchmark, agent   │
├─────────────────────────────────────────────────────┤
│              Examples / Workbooks (ongoing)          │
│       real tasks, proof flows, thin package usage     │
├─────────────────────────────────────────────────────┤
│                     API (Phase 1+)                   │
│         canonical @mlxts/* package surfaces          │
└─────────────────────────────────────────────────────┘
```

Each layer builds on the one below it. A broken API surface makes everything above it worse. A beautiful GUI can't compensate for confusing types.

---

## API Surface — The Foundation

**Users**: TypeScript developers building ML applications, researchers experimenting with models, learners studying transformers.

**When**: Phase 1 onward. This is the first and most critical surface.

### Principles

**Predictable over clever.** A developer should be able to guess the API without reading docs. If `mx.add(a, b)` exists, `mx.subtract(a, b)` should work identically. No special cases, no surprising overloads.

**Types are documentation.** The type signature should tell you what a function does, what it accepts, and what it returns. If you need to read a comment to use it correctly, the types aren't good enough.

**Errors are part of the interface.** When something goes wrong, the error message should tell you: what happened, why it happened, and what to do about it. Never expose raw C++ errors or pointer addresses to the user.

**Composable over monolithic.** Small functions that combine well beat large functions with many options. `mx.matmul(mx.transpose(a), b)` is better than `mx.matmul(a, b, { transposeFirst: true })`.

### Standards

- Every public function has a TypeScript overload signature (no `...args: any[]`)
- Every public type is exported and documented
- Every error thrown includes the operation name, expected input, and actual input
- No function has more than 5 required parameters (use config objects beyond that)
- Return types are always concrete, never `any` or `unknown` at the public boundary
- Async operations are explicitly marked — no hidden promises

### Example of what good looks like

```typescript
// Clear, typed, predictable
const weights = mx.random.normal([768, 768]);
const input = mx.zeros([32, 128, 768]);
const output = mx.matmul(input, weights);
mx.eval(output);

// Error tells you exactly what's wrong
// "mx.matmul: dimension mismatch — a.shape[-1] is 768, b.shape[0] is 512.
//  a.shape: [32, 128, 768], b.shape: [512, 768]"
```

---

## CLI Surface — The Operator Interface

**Users**: Developers training models, running experiments, inspecting checkpoints, generating text.

**When**: Phase 4-5. Built on top of the API.

### Principles

**Discoverable.** `--help` on any command tells you everything you need. No hidden flags. No required reading of external docs to run a basic training job.

**Progressive disclosure.** The simplest invocation works with sensible defaults. Advanced options are available but never required.

```bash
# Simple — works out of the box
nanogpt train

# Advanced — every knob is available
nanogpt train --preset gpt-small --data ./data/shakespeare.txt --lr 3e-4 --batch-size 2 --grad-accum 8 --max-steps 5000
```

**Agent-first structured output.** Agent-facing finite commands follow the repo-local AXI skill at [`.agents/skills/axi/SKILL.md`](../.agents/skills/axi/SKILL.md): TOON-shaped stdout, compact default schemas, explicit empty states, actionable structured errors, and contextual next-step hints. JSON remains available only as an explicit compatibility or export format where a surface already promises it.

**Long-run control is explicit.** Overnight training is supervised through a run-local directory with structured events, status snapshots, and checkpointed stop/resume. We do not rely on hidden daemons or ad hoc shell state.

**Operational evidence matters.** A long-run training surface is not trustworthy just because it prints loss. It needs memory, throughput, checkpoint, and stop/resume signals that let an operator decide what to do next without guessing.

**Semantic progress matters too.** Loss curves are necessary, but they are not the whole story. Training and acceptance surfaces should make it easy to inspect occasional generated samples so a human can see whether the model is actually learning language-like structure.

**Respect the terminal.** Detect terminal width. Use color only when stdout is a TTY and the output is not an AXI data stream. Support `NO_COLOR`. Never break pipe chains.

### Standards

- Every command has `--help` with examples
- Agent-facing finite one-shot commands emit AXI/TOON on stdout by default
- JSON is an explicit compatibility or export mode, not the default agent channel
- Long-running servers and interactive REPLs expose structured telemetry, status, or transcript surfaces instead of one final structured result
- Exit codes are meaningful: 0 = success including no-ops, 1 = error, 2 = usage error
- Long-running operations show progress (spinner or progress bar)
- Config files override defaults; flags override config files; env vars are a last resort
- No interactive prompts in non-TTY environments
- Structured errors go to stdout for agent consumption; diagnostics and progress go to stderr
- Long-running training commands emit enough structured telemetry to diagnose throughput drift, memory pressure, checkpoint health, and stop/resume state
- Long-running training and acceptance flows should expose occasional generated samples so semantic progress is visible alongside scalar metrics
- Overnight or resumable training should go through the canonical `cd examples/nanogpt && bun run manager ...` manager flow, not loose root scripts or one-off shell wrappers

### Example of what good looks like

```bash
$ nanogpt train --preset gpt-tiny
Training gpt-tiny on Shakespeare

  Step    Loss    LR        Tokens/sec
  ──────────────────────────────────────
  100     4.21    3.0e-4    12,400
  200     3.58    3.0e-4    12,350
  300     2.94    2.9e-4    12,380
  ...
  5000    1.42    1.2e-5    12,290

Training complete. Resume checkpoint saved: .nanogpt-checkpoints/gpt-tiny-resume-step-5000/

$ nanogpt generate --checkpoint .nanogpt-checkpoints/gpt-tiny-resume-step-5000 --prompt "To be or"
To be or not to be, that is the question—
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

### Overnight operator flow

```bash
$ cd examples/nanogpt && bun run manager start --preset gpt-small --max-steps 5000
Started run 2026-04-04T01-14-53-gpt-small

$ cd examples/nanogpt && bun run manager status --name 2026-04-04T01-14-53-gpt-small
Run 2026-04-04T01-14-53-gpt-small
  state: running
  step: 750 / 5000
  loss: 2.8100  val: 2.7900
  tokens/sec: 4,100
  active memory: 2,048 MB
  cache memory: 3,072 MB
  latest checkpoint: .nanogpt-runs/.../checkpoints/gpt-small-snapshot-step-750

$ cd examples/nanogpt && bun run manager stop --name 2026-04-04T01-14-53-gpt-small
Requested stop for run 2026-04-04T01-14-53-gpt-small

$ cd examples/nanogpt && bun run manager resume --from 2026-04-04T01-14-53-gpt-small --max-steps 10000
Started run 2026-04-04T04-00-00-gpt-small
```

The key design point is that stop/resume is checkpoint-driven, not process-magic. A stop request is acknowledged in the run directory, the trainer exits cleanly, and resume restarts from the latest resumable checkpoint.

The engineering implication is just as important as the operator experience: this surface is production code. Changes to it need the same runtime review, tests, and evidence as tensor or optimizer code.

This operator flow lives in `examples/nanogpt` because nanoGPT is now an
intentional example surface rather than a publishable package. The behavior is
canonical; the reusable abstractions should still live in `@mlxts/*`.

---

## Serving Surface — The Local Model Endpoint

**Users**: Developers running local models from OpenAI-compatible tools, agents,
scripts, and product prototypes.

**When**: Phase 9 onward. Built on `@mlxts/transformers`, tokenizers, runtime
controls, and the shared protocol-neutral generation request shape.

### Principles

**First-class package, not an example.** Loading and serving models belongs in
`@mlxts/serve`. Examples can demonstrate usage, but they do not own the product
contract.

**Protocol adapters stay thin.** OpenAI completions, chat completions,
text Responses, and bounded Anthropic Messages normalize into one internal
request model. They do not normalize into each other and they do not carry
model-family prompt logic.

**Model-native by default.** If a request omits sampling parameters, serving
preserves that omission so checkpoint `generation_config.json` can apply. Safety
or determinism modes must be explicit.

**Truthful concurrency.** Admission coalescing, static batch decode, and
continuous token-level batching are different claims. Logs, benchmarks, and
operator docs must name the exact layer being exercised.

**Operationally visible.** Serving must report model ids, request limits,
memory/admission metadata, generation start/completion/error events, streaming
lifecycle, and enough benchmark counters to debug multi-agent use.

### Standards

- `mlxts-serve <model>` starts a useful local endpoint with minimal flags
- `/health`, `/info`, `/v1/models`, and supported protocol routes are explicit
- protocol gaps reject clearly instead of accepting unsupported semantics
- request limits separate prompt tokens, generated tokens, total tokens, and
  memory-utilization preflight
- generated reasoning is surfaced as reasoning metadata, not leaked into visible
  assistant text
- benchmark claims use `bench:serve` for endpoint behavior and paired
  generation parity only for model-runtime comparison

---

## Agent Surface — The Local Tool Loop

**Users**: Developers who want to talk to a local model like an assistant and
inspect how tools, reasoning, and observations work.

**When**: Phase 9 onward. Built on `@mlxts/serve` chat routes and package-owned
tool-loop primitives.

### Principles

**The agent owns the loop.** Conversation state, tool schemas, tool-call
parsing, tool execution, observations, max-iteration behavior, and CLI
presentation belong in `@mlxts/agent`.

**Serving owns the endpoint.** `@mlxts/serve` executes model requests and formats
wire protocols. It does not execute tools.

**Inspectable by default.** Reasoning, assistant text, tool calls, tool results,
and stop notices should be visually distinct without feeling like raw HTTP logs.

**Safe before powerful.** Read-only file tools are the first slice. Shell tools,
write/edit tools, MCP, approvals, and sandbox policy are separate explicit
layers.

### Standards

- streaming is the default interactive path when the served endpoint supports it
- `--greedy` / `--deterministic` are explicit safety modes, not hidden defaults
- Qwen-style thinking and native tool-call syntax are handled deliberately
- a real interactive smoke covers multiple turns, reasoning display, tool use,
  and a final answer after observation

---

## Examples / Workbooks — The Learning and Proof Surface

**Users**: Developers learning the package stack, researchers reproducing a
flow, agents using the repo as a task workbook, and maintainers proving that a
surface works end to end.

**When**: Ongoing. Examples exist when they teach or validate a real workflow.

### Principles

**Thin over packages.** If reusable behavior appears in multiple examples, move
it into a canonical package instead of growing an example-local framework.

**Real-world over toy demos.** Examples should feel like ML workbooks: clear
setup, meaningful data/model choices, observable results, and honest limits.

**Proof labels matter.** Experimental proof assets can stay in the repo when
they teach something real, but they must not silently become product claims.

**Agent-readable.** A coding agent should be able to open an example, understand
the intended flow, run the narrow validation, and improve the reusable package
surface underneath it.

### Standards

- examples document commands from their own directory when the workflow is
  example-owned
- reusable package APIs are preferred over root scripts or loose helper layers
- examples include validation or review notes when they are used for proof
- no example is allowed to become the hidden product surface for serving,
  training, or agent behavior

---

## TUI Surface — The Training Dashboard (Future)

**Users**: Developers who want a richer view during training — live loss curves, GPU utilization, sample outputs.

**When**: After the package-first Phase 5 transition settles. Builds on CLI infrastructure.

### Principles

**Glanceable.** The most important information (current loss, step count, ETA) is always visible without scrolling.

**Non-destructive.** The TUI is a view layer. It never modifies training state. Closing it doesn't stop training.

**Keyboard-first.** Every action is reachable via keyboard. Mouse is optional.

### Vision

```
┌─ nanogpt training ──────────────────────────────────┐
│                                                      │
│  Model: gpt-small                                    │
│  Dataset: openwebtext (9B tokens)                    │
│  Step: 2,340 / 50,000  ████████░░░░░░░░░ 4.7%       │
│                                                      │
│  Loss ─────────────────────────────       Tokens/s   │
│  4.0 │╲                                  12,400     │
│  3.0 │ ╲                                            │
│  2.0 │  ╲___                              GPU: 94%  │
│  1.0 │      ╲___╲                         Mem: 42GB │
│      └──────────────────                             │
│        0    500   1000  1500  2000                    │
│                                                      │
│  Latest sample:                                      │
│  "The king did speak unto the court and..."          │
│                                                      │
│  [q] quit  [p] pause  [s] sample  [c] checkpoint    │
└──────────────────────────────────────────────────────┘
```

### Standards

- Minimum terminal size: 80x24
- Graceful degradation on smaller terminals
- All data available via the structured CLI interface; TUI is just a renderer
- Refresh rate respects terminal performance (no flickering)

---

## GUI Surface — The Web Interface (Future)

**Users**: Broader audience — learners who prefer visual interfaces, teams sharing training runs, demo deployments.

**When**: Post-Phase 6. Builds on API and CLI.

### Principles

**The API is the product, not the GUI.** The GUI is a client of the same API that the CLI and TUI use. No special backend endpoints. No state that only exists in the GUI.

**Educational by default.** Visualize what the model is learning — attention patterns, loss landscapes, token probabilities. The GUI should teach, not just display.

**Shareable.** A training run should be shareable via URL. A generated text sample should be embeddable.

### Standards

- Built on the same API surface — no backend-for-frontend
- Responsive design (works on mobile for monitoring)
- Accessible (WCAG 2.1 AA minimum)
- No client-side state that can't be reconstructed from the API

---

## Cross-Surface Principles

These apply to every surface:

1. **Consistency.** The same concept has the same name everywhere. If the API calls it `batchSize`, the CLI flag is `--batch-size`, the config key is `batchSize`, and the TUI displays "Batch size". No synonyms.

2. **No silent failures.** Every surface reports errors clearly. The training doesn't quietly produce garbage — it tells you why.

3. **Offline-first.** Everything runs locally on the user's Mac. No telemetry, no cloud dependencies, no account required. The network is optional.

4. **Layered complexity.** Each surface reveals more detail as the user asks for it. The default is simple. Complexity is opt-in.

5. **Beautiful defaults.** The zero-config experience should look and feel polished. Colors, spacing, alignment, typography (in GUI) — these details signal quality and build trust.

---

## How This Guides Agent Work

When building any feature, agents should ask:

- **API**: Is this function predictable? Can the types alone guide usage? Are errors helpful?
- **CLI**: If this feature had a CLI command, would it be discoverable? What would `--help` say?
- **TUI**: If this data were displayed in a dashboard, is it structured enough to render?
- **GUI**: If someone shared this as a URL, would it make sense to a viewer?

Even when building Phase 1 (pure API), these questions shape the design. A well-structured API makes every future surface easier. A poorly structured one makes them all harder.
