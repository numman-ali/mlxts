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
│                     CLI (Phase 4-5)                  │
│        train, generate, inspect, benchmark           │
├─────────────────────────────────────────────────────┤
│                     API (Phase 1-3)                  │
│     mlx-ts core, nn layers, optimizers, nanogpt      │
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
nanogpt train --dataset shakespeare --model gpt2-small --lr 3e-4 --batch-size 32 --max-steps 5000
```

**Structured output.** Human-readable by default, machine-parseable with `--json`. Scripts and CI can consume the output reliably.

**Respect the terminal.** Detect terminal width. Use color only when stdout is a TTY. Support `NO_COLOR`. Never break pipe chains.

### Standards

- Every command has `--help` with examples
- Every command supports `--json` for structured output
- Exit codes are meaningful: 0 = success, 1 = user error, 2 = system error
- Long-running operations show progress (spinner or progress bar)
- Config files override defaults; flags override config files; env vars are a last resort
- No interactive prompts in non-TTY environments
- Errors go to stderr, results go to stdout

### Example of what good looks like

```bash
$ nanogpt train --dataset shakespeare --model gpt2-tiny
Training gpt2-tiny on shakespeare (1.1M chars)

  Step    Loss    LR        Tokens/sec
  ──────────────────────────────────────
  100     4.21    3.0e-4    12,400
  200     3.58    3.0e-4    12,350
  300     2.94    2.9e-4    12,380
  ...
  5000    1.42    1.2e-5    12,290

Training complete. Checkpoint saved: checkpoints/gpt2-tiny-step5000.safetensors

$ nanogpt generate --checkpoint checkpoints/gpt2-tiny-step5000.safetensors --prompt "To be or"
To be or not to be, that is the question—
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

---

## TUI Surface — The Training Dashboard (Future)

**Users**: Developers who want a richer view during training — live loss curves, GPU utilization, sample outputs.

**When**: Post-Phase 5. Builds on CLI infrastructure.

### Principles

**Glanceable.** The most important information (current loss, step count, ETA) is always visible without scrolling.

**Non-destructive.** The TUI is a view layer. It never modifies training state. Closing it doesn't stop training.

**Keyboard-first.** Every action is reachable via keyboard. Mouse is optional.

### Vision

```
┌─ nanogpt training ──────────────────────────────────┐
│                                                      │
│  Model: gpt2-small (124M params)                     │
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
- All data available via the CLI `--json` interface (TUI is just a renderer)
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
