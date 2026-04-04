# Code Standards

This project is maintained primarily by AI agents but read and collaborated on by humans. The code must be beautiful, clear, and self-documenting. Structure and discipline are not afterthoughts — they are core to the project's identity.

## Philosophy

**Code is read far more than it is written.** Every function, every type, every file should be immediately understandable to a competent TypeScript developer who has never seen this codebase before.

**Self-documenting means the code explains itself.** Names, types, and structure do the heavy lifting. Comments explain *why*, never *what*.

**Simplicity is not laziness.** The simplest correct solution is the best solution. Cleverness that obscures intent is a defect.

**This repo is forward-only.** Do not keep legacy compatibility layers, deprecated modes, stale file formats, or “just in case” code paths. If we no longer want to support something, remove it cleanly and update the docs in the same change.

**Bun is the runtime.** Do not design for multiple JS runtimes. Prefer Bun-native APIs and Bun runtime semantics. Avoid `node:*` imports; if you need a standard-library helper that Bun already supports, use the neutral module specifier under Bun rather than a Node-branded one.

## Naming

### Files

- Lowercase with hyphens: `causal-attention.ts`, `layer-norm.ts`
- One primary export per file (the thing the file is named after)
- Test files alongside source: `linear.ts` → `linear.test.ts`
- Index files only re-export — no logic in `index.ts`

### Variables and functions

- **Descriptive over short**: `queryProjection` not `qProj`, `batchSize` not `bs`
- **Exception**: well-known ML conventions are kept: `x`, `y` for input/output in forward passes, `Q`, `K`, `V` for query/key/value, `lr` for learning rate
- **Boolean names**: use `is`/`has`/`should` prefix: `isTraining`, `hasBias`
- **Function names**: verb-first: `computeAttention`, `createMask`, `loadWeights`

### Types and classes

- PascalCase: `MxArray`, `TransformerBlock`, `TrainingConfig`
- Generic type params: single letter is fine for simple cases (`T`), descriptive for complex (`TConfig extends ModelConfig`)
- Enum-like constants: use `as const` objects, not TypeScript enums

### Constants

- UPPER_SNAKE_CASE for true constants: `MAX_SEQUENCE_LENGTH`, `DEFAULT_LEARNING_RATE`
- Regular camelCase for computed or configured values

## Structure

### File organization

```typescript
// 1. Imports (external, then internal, blank line between groups)
import { dlopen } from "bun:ffi";

import { MxArray } from "../core/array";
import { ffi } from "../core/ffi";

// 2. Types and interfaces
interface LinearConfig {
  inputFeatures: number;
  outputFeatures: number;
  hasBias?: boolean;
}

// 3. Constants (if any)
const DEFAULT_CONFIG: LinearConfig = { ... };

// 4. Main export (the thing this file is about)
export class Linear extends Module {
  ...
}

// 5. Helper functions (private to this file, below the main export)
function initializeWeight(shape: number[]): MxArray {
  ...
}
```

### Function length

- Aim for functions under 30 lines
- If a function is longer, it probably has extractable sub-steps
- Exception: training loops and similar orchestration can be longer if the flow is linear and clear

### Module organization

```
src/
├── core/           # Foundation — tensor ops, FFI, memory
│   ├── array.ts
│   ├── ops.ts
│   └── ...
├── nn/             # Neural network layers
│   ├── module.ts   # Base class first
│   ├── linear.ts
│   └── ...
├── optimizers/     # Optimization algorithms
│   ├── optimizer.ts
│   └── ...
└── index.ts        # Public API surface
```

## TypeScript Practices

### Strict mode, always

```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true
  }
}
```

A clean `bun run typecheck` is a required development gate, not optional polish. If the type checker is unhappy, the design is incomplete.

### Use the type system fully

```typescript
// Good: types document the contract
type DType = "float32" | "float16" | "bfloat16" | "int32" | "int64" | "bool";

function zeros(shape: number[], dtype: DType = "float32"): MxArray;

// Bad: stringly typed, no autocomplete, no safety
function zeros(shape: number[], dtype: string): any;
```

### Avoid type assertions

Type assertions such as `value as SomeType` or `as unknown as` are a last resort, not a normal workflow.

- Prefer narrowing, discriminated unions, typed helper functions, and precise return types.
- If an assertion is unavoidable at the FFI boundary, isolate it in one tiny helper and explain why the boundary needs it.
- Do not spread boundary assertions into higher-level tensor, nn, optimizer, or application code.
- Never use a type assertion to silence a real type design problem.
- **Enforced by:** `bun run check:assertions` scans production code for `as` casts (Biome has no rule for this). Non-null assertions (`!`) and `any` are enforced by Biome at `error` level, with overrides for `src/core/ffi/` (boundary package) and test files.

### Prefer `readonly` for data that shouldn't change

```typescript
class MxArray {
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly ndim: number;
}
```

### Use `using` for resource management

```typescript
// In hot loops, explicit disposal prevents GC pressure
function trainStep(model: GPT, batch: Batch): number {
  using logits = model.forward(batch.input);
  using loss = crossEntropy(logits, batch.target);
  // loss is disposed when scope exits
  return loss.item();
}
```

### Keep tensor lifetimes visible

In runtime-sensitive code, do not hide disposable `MxArray` values inside nested expressions.

```typescript
// Good: every disposable intermediate has a visible lifetime
using reshaped = reshape(x, [batchSize, sequenceLength, heads, headDim]);
using transposed = transpose(reshaped, [0, 2, 1, 3]);

// Bad: the inner reshape returns an MxArray that is easy to miss in review
using transposed = transpose(reshape(x, [batchSize, sequenceLength, heads, headDim]), [0, 2, 1, 3]);
```

Rules:

- If an operation returns `MxArray` and you are not returning it directly, bind it to a local name.
- Prefer `using` for lexical ownership and `try/finally` for non-lexical native handle cleanup.
- Keep `eval()` / `asyncEval()` / `synchronize()` calls explicit and justified. Hidden synchronization is a correctness and performance bug source.
- Runtime-sensitive code should read linearly to a TypeScript developer who is checking ownership by eye.
- FFI result-pointer writes must use per-call `OutSlot`-style helpers. Shared reusable output buffers are not allowed.
- Transforms that hold native resources beyond a single invocation should expose explicit disposal rather than depending on GC timing.
- `bun run check:tensor-lifetimes` is the fast static backstop for this rule. It is intentionally narrow, AST-based, and should stay high-signal.
- The canonical tracked tensor-op list lives in `scripts/`; when a new tensor-producing primitive is added, update that list instead of teaching the codebase to ignore the smell.

### Runtime incidents must leave behind a stronger system

When a crash, leak, or major performance regression is fixed, the same change must also add at least one preventive improvement:

- a direct regression test
- a benchmark or soak script
- a stricter validation gate
- a documented repo rule

Fixing the bug is not enough. The repo has to learn from it.

### No `any` except at the FFI boundary

The FFI boundary package (`src/core/ffi/`) may use `any` or a minimal, documented type assertion where Bun's FFI types require it. Nowhere else. If you're tempted to use either in normal code, the type design needs improvement.

### Treat type assertions as a design smell

- Avoid `as`, non-null assertions (`!`), and other type escape hatches in production code
- If an assertion is truly unavoidable, isolate it at the FFI boundary and keep it local to the helper that needs it
- Do not use type assertions to silence real uncertainty; improve the types, add validation, or narrow the value first
- Prefer runtime checks that teach the type system something true over casts that merely quiet the compiler
- Use type guard functions (e.g., `isParameterTree()`) to narrow union types without `as` casts

### nn module convention

Public `MxArray` and `Module` fields are scanned as parameters by `Module.parameters()`. Internal state that is not a parameter (config values, caches, running statistics) must use JS `#` private fields or be a non-MxArray type. This ensures the property-scanning approach correctly identifies only learnable parameters.

The scan keys are cached after the first parameter walk. Public parameter and sub-module fields therefore need to be assigned during construction or as class field initializers before the first call to `parameters()` / `trainableParameters()`.

Shared public parameter aliases are still not supported. Functional weight tying is fine, but do not register the same logical parameter through multiple public fields until the parameter tree and optimizer layers grow explicit alias semantics.

### Typecheck is part of the contract

- `bun run typecheck` must pass before code is considered review-ready
- Type errors are not "cleanup later" work; they mean the design contract is still incomplete
- A passing test suite does not compensate for failing static types

## Comments

### When to comment

- **Why** something exists (not what it does)
- **Non-obvious constraints** ("MLX requires eval() before reading array data")
- **ML-specific context** for developers new to ML ("Causal mask prevents attending to future tokens")
- **Performance notes** ("Using in-place op to avoid allocation in the training loop")

### When not to comment

- What the code literally does (the code says that)
- Type information (TypeScript says that)
- Obvious patterns
- Commented-out code (delete it; git has history)

### JSDoc for public APIs

Every exported function, class, and type gets a JSDoc block:

```typescript
/**
 * Multi-head causal self-attention.
 *
 * Splits input into multiple heads, computes scaled dot-product attention
 * with a causal mask, and projects back to the original dimension.
 */
export class CausalSelfAttention extends Module {
  /**
   * @param x - Input tensor of shape [batch, sequence, embedding]
   * @returns Output tensor of same shape as input
   */
  forward(x: MxArray): MxArray { ... }
}
```

## Testing

### Every module has tests

- Test file lives next to source: `linear.ts` → `linear.test.ts`
- Test the public API, not internals
- Each test should be understandable in isolation
- Exported core functions need direct unit coverage, not just incidental coverage through smoke tests

### Coverage is part of the contract

- `mlx-ts` must stay at or above `95%` line coverage and `90%` function coverage
- `nanogpt` must stay at or above `90%` line coverage and `85%` function coverage
- Enforced by: `bun run check:coverage` for package coverage, plus `bun run check:runtime-review` for runtime-sensitive diffs. If coverage reports branch counters, the gate also enforces branch coverage instead of inventing one.
- `bun run validate` includes both gates and is the standard pre-commit/review-ready path
- Prefer tests that exercise real behavior, edge cases, and failure paths over tests written only to bump percentages
- Dynamic paths count: default-device switching, autograd error propagation, shape/axis variants, and cleanup paths all need explicit coverage
- Long-running acceptance scripts are separate from `validate`, but the repo should still provide canonical scripted paths for them
- Long-running training control should be checkpoint-first: graceful stop + resume beats hidden in-memory pause state
- The supervised run-manager flow under `packages/nanogpt/src/run/` is production code, not glue. Test it, review it, and keep it under the same quality gates as the model code.
- Snapshot checkpoints are for frequent model saves; resume checkpoints are for exact continuation with optimizer state. Do not blur those meanings in code or docs.
- Use `bun run bench:memory` and the `bun run soak:gpt-*` ladder before trusting a new hot-path optimization or an overnight run.
- Runtime-sensitive diffs must add or update a review artifact under `docs/reviews/`. The review record is part of the deliverable, not optional process overhead, and its `Files Reviewed` section must name the exact changed runtime-sensitive files.

### Test naming

```typescript
describe("Linear", () => {
  test("forward produces correct output shape", () => { ... });
  test("forward with bias adds bias term", () => { ... });
  test("forward without bias omits bias term", () => { ... });
  test("gradients flow through forward pass", () => { ... });
});
```

### Test structure

```typescript
test("matmul produces correct result for known values", () => {
  // Arrange
  const a = mx.array([[1, 2], [3, 4]]);
  const b = mx.array([[5, 6], [7, 8]]);

  // Act
  const result = mx.matmul(a, b);
  mx.eval(result);

  // Assert
  expect(result.toList()).toEqual([[19, 22], [43, 50]]);
});
```

## Error Handling

- At the FFI boundary: check return values, throw typed errors
- In nn layers: validate shapes at construction time, fail fast with clear messages
- In training loops: catch and report, don't silently continue
- Never swallow errors. Never use empty catch blocks.

```typescript
// Good: clear, actionable error
if (weight.shape[0] !== inputFeatures) {
  throw new Error(
    `Linear: weight shape[0] is ${weight.shape[0]}, expected ${inputFeatures}. ` +
    `Weight shape: [${weight.shape}], input features: ${inputFeatures}`
  );
}
```

## Review Checklist

When reviewing code (whether written by a human or an agent), check for:

- [ ] **"Where did we teach TypeScript the truth?"** — not just "did the compiler go green?" Types should express real invariants, not suppress inconvenient errors.
- [ ] **No type assertions outside `src/core/ffi/`** — if `as` or `!` appears in ops, nn, or application code, the design is incomplete.
- [ ] **No legacy compatibility scaffolding** — if we are carrying deprecated formats or fallback modes “just in case,” the surface is not clean enough.
- [ ] **ABI correctness** — FFI symbol declarations in `src/core/ffi/symbols.ts` must match the actual mlx-c header signatures. Creation functions return by value; operations use output pointers; property getters return directly.
- [ ] **Resource cleanup** — every native handle temporary (`mlx_vector_array`, device handles, RNG key splits) uses `try/finally`.
- [ ] **Tensor lifetimes are locally visible** — no anonymous disposable `MxArray` intermediates hiding inside nested hot-path expressions.
- [ ] **`bun run typecheck` passes** — code is not review-ready until static types are clean.
- [ ] **`bun run check:runtime-review` passes when required** — runtime-sensitive diffs need a `docs/reviews/` artifact with the required sections, and the `Files Reviewed` list must match the changed runtime-sensitive files.
- [ ] **`bun run check:coverage` passes** — `mlx-ts` stays at or above `95%` lines / `90%` funcs and `nanogpt` stays at or above `90%` lines / `85%` funcs.
- [ ] **Unit tests cover exported behavior directly** — don’t rely on one or two broad smoke tests to “accidentally” hit important branches.
- [ ] **Error messages are actionable** — include what was expected, what was received, and where.
- [ ] **The fix improved the system** — incidents must leave behind a rule, test, benchmark, or gate that would have caught them sooner next time.

## Git

- Commit messages: imperative mood, explain why not what
- One logical change per commit
- Agents include `Co-Authored-By` in commit messages
- Branch naming: `phase-1/core-bindings`, `phase-2/autograd`, etc.
