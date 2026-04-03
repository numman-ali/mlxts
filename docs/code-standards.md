# Code Standards

This project is maintained primarily by AI agents but read and collaborated on by humans. The code must be beautiful, clear, and self-documenting. Structure and discipline are not afterthoughts — they are core to the project's identity.

## Philosophy

**Code is read far more than it is written.** Every function, every type, every file should be immediately understandable to a competent TypeScript developer who has never seen this codebase before.

**Self-documenting means the code explains itself.** Names, types, and structure do the heavy lifting. Comments explain *why*, never *what*.

**Simplicity is not laziness.** The simplest correct solution is the best solution. Cleverness that obscures intent is a defect.

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
- **Enforced by:** `bun run check:assertions` scans production code for `as` casts (Biome has no rule for this). Non-null assertions (`!`) and `any` are enforced by Biome at `error` level, with overrides for `ffi.ts` (boundary) and test files.

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

### No `any` except at the FFI boundary

The FFI layer (`ffi.ts`) may use `any` or a minimal, documented type assertion where Bun's FFI types require it. Nowhere else. If you're tempted to use either in normal code, the type design needs improvement.

### Treat type assertions as a design smell

- Avoid `as`, non-null assertions (`!`), and other type escape hatches in production code
- If an assertion is truly unavoidable, isolate it at the FFI boundary and keep it local to the helper that needs it
- Do not use type assertions to silence real uncertainty; improve the types, add validation, or narrow the value first
- Prefer runtime checks that teach the type system something true over casts that merely quiet the compiler

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
- [ ] **No type assertions outside `ffi.ts`** — if `as` or `!` appears in ops, nn, or application code, the design is incomplete.
- [ ] **ABI correctness** — FFI symbol declarations in `ffi.ts` must match the actual mlx-c header signatures. Creation functions return by value; operations use output pointers; property getters return directly.
- [ ] **Resource cleanup** — every native handle temporary (`mlx_vector_array`, device handles, RNG key splits) uses `try/finally`.
- [ ] **`bun run typecheck` passes** — code is not review-ready until static types are clean.
- [ ] **Error messages are actionable** — include what was expected, what was received, and where.

## Git

- Commit messages: imperative mood, explain why not what
- One logical change per commit
- Agents include `Co-Authored-By` in commit messages
- Branch naming: `phase-1/core-bindings`, `phase-2/autograd`, etc.
