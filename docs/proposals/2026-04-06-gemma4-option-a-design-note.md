# Design Note: Option A Cache-Contract Refactor Questions

## Status

This note is historical context for the earlier Gemma 4 cache-contract
investigation. It is no longer the active mainline next step after the
readability reset and native-seam retrospective recorded on 2026-04-08.

## Purpose

This note exists to force the next implementation phase to answer the contract
questions before any cache refactor code starts.

It is intentionally short. The goal is to pin down the private boundary and the
expected leverage, not to pre-write the implementation.

## Questions To Answer

### 1. Which private boundary changes?

Choose exactly one as the primary target:

- cache internals only
- cache/attention boundary
- deeper core/native boundary

If more than one boundary changes, explain why the simpler boundary was
insufficient.

### 2. What is the steady-state `updateAndFetch()` return contract?

Pick one:

- caller owns returned arrays
- cache owns returned arrays and caller borrows them
- stable wrapper object with internal pointer replacement
- raw-pointer internal state with wrapped boundary return
- custom native helper keeps stable cache identity

The chosen contract must explain:

- who frees what
- how attention disposal changes
- how mixed full/sliding cache families stay coherent under the same model code

### 3. What is the irreducible wrapper floor if no native helper is added?

Quantify the minimum expected wrapper events per cache-owning layer per token
for:

- full cache
- saturated sliding cache
- mixed-pattern cache

The point is to decide whether a pure TypeScript/private-contract refactor is
enough before adding custom native work.

### 4. Which existing `mlx-c` primitives are exhausted first?

Before proposing custom C++, state explicitly why current primitives such as:

- `mlx_slice_update`
- `mlx_slice_update_dynamic`
- `mlx_scatter`
- `mlx_put_along_axis`

are not sufficient for the chosen contract.

### 5. What is the expected reduction per decode token?

Estimate, separately for:

- wrapper creations
- retain/free cycles
- returned view churn
- cache buffer replacements

The estimate does not need to be perfect, but it must be concrete enough that a
reviewer can tell whether the design is worth trying.

## Candidate Shapes Worth Evaluating

These are all in-bounds for the design note. None should be ignored.

### A1. Borrowed cache-owned returns

The cache owns steady-state buffers and attention borrows them.

### A2. Raw-pointer cache internals

Cache state is stored as raw native pointers; TypeScript wraps only at the
boundary where that is still needed.

### A3. Stable-wrapper pointer swap

Stable JS-visible cache wrappers survive while their underlying native pointers
change.

### B1. Thin custom C++ cache write helper

One small native helper keeps buffer identity stable across hot writes.

### B2. Native cache module

The entire cache state machine moves below the TypeScript coordination layer.

## Completion Criteria

This note is ready for implementation review when:

1. one primary shape is recommended
2. one fallback shape is named
3. the ownership contract is explicit
4. the expected leverage is written down
5. the reason for not choosing the other shapes is stated clearly
