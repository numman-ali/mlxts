# Core Compile Barrel Cleanup Review

## Summary

Removed compile-control runtime knobs from the `@mlxts/core` top-level barrel.
The semantic transform constructors remain public: `compile`, `compileMany`,
`checkpoint`, `grad`, `valueAndGrad`, `mxEval`, and `mxAsyncEval`.

## Files Reviewed

- `packages/core/src/index.ts`

## Tensor Lifetime Audit

No tensor-producing primitives, transform implementations, native ownership, or
FFI calls changed. The production change is limited to top-level export shape.

## Memory / Performance Evidence

This tranche makes no performance claim. No runtime execution path changed, and
`bench:generation` / `bench:generation:parity` are not applicable to this barrel
export cleanup.

Focused validation passed:

- `bun test packages/core/src/index.test.ts packages/core/src/transforms.test.ts` passed: 40 tests.
- `bun run validate`

## Independent Review

The audit and `packages/core/AGENTS.md` both place compile-strategy plumbing
under `transforms-*.ts` and reserve the public barrel for semantic names. I
searched for `clearCompileCache`, `disableCompile`, `enableCompile`, and
`setCompileMode` usages before editing; the only public leak was
`packages/core/src/index.ts`, with core-local transform tests already importing
the controls from `./transforms`.

## Remaining Risks / Follow-ups

No residual call-site rewiring was needed because no non-core package consumed
these top-level exports.
