import type { MxArray } from "./array";
import { checkStatus } from "./error";
import type { GradFn, MultiOutputFn } from "./ffi/closure-bridge";
import { ffi } from "./ffi/lib";
import {
  applyClosureMultiTransform,
  applyClosureTransform,
  attachDisposableTransform,
  type CachedClosureMultiTransform,
  type CachedClosureTransform,
  closureMultiTransformRegistry,
  closureTransformRegistry,
  createCachedClosureMultiTransform,
  createCachedClosureTransform,
  type DisposableTransform,
  disposeCachedClosureMultiTransform,
  disposeCachedClosureTransform,
} from "./transforms-base";

export type CompileMode = "disabled" | "no_simplify" | "no_fuse" | "enabled";

const COMPILE_MODE_TO_NATIVE: Record<CompileMode, number> = {
  disabled: 0,
  no_simplify: 1,
  no_fuse: 2,
  enabled: 3,
};

export function compile(
  fn: GradFn,
  options?: { shapeless?: boolean },
): DisposableTransform<(...args: MxArray[]) => MxArray> {
  let cached: CachedClosureTransform | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): MxArray => {
      if (disposed) {
        throw new Error("compile: transform has already been disposed");
      }
      if (cached === null) {
        cached = createCachedClosureTransform(
          fn,
          "compile",
          (out, closure) => {
            checkStatus(ffi.mlx_compile(out, closure, options?.shapeless ?? false), "compile");
          },
          (transform) => {
            ffi.mlx_closure_free(transform);
          },
        );
        closureTransformRegistry.register(transformed, cached, transformed);
      }
      return applyClosureTransform(cached, args, "compile_apply");
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      closureTransformRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedClosureTransform(cached);
        cached = null;
      }
    },
  );

  return transformed;
}

export function checkpoint(fn: GradFn): DisposableTransform<(...args: MxArray[]) => MxArray> {
  let cached: CachedClosureTransform | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): MxArray => {
      if (disposed) {
        throw new Error("checkpoint: transform has already been disposed");
      }
      if (cached === null) {
        cached = createCachedClosureTransform(
          fn,
          "checkpoint",
          (out, closure) => {
            checkStatus(ffi.mlx_checkpoint(out, closure), "checkpoint");
          },
          (transform) => {
            ffi.mlx_closure_free(transform);
          },
        );
        closureTransformRegistry.register(transformed, cached, transformed);
      }
      return applyClosureTransform(cached, args, "checkpoint_apply");
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      closureTransformRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedClosureTransform(cached);
        cached = null;
      }
    },
  );

  return transformed;
}

/**
 * Compile a multi-output function. Like `compile()` but supports functions
 * that return multiple arrays. Used for fusing multiple head projections
 * into a single compiled graph.
 */
export function compileMany(
  fn: MultiOutputFn,
  options?: { shapeless?: boolean },
): DisposableTransform<(...args: MxArray[]) => MxArray[]> {
  let cached: CachedClosureMultiTransform | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): MxArray[] => {
      if (disposed) {
        throw new Error("compileMany: transform has already been disposed");
      }
      if (cached === null) {
        cached = createCachedClosureMultiTransform(
          fn,
          "compileMany",
          (out, closure) => {
            checkStatus(ffi.mlx_compile(out, closure, options?.shapeless ?? false), "compileMany");
          },
          (transform) => {
            ffi.mlx_closure_free(transform);
          },
        );
        closureMultiTransformRegistry.register(transformed, cached, transformed);
      }
      return applyClosureMultiTransform(cached, args, "compileMany_apply");
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      closureMultiTransformRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedClosureMultiTransform(cached);
        cached = null;
      }
    },
  );

  return transformed;
}

export function clearCompileCache(): void {
  checkStatus(ffi.mlx_detail_compile_clear_cache(), "detail_compile_clear_cache");
}

export function enableCompile(): void {
  checkStatus(ffi.mlx_enable_compile(), "enable_compile");
}

export function disableCompile(): void {
  checkStatus(ffi.mlx_disable_compile(), "disable_compile");
}

export function setCompileMode(mode: CompileMode): void {
  checkStatus(ffi.mlx_set_compile_mode(COMPILE_MODE_TO_NATIVE[mode]), "set_compile_mode");
}
