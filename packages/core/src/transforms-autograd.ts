import type { MxArray } from "./array";
import type { GradFn } from "./ffi/closure-bridge";
import {
  applyValueAndGrad,
  attachDisposableTransform,
  type CachedValueAndGrad,
  createCachedValueAndGrad,
  type DisposableTransform,
  disposeCachedValueAndGrad,
  normalizeArgnums,
  sameIndices,
  valueAndGradRegistry,
} from "./transforms-base";

export function valueAndGrad(
  fn: GradFn,
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray]>;
export function valueAndGrad(
  fn: GradFn,
  argnums: number,
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray]>;
export function valueAndGrad(
  fn: GradFn,
  argnums: number[],
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray[]]>;
export function valueAndGrad(
  fn: GradFn,
  argnums?: number | number[],
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray] | [MxArray, MxArray[]]> {
  const isMulti = Array.isArray(argnums);
  let cached: CachedValueAndGrad | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): [MxArray, MxArray] | [MxArray, MxArray[]] => {
      if (disposed) {
        throw new Error("valueAndGrad: transform has already been disposed");
      }
      const indices = normalizeArgnums(argnums, args.length);
      if (cached === null) {
        cached = createCachedValueAndGrad(fn, indices);
        valueAndGradRegistry.register(transformed, cached, transformed);
      } else if (!sameIndices(cached.argnums, indices)) {
        throw new Error(
          `valueAndGrad: argnums changed between calls. Expected [${cached.argnums.join(", ")}], got [${indices.join(", ")}].`,
        );
      }
      const { value, grads } = applyValueAndGrad(cached, args);

      if (isMulti) {
        return [value, grads];
      }
      const singleGrad = grads[0];
      if (singleGrad === undefined) {
        throw new Error("valueAndGrad: gradient vector was empty");
      }
      for (let index = 1; index < grads.length; index++) {
        grads[index]?.free();
      }
      return [value, singleGrad];
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      valueAndGradRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedValueAndGrad(cached);
        cached = null;
      }
    },
  );

  return transformed;
}

export function grad(fn: GradFn): DisposableTransform<(...args: MxArray[]) => MxArray>;
export function grad(
  fn: GradFn,
  argnums: number,
): DisposableTransform<(...args: MxArray[]) => MxArray>;
export function grad(
  fn: GradFn,
  argnums: number[],
): DisposableTransform<(...args: MxArray[]) => MxArray[]>;
export function grad(
  fn: GradFn,
  argnums?: number | number[],
): DisposableTransform<(...args: MxArray[]) => MxArray | MxArray[]> {
  const isMulti = Array.isArray(argnums);
  let cached: CachedValueAndGrad | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): MxArray | MxArray[] => {
      if (disposed) {
        throw new Error("grad: transform has already been disposed");
      }
      const indices = normalizeArgnums(argnums, args.length);
      if (cached === null) {
        cached = createCachedValueAndGrad(fn, indices);
        valueAndGradRegistry.register(transformed, cached, transformed);
      } else if (!sameIndices(cached.argnums, indices)) {
        throw new Error(
          `grad: argnums changed between calls. Expected [${cached.argnums.join(", ")}], got [${indices.join(", ")}].`,
        );
      }
      const { value, grads } = applyValueAndGrad(cached, args);
      value.free();

      if (isMulti) {
        return grads;
      }
      const singleGrad = grads[0];
      if (singleGrad === undefined) {
        throw new Error("grad: gradient vector was empty");
      }
      for (let index = 1; index < grads.length; index++) {
        grads[index]?.free();
      }
      return singleGrad;
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      valueAndGradRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedValueAndGrad(cached);
        cached = null;
      }
    },
  );

  return transformed;
}
