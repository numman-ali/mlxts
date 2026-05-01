import type { MxArray } from "@mlxts/core";
import {
  add,
  asType,
  divide,
  flatten,
  formatShape,
  greater,
  max,
  mean,
  min,
  multiply,
  reshape,
  retainArray,
  sqrt,
  square,
  subtract,
  sum,
  where,
} from "@mlxts/core";

import type { Ltx2TextConnectorsConfig } from "./config";

function expectHiddenStack(
  hiddenStates: MxArray,
  config: Ltx2TextConnectorsConfig,
): { batch: number; length: number } {
  const [batch, length, channels, layers] = hiddenStates.shape;
  if (
    hiddenStates.shape.length !== 4 ||
    batch === undefined ||
    length === undefined ||
    channels !== config.captionChannels ||
    layers !== config.textProjInFactor
  ) {
    throw new Error(
      `LTX2TextConnectors: text hidden states must have shape [batch, length, ${config.captionChannels}, ${config.textProjInFactor}], got ${formatShape(
        hiddenStates.shape,
      )}.`,
    );
  }
  return { batch, length };
}

function hiddenStack(
  hiddenStates: MxArray,
  config: Ltx2TextConnectorsConfig,
): { stack: MxArray; ownsStack: boolean; batch: number; length: number } {
  const [batch, length, channels] = hiddenStates.shape;
  if (
    hiddenStates.shape.length === 3 &&
    batch !== undefined &&
    length !== undefined &&
    channels === config.textEncoderDim
  ) {
    return {
      stack: reshape(hiddenStates, [
        batch,
        length,
        config.captionChannels,
        config.textProjInFactor,
      ]),
      ownsStack: true,
      batch,
      length,
    };
  }
  const shape = expectHiddenStack(hiddenStates, config);
  return { stack: hiddenStates, ownsStack: false, batch: shape.batch, length: shape.length };
}

function expectAttentionMask(mask: MxArray, batch: number, length: number): MxArray {
  if (mask.shape.length !== 2 || mask.shape[0] !== batch || mask.shape[1] !== length) {
    throw new Error(
      `LTX2TextConnectors: attention mask must have shape [${batch}, ${length}], got ${formatShape(
        mask.shape,
      )}.`,
    );
  }
  return greater(mask, 0);
}

/** Flatten Gemma per-layer hidden states into Diffusers LTX-2 connector input form. */
export function ltx2FlattenTextHiddenStates(
  hiddenStates: MxArray,
  config: Ltx2TextConnectorsConfig,
): MxArray {
  const { stack, ownsStack, batch, length } = hiddenStack(hiddenStates, config);
  try {
    return reshape(stack, [batch, length, config.textEncoderDim]);
  } finally {
    if (ownsStack) {
      stack.free();
    }
  }
}

/** Normalize a Gemma hidden-state stack using LTX-2.0 masked mean/range semantics. */
export function ltx2PerLayerMaskedMeanNorm(
  hiddenStates: MxArray,
  attentionMask: MxArray,
  config: Ltx2TextConnectorsConfig,
  scaleFactor = 8,
): MxArray {
  const { stack, ownsStack, batch, length } = hiddenStack(hiddenStates, config);
  const boolMask = expectAttentionMask(attentionMask, batch, length);
  try {
    using mask4d = reshape(boolMask, [batch, length, 1, 1]);
    using maskFloat = asType(mask4d, stack.dtype);
    using masked = multiply(stack, maskFloat);
    using maskedSum = sum(masked, [1, 2], true);
    using tokenMask = asType(boolMask, stack.dtype);
    using validTokens = sum(tokenMask, 1, false);
    using rawDenominator = reshape(multiply(validTokens, config.captionChannels), [batch, 1, 1, 1]);
    using denominator = add(rawDenominator, 1e-6);
    using meanValues = divide(maskedSum, denominator);
    using minMasked = where(mask4d, stack, Number.POSITIVE_INFINITY);
    using maxMasked = where(mask4d, stack, Number.NEGATIVE_INFINITY);
    using minValues = min(minMasked, [1, 2], true);
    using maxValues = max(maxMasked, [1, 2], true);
    using centered = subtract(stack, meanValues);
    using range = add(subtract(maxValues, minValues), 1e-6);
    using safeRange = where(greater(range, 0), range, 1);
    using normalized = divide(centered, safeRange);
    using scaled = multiply(normalized, scaleFactor);
    using flat = flatten(scaled, 2, 3);
    using mask3d = reshape(boolMask, [batch, length, 1]);
    return where(mask3d, flat, 0);
  } finally {
    boolMask.free();
    if (ownsStack) {
      stack.free();
    }
  }
}

/** Normalize each token over caption channels before LTX-2.3 per-modality projection. */
export function ltx2PerTokenRmsNorm(
  hiddenStates: MxArray,
  attentionMask: MxArray,
  config: Ltx2TextConnectorsConfig,
): MxArray {
  const { stack, ownsStack, batch, length } = hiddenStack(hiddenStates, config);
  const boolMask = expectAttentionMask(attentionMask, batch, length);
  try {
    using squared = square(stack);
    using variance = mean(squared, 2, true);
    using denom = sqrt(add(variance, 1e-6));
    using normalized = divide(stack, denom);
    using flat = flatten(normalized, 2, 3);
    using mask3d = reshape(boolMask, [batch, length, 1]);
    return where(mask3d, flat, 0);
  } finally {
    boolMask.free();
    if (ownsStack) {
      stack.free();
    }
  }
}

/** Retain a binary connector mask after validating the text batch shape. */
export function ltx2ConnectorBinaryMask(
  attentionMask: MxArray,
  batch: number,
  length: number,
): MxArray {
  using mask = expectAttentionMask(attentionMask, batch, length);
  return asType(mask, "int32");
}

/** Return an owned stack view that tests can dispose explicitly. */
export function retainLtx2HiddenStack(
  hiddenStates: MxArray,
  config: Ltx2TextConnectorsConfig,
): MxArray {
  const { stack, ownsStack } = hiddenStack(hiddenStates, config);
  if (ownsStack) {
    return stack;
  }
  return retainArray(stack);
}
