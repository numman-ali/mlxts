import type { MxArray } from "@mlxts/core";
import { retainArray, split } from "@mlxts/core";
import { Linear, Module, silu } from "@mlxts/nn";

import {
  affineFreeLayerNorm,
  applyScaleShift,
  assertSequence3d,
  freeArrays,
  reshapeModulation,
} from "./tensor-utils";

export type StableDiffusion3AdaLayerNormZeroOutput = {
  hiddenStates: MxArray;
  gateMsa: MxArray;
  shiftMlp: MxArray;
  scaleMlp: MxArray;
  gateMlp: MxArray;
};

export type StableDiffusion35AdaLayerNormZeroXOutput = StableDiffusion3AdaLayerNormZeroOutput & {
  hiddenStates2: MxArray;
  gateMsa2: MxArray;
};

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

export function disposeStableDiffusion3AdaLayerNormZero(
  output: StableDiffusion3AdaLayerNormZeroOutput,
): void {
  output.hiddenStates.free();
  output.gateMsa.free();
  output.shiftMlp.free();
  output.scaleMlp.free();
  output.gateMlp.free();
}

export function disposeStableDiffusion35AdaLayerNormZeroX(
  output: StableDiffusion35AdaLayerNormZeroXOutput,
): void {
  disposeStableDiffusion3AdaLayerNormZero(output);
  output.hiddenStates2.free();
  output.gateMsa2.free();
}

/** SD3 AdaLN-Zero image/context modulation. */
export class StableDiffusion3AdaLayerNormZero extends Module {
  linear: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number) {
    super();
    this.linear = new Linear(hiddenSize, hiddenSize * 6);
    this.#hiddenSize = hiddenSize;
  }

  forward(x: MxArray, emb?: MxArray): MxArray {
    if (emb === undefined) {
      throw new Error("StableDiffusion3AdaLayerNormZero.forward: emb is required.");
    }
    const output = this.modulate(x, emb);
    try {
      return retainArray(output.hiddenStates);
    } finally {
      disposeStableDiffusion3AdaLayerNormZero(output);
    }
  }

  /** Return hidden states plus residual gates for an SD3 transformer block. */
  modulate(x: MxArray, emb: MxArray): StableDiffusion3AdaLayerNormZeroOutput {
    const shape = assertSequence3d(x, "StableDiffusion3AdaLayerNormZero.modulate");
    if (shape.channels !== this.#hiddenSize) {
      throw new Error("StableDiffusion3AdaLayerNormZero.modulate: hidden size mismatch.");
    }
    if (
      emb.shape.length !== 2 ||
      emb.shape[0] !== shape.batch ||
      emb.shape[1] !== this.#hiddenSize
    ) {
      throw new Error(
        "StableDiffusion3AdaLayerNormZero.modulate: emb shape must match [batch, hiddenSize].",
      );
    }
    using activated = silu(emb);
    using projected = this.linear.forward(activated);
    const parts = split(projected, 6, -1);
    try {
      const shiftMsa = reshapeModulation(
        partAt(parts, 0, "StableDiffusion3AdaLayerNormZero.modulate"),
        shape.batch,
        this.#hiddenSize,
      );
      const scaleMsa = reshapeModulation(
        partAt(parts, 1, "StableDiffusion3AdaLayerNormZero.modulate"),
        shape.batch,
        this.#hiddenSize,
      );
      try {
        using normalized = affineFreeLayerNorm(
          x,
          this.#hiddenSize,
          "StableDiffusion3AdaLayerNormZero.modulate",
        );
        return {
          hiddenStates: applyScaleShift(normalized, shiftMsa, scaleMsa),
          gateMsa: reshapeModulation(
            partAt(parts, 2, "StableDiffusion3AdaLayerNormZero.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
          shiftMlp: reshapeModulation(
            partAt(parts, 3, "StableDiffusion3AdaLayerNormZero.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
          scaleMlp: reshapeModulation(
            partAt(parts, 4, "StableDiffusion3AdaLayerNormZero.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
          gateMlp: reshapeModulation(
            partAt(parts, 5, "StableDiffusion3AdaLayerNormZero.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
        };
      } finally {
        shiftMsa.free();
        scaleMsa.free();
      }
    } finally {
      freeArrays(parts);
    }
  }
}

/** SD3.5 AdaLN-Zero-X modulation for dual-attention layers. */
export class StableDiffusion35AdaLayerNormZeroX extends Module {
  linear: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number) {
    super();
    this.linear = new Linear(hiddenSize, hiddenSize * 9);
    this.#hiddenSize = hiddenSize;
  }

  forward(x: MxArray, emb?: MxArray): MxArray {
    if (emb === undefined) {
      throw new Error("StableDiffusion35AdaLayerNormZeroX.forward: emb is required.");
    }
    const output = this.modulate(x, emb);
    try {
      return retainArray(output.hiddenStates);
    } finally {
      disposeStableDiffusion35AdaLayerNormZeroX(output);
    }
  }

  /** Return both image attention inputs plus residual gates for SD3.5. */
  modulate(x: MxArray, emb: MxArray): StableDiffusion35AdaLayerNormZeroXOutput {
    const shape = assertSequence3d(x, "StableDiffusion35AdaLayerNormZeroX.modulate");
    if (shape.channels !== this.#hiddenSize) {
      throw new Error("StableDiffusion35AdaLayerNormZeroX.modulate: hidden size mismatch.");
    }
    if (
      emb.shape.length !== 2 ||
      emb.shape[0] !== shape.batch ||
      emb.shape[1] !== this.#hiddenSize
    ) {
      throw new Error(
        "StableDiffusion35AdaLayerNormZeroX.modulate: emb shape must match [batch, hiddenSize].",
      );
    }
    using activated = silu(emb);
    using projected = this.linear.forward(activated);
    const parts = split(projected, 9, -1);
    try {
      const shiftMsa = reshapeModulation(
        partAt(parts, 0, "StableDiffusion35AdaLayerNormZeroX.modulate"),
        shape.batch,
        this.#hiddenSize,
      );
      const scaleMsa = reshapeModulation(
        partAt(parts, 1, "StableDiffusion35AdaLayerNormZeroX.modulate"),
        shape.batch,
        this.#hiddenSize,
      );
      const shiftMsa2 = reshapeModulation(
        partAt(parts, 6, "StableDiffusion35AdaLayerNormZeroX.modulate"),
        shape.batch,
        this.#hiddenSize,
      );
      const scaleMsa2 = reshapeModulation(
        partAt(parts, 7, "StableDiffusion35AdaLayerNormZeroX.modulate"),
        shape.batch,
        this.#hiddenSize,
      );
      try {
        using normalized = affineFreeLayerNorm(
          x,
          this.#hiddenSize,
          "StableDiffusion35AdaLayerNormZeroX.modulate",
        );
        return {
          hiddenStates: applyScaleShift(normalized, shiftMsa, scaleMsa),
          gateMsa: reshapeModulation(
            partAt(parts, 2, "StableDiffusion35AdaLayerNormZeroX.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
          shiftMlp: reshapeModulation(
            partAt(parts, 3, "StableDiffusion35AdaLayerNormZeroX.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
          scaleMlp: reshapeModulation(
            partAt(parts, 4, "StableDiffusion35AdaLayerNormZeroX.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
          gateMlp: reshapeModulation(
            partAt(parts, 5, "StableDiffusion35AdaLayerNormZeroX.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
          hiddenStates2: applyScaleShift(normalized, shiftMsa2, scaleMsa2),
          gateMsa2: reshapeModulation(
            partAt(parts, 8, "StableDiffusion35AdaLayerNormZeroX.modulate"),
            shape.batch,
            this.#hiddenSize,
          ),
        };
      } finally {
        shiftMsa.free();
        scaleMsa.free();
        shiftMsa2.free();
        scaleMsa2.free();
      }
    } finally {
      freeArrays(parts);
    }
  }
}

/** SD3 adaptive layer norm used before the final projection and final context pre-pass. */
export class StableDiffusion3AdaLayerNormContinuous extends Module {
  linear: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number, conditioningDims: number) {
    super();
    this.linear = new Linear(conditioningDims, hiddenSize * 2);
    this.#hiddenSize = hiddenSize;
  }

  forward(x: MxArray, conditioning: MxArray): MxArray {
    const shape = assertSequence3d(x, "StableDiffusion3AdaLayerNormContinuous.forward");
    if (shape.channels !== this.#hiddenSize) {
      throw new Error("StableDiffusion3AdaLayerNormContinuous.forward: hidden size mismatch.");
    }
    if (conditioning.shape.length !== 2 || conditioning.shape[0] !== shape.batch) {
      throw new Error(
        "StableDiffusion3AdaLayerNormContinuous.forward: conditioning must have shape [batch, channels].",
      );
    }
    using activated = silu(conditioning);
    using projected = this.linear.forward(activated);
    const parts = split(projected, 2, -1);
    try {
      using scale = reshapeModulation(
        partAt(parts, 0, "StableDiffusion3AdaLayerNormContinuous.forward"),
        shape.batch,
        this.#hiddenSize,
      );
      using shift = reshapeModulation(
        partAt(parts, 1, "StableDiffusion3AdaLayerNormContinuous.forward"),
        shape.batch,
        this.#hiddenSize,
      );
      using normalized = affineFreeLayerNorm(
        x,
        this.#hiddenSize,
        "StableDiffusion3AdaLayerNormContinuous.forward",
      );
      return applyScaleShift(normalized, shift, scale);
    } finally {
      freeArrays(parts);
    }
  }
}
