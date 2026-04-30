/**
 * Stable Diffusion UNet timestep embedding modules.
 * @module
 */

import type { DType, MxArray } from "@mlxts/core";
import {
  array,
  asType,
  broadcastTo,
  concatenate,
  cos,
  multiply,
  reshape,
  retainArray,
  sin,
  zeros,
} from "@mlxts/core";
import { Linear, Module, silu } from "@mlxts/nn";

function normalizeTimesteps(timesteps: number | MxArray, batch: number, dtype: DType): MxArray {
  if (typeof timesteps === "number") {
    using value = array([timesteps], "float32");
    return batch === 1 ? retainArray(value) : broadcastTo(value, [batch]);
  }
  const rankOne = timesteps.shape.length === 0 ? reshape(timesteps, [1]) : retainArray(timesteps);
  try {
    const length = rankOne.shape[0];
    let batched: MxArray;
    if (length === batch) {
      batched = retainArray(rankOne);
    } else if (length === 1) {
      batched = broadcastTo(rankOne, [batch]);
    } else {
      throw new Error(
        `StableDiffusionSinusoidalTimesteps.forward: timestep length ${length ?? "undefined"} does not match batch ${batch}.`,
      );
    }
    if (batched.dtype === dtype) {
      return batched;
    }
    using ownedBatched = batched;
    return asType(ownedBatched, dtype);
  } finally {
    rankOne.free();
  }
}

/** Timestep MLP used by Stable Diffusion UNet conditioning. */
export class StableDiffusionTimestepEmbedding extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(inputDims: number, outputDims: number) {
    super();
    this.linear1 = new Linear(inputDims, outputDims);
    this.linear2 = new Linear(outputDims, outputDims);
  }

  /** Project sinusoidal timestep features into the UNet time embedding space. */
  forward(x: MxArray): MxArray {
    using first = this.linear1.forward(x);
    using activated = silu(first);
    return this.linear2.forward(activated);
  }
}

/** Diffusers-compatible sinusoidal timestep projection. */
export class StableDiffusionSinusoidalTimesteps implements Disposable {
  #invFrequency: MxArray;
  #dims: number;
  #flipSinToCos: boolean;
  #freqShift: number;
  #scale: number;

  constructor(dims: number, flipSinToCos: boolean, freqShift: number, scale = 1) {
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new Error(`StableDiffusionSinusoidalTimesteps: dims must be positive, got ${dims}.`);
    }
    this.#dims = dims;
    this.#flipSinToCos = flipSinToCos;
    this.#freqShift = freqShift;
    this.#scale = scale;
    this.#invFrequency = this.#buildInvFrequency();
  }

  /** Project scalar timesteps into sinusoidal embedding rows. */
  forward(timesteps: number | MxArray, batch: number, dtype: DType): MxArray {
    using normalizedTimesteps = normalizeTimesteps(timesteps, batch, "float32");
    using timestepColumn = reshape(normalizedTimesteps, [batch, 1]);
    using frequencyRow = reshape(this.#invFrequency, [1, this.#invFrequency.shape[0] ?? 0]);
    using phases = multiply(timestepColumn, frequencyRow);
    using scaledPhases = this.#scale === 1 ? retainArray(phases) : multiply(phases, this.#scale);
    using sine = sin(scaledPhases);
    using cosine = cos(scaledPhases);
    const ordered = this.#flipSinToCos
      ? concatenate([cosine, sine], -1)
      : concatenate([sine, cosine], -1);
    try {
      if (this.#dims % 2 === 0) {
        return dtype === "float32" ? retainArray(ordered) : asType(ordered, dtype);
      }
      using padding = zeros([batch, 1], ordered.dtype);
      using padded = concatenate([ordered, padding], -1);
      return dtype === padded.dtype ? retainArray(padded) : asType(padded, dtype);
    } finally {
      ordered.free();
    }
  }

  /** Return the raw inverse-frequency vector for numeric parity tests. */
  frequencies(): MxArray {
    return retainArray(this.#invFrequency);
  }

  #buildInvFrequency(): MxArray {
    const halfDims = Math.floor(this.#dims / 2);
    const denominator = halfDims - this.#freqShift;
    if (halfDims <= 0 || denominator <= 0) {
      throw new Error(
        "StableDiffusionSinusoidalTimesteps: dims and freqShift produce no frequencies.",
      );
    }
    const values = Array.from({ length: halfDims }, (_, index) =>
      Math.exp((-Math.log(10_000) * index) / denominator),
    );
    return array(values, "float32");
  }

  [Symbol.dispose](): void {
    this.#invFrequency.free();
  }
}
