import type { DType, MxArray } from "@mlxts/core";
import { formatShape, reshape, retainArray } from "@mlxts/core";
import { Linear, Module, silu } from "@mlxts/nn";

import { LtxVideoCombinedTimestepSizeEmbeddings } from "./conditioning";

export type Ltx2AdaLayerNormSingleOutput = {
  modulation: MxArray;
  embeddedTimestep: MxArray;
};

/** Configurable LTX-2 AdaLayerNormSingle timestep embedder. */
export class Ltx2AdaLayerNormSingle extends Module {
  emb: LtxVideoCombinedTimestepSizeEmbeddings;
  linear: Linear;
  #hiddenSize: number;
  #numModParams: number;

  constructor(hiddenSize: number, numModParams: number) {
    super();
    if (!Number.isInteger(numModParams) || numModParams <= 0) {
      throw new Error("Ltx2AdaLayerNormSingle: numModParams must be a positive integer.");
    }
    this.emb = new LtxVideoCombinedTimestepSizeEmbeddings(hiddenSize);
    this.linear = new Linear(hiddenSize, hiddenSize * numModParams);
    this.#hiddenSize = hiddenSize;
    this.#numModParams = numModParams;
  }

  /** Return flattened modulation parameters and final timestep embeddings. */
  embed(timesteps: MxArray, dtype: DType): Ltx2AdaLayerNormSingleOutput {
    const [batch] = timesteps.shape;
    if (timesteps.shape.length !== 1 || batch === undefined) {
      throw new Error(
        `Ltx2AdaLayerNormSingle.embed: timestep must be rank 1, got ${formatShape(
          timesteps.shape,
        )}.`,
      );
    }
    using embeddedTimestepFlat = this.emb.embed(timesteps, dtype);
    using activated = silu(embeddedTimestepFlat);
    using modulationFlat = this.linear.forward(activated);
    using modulation = reshape(modulationFlat, [batch, 1, this.#hiddenSize * this.#numModParams]);
    using embeddedTimestep = reshape(embeddedTimestepFlat, [batch, 1, this.#hiddenSize]);
    return {
      modulation: retainArray(modulation),
      embeddedTimestep: retainArray(embeddedTimestep),
    };
  }

  forward(timesteps: MxArray): MxArray {
    const output = this.embed(timesteps, timesteps.dtype);
    try {
      return retainArray(output.modulation);
    } finally {
      disposeLtx2AdaLayerNormSingleOutput(output);
    }
  }
}

/** Dispose tensors returned by `Ltx2AdaLayerNormSingle.embed`. */
export function disposeLtx2AdaLayerNormSingleOutput(output: Ltx2AdaLayerNormSingleOutput): void {
  output.modulation.free();
  output.embeddedTimestep.free();
}
