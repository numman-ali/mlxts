import type { MxArray } from "@mlxts/core";
import { concatenate, reshape, scaledDotProductAttention, transpose } from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import type { StableDiffusion3QkNorm } from "./config";
import { assertAttention4d, assertSequence3d, sliceAxis } from "./tensor-utils";

type AttentionProjection = {
  queries: MxArray;
  keys: MxArray;
  values: MxArray;
};

export type StableDiffusion3JointAttentionOutput = {
  hidden: MxArray;
  context: MxArray | null;
};

function qkNormModule(qkNorm: StableDiffusion3QkNorm, headDim: number): RMSNorm | null {
  if (qkNorm === null) {
    return null;
  }
  if (qkNorm === "rms_norm") {
    return new RMSNorm(headDim, 1e-6);
  }
  throw new Error(`StableDiffusion3JointAttention: unsupported qkNorm ${String(qkNorm)}.`);
}

function disposeProjection(projection: AttentionProjection): void {
  projection.queries.free();
  projection.keys.free();
  projection.values.free();
}

function projectSequence(options: {
  input: MxArray;
  linear: Linear;
  heads: number;
  headDim: number;
  owner: string;
}): MxArray {
  const shape = assertSequence3d(options.input, options.owner);
  using projected = options.linear.forward(options.input);
  using projectedHeads = reshape(projected, [
    shape.batch,
    shape.length,
    options.heads,
    options.headDim,
  ]);
  return transpose(projectedHeads, [0, 2, 1, 3]);
}

function normalizeProjection(projection: MxArray, norm: RMSNorm | null): MxArray {
  if (norm === null) {
    return projection;
  }
  try {
    return norm.forward(projection);
  } finally {
    projection.free();
  }
}

function attentionToSequence(
  attention: MxArray,
  batch: number,
  length: number,
  innerDim: number,
): MxArray {
  using sequenceFirst = transpose(attention, [0, 2, 1, 3]);
  return reshape(sequenceFirst, [batch, length, innerDim]);
}

/** SD3 joint image/context attention over `[image, context]` sequence order. */
export class StableDiffusion3JointAttention extends Module {
  toQ: Linear;
  toK: Linear;
  toV: Linear;
  addQProj: Linear;
  addKProj: Linear;
  addVProj: Linear;
  toOut: Linear;
  toAddOut: Linear | null;
  normQ: RMSNorm | null;
  normK: RMSNorm | null;
  normAddedQ: RMSNorm | null;
  normAddedK: RMSNorm | null;
  #heads: number;
  #headDim: number;
  #innerDim: number;
  #contextPreOnly: boolean;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    headDim: number;
    qkNorm: StableDiffusion3QkNorm;
    contextPreOnly: boolean;
  }) {
    super();
    if (options.hiddenSize !== options.numHeads * options.headDim) {
      throw new Error("StableDiffusion3JointAttention: hiddenSize must equal heads * headDim.");
    }
    this.toQ = new Linear(options.hiddenSize, options.hiddenSize);
    this.toK = new Linear(options.hiddenSize, options.hiddenSize);
    this.toV = new Linear(options.hiddenSize, options.hiddenSize);
    this.addQProj = new Linear(options.hiddenSize, options.hiddenSize);
    this.addKProj = new Linear(options.hiddenSize, options.hiddenSize);
    this.addVProj = new Linear(options.hiddenSize, options.hiddenSize);
    this.toOut = new Linear(options.hiddenSize, options.hiddenSize);
    this.toAddOut = options.contextPreOnly
      ? null
      : new Linear(options.hiddenSize, options.hiddenSize);
    this.normQ = qkNormModule(options.qkNorm, options.headDim);
    this.normK = qkNormModule(options.qkNorm, options.headDim);
    this.normAddedQ = qkNormModule(options.qkNorm, options.headDim);
    this.normAddedK = qkNormModule(options.qkNorm, options.headDim);
    this.#heads = options.numHeads;
    this.#headDim = options.headDim;
    this.#innerDim = options.hiddenSize;
    this.#contextPreOnly = options.contextPreOnly;
  }

  forward(_hiddenStates: MxArray): MxArray {
    throw new Error("StableDiffusion3JointAttention.forward: use run() inside an SD3 block.");
  }

  /** Run joint attention and split outputs back to image/context streams. */
  run(hiddenStates: MxArray, encoderHiddenStates: MxArray): StableDiffusion3JointAttentionOutput {
    const imageShape = assertSequence3d(
      hiddenStates,
      "StableDiffusion3JointAttention.run hiddenStates",
    );
    const contextShape = assertSequence3d(
      encoderHiddenStates,
      "StableDiffusion3JointAttention.run encoderHiddenStates",
    );
    if (contextShape.batch !== imageShape.batch || contextShape.channels !== imageShape.channels) {
      throw new Error(
        "StableDiffusion3JointAttention.run: context shape must match image batch and channels.",
      );
    }
    const imageProjection = this.#projectImage(hiddenStates);
    const contextProjection = this.#projectContext(encoderHiddenStates);
    try {
      using queries = concatenate([imageProjection.queries, contextProjection.queries], 2);
      using keys = concatenate([imageProjection.keys, contextProjection.keys], 2);
      using values = concatenate([imageProjection.values, contextProjection.values], 2);
      const attentionShape = assertAttention4d(
        queries,
        "StableDiffusion3JointAttention.run queries",
      );
      using attended = scaledDotProductAttention(queries, keys, values, {
        scale: this.#headDim ** -0.5,
      });
      using sequence = attentionToSequence(
        attended,
        imageShape.batch,
        attentionShape.length,
        this.#innerDim,
      );
      using hiddenSlice = sliceAxis(sequence, 1, 0, imageShape.length);
      using contextSlice = sliceAxis(sequence, 1, imageShape.length, attentionShape.length);
      const hidden = this.toOut.forward(hiddenSlice);
      if (this.#contextPreOnly) {
        return { hidden, context: null };
      }
      if (this.toAddOut === null) {
        hidden.free();
        throw new Error("StableDiffusion3JointAttention.run: missing context output projection.");
      }
      try {
        return {
          hidden,
          context: this.toAddOut.forward(contextSlice),
        };
      } catch (error) {
        hidden.free();
        throw error;
      }
    } finally {
      disposeProjection(imageProjection);
      disposeProjection(contextProjection);
    }
  }

  #projectImage(hiddenStates: MxArray): AttentionProjection {
    const queries = normalizeProjection(
      projectSequence({
        input: hiddenStates,
        linear: this.toQ,
        heads: this.#heads,
        headDim: this.#headDim,
        owner: "StableDiffusion3JointAttention.run image queries",
      }),
      this.normQ,
    );
    const keys = normalizeProjection(
      projectSequence({
        input: hiddenStates,
        linear: this.toK,
        heads: this.#heads,
        headDim: this.#headDim,
        owner: "StableDiffusion3JointAttention.run image keys",
      }),
      this.normK,
    );
    const values = projectSequence({
      input: hiddenStates,
      linear: this.toV,
      heads: this.#heads,
      headDim: this.#headDim,
      owner: "StableDiffusion3JointAttention.run image values",
    });
    return { queries, keys, values };
  }

  #projectContext(encoderHiddenStates: MxArray): AttentionProjection {
    const queries = normalizeProjection(
      projectSequence({
        input: encoderHiddenStates,
        linear: this.addQProj,
        heads: this.#heads,
        headDim: this.#headDim,
        owner: "StableDiffusion3JointAttention.run context queries",
      }),
      this.normAddedQ,
    );
    const keys = normalizeProjection(
      projectSequence({
        input: encoderHiddenStates,
        linear: this.addKProj,
        heads: this.#heads,
        headDim: this.#headDim,
        owner: "StableDiffusion3JointAttention.run context keys",
      }),
      this.normAddedK,
    );
    const values = projectSequence({
      input: encoderHiddenStates,
      linear: this.addVProj,
      heads: this.#heads,
      headDim: this.#headDim,
      owner: "StableDiffusion3JointAttention.run context values",
    });
    return { queries, keys, values };
  }
}

/** SD3.5 image-only self-attention used by dual-attention layers. */
export class StableDiffusion3SelfAttention extends Module {
  toQ: Linear;
  toK: Linear;
  toV: Linear;
  toOut: Linear;
  normQ: RMSNorm | null;
  normK: RMSNorm | null;
  #heads: number;
  #headDim: number;
  #innerDim: number;

  constructor(
    hiddenSize: number,
    numHeads: number,
    headDim: number,
    qkNorm: StableDiffusion3QkNorm,
  ) {
    super();
    if (hiddenSize !== numHeads * headDim) {
      throw new Error("StableDiffusion3SelfAttention: hiddenSize must equal heads * headDim.");
    }
    this.toQ = new Linear(hiddenSize, hiddenSize);
    this.toK = new Linear(hiddenSize, hiddenSize);
    this.toV = new Linear(hiddenSize, hiddenSize);
    this.toOut = new Linear(hiddenSize, hiddenSize);
    this.normQ = qkNormModule(qkNorm, headDim);
    this.normK = qkNormModule(qkNorm, headDim);
    this.#heads = numHeads;
    this.#headDim = headDim;
    this.#innerDim = hiddenSize;
  }

  forward(hiddenStates: MxArray): MxArray {
    const shape = assertSequence3d(hiddenStates, "StableDiffusion3SelfAttention.forward");
    const projection = this.#project(hiddenStates);
    try {
      using attended = scaledDotProductAttention(
        projection.queries,
        projection.keys,
        projection.values,
        { scale: this.#headDim ** -0.5 },
      );
      using sequence = attentionToSequence(attended, shape.batch, shape.length, this.#innerDim);
      return this.toOut.forward(sequence);
    } finally {
      disposeProjection(projection);
    }
  }

  #project(hiddenStates: MxArray): AttentionProjection {
    const queries = normalizeProjection(
      projectSequence({
        input: hiddenStates,
        linear: this.toQ,
        heads: this.#heads,
        headDim: this.#headDim,
        owner: "StableDiffusion3SelfAttention.forward queries",
      }),
      this.normQ,
    );
    const keys = normalizeProjection(
      projectSequence({
        input: hiddenStates,
        linear: this.toK,
        heads: this.#heads,
        headDim: this.#headDim,
        owner: "StableDiffusion3SelfAttention.forward keys",
      }),
      this.normK,
    );
    const values = projectSequence({
      input: hiddenStates,
      linear: this.toV,
      heads: this.#heads,
      headDim: this.#headDim,
      owner: "StableDiffusion3SelfAttention.forward values",
    });
    return { queries, keys, values };
  }
}
