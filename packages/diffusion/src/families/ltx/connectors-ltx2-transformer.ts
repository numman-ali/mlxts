import {
  add,
  broadcastTo,
  expandDims,
  fastRmsNorm,
  formatShape,
  geluApprox,
  greater,
  MxArray,
  type MxArray as MxArrayType,
  ones,
  random,
  retainArray,
  takeAlongAxis,
  takeAxis,
  where,
} from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import type { Ltx2RopeType } from "./config";
import { Ltx2ConnectorAttention } from "./connectors-ltx2-attention";
import { ltx2ConnectorBinaryMask } from "./connectors-ltx2-normalization";
import { createLtx2ConnectorRotaryEmbeddings } from "./connectors-ltx2-rotary";
import type { LtxRotaryEmbeddings } from "./embeddings";
import { assertSequence3d, checkedModule } from "./tensor-utils";

function randomRegisters(count: number, hiddenSize: number): MxArrayType {
  return random.uniform(-1, 1, [count, hiddenSize]);
}

function connectorAttentionMask(mask: MxArrayType, batch: number, length: number): MxArrayType {
  if (mask.shape.length !== 2 || mask.shape[0] !== batch || mask.shape[1] !== length) {
    throw new Error(
      `LTX2ConnectorTransformer1d: attention mask must have shape [${batch}, ${length}], got ${formatShape(
        mask.shape,
      )}.`,
    );
  }
  using boolMask = mask.dtype === "bool" ? retainArray(mask) : greater(mask, 0);
  using keyMask = expandDims(boolMask, 1);
  return expandDims(keyMask, 1);
}

function assertRegisterReplacementShape(
  hiddenStates: MxArrayType,
  attentionMask: MxArrayType,
  learnableRegisters: MxArrayType,
): { batch: number; length: number; registerCount: number } {
  const shape = assertSequence3d(
    hiddenStates,
    "replaceLtx2ConnectorPaddingWithRegisters hiddenStates",
  );
  if (
    attentionMask.shape.length !== 2 ||
    attentionMask.shape[0] !== shape.batch ||
    attentionMask.shape[1] !== shape.length
  ) {
    throw new Error("replaceLtx2ConnectorPaddingWithRegisters: attention mask shape mismatch.");
  }
  const [registerCount, registerChannels] = learnableRegisters.shape;
  if (
    registerCount === undefined ||
    registerChannels !== shape.channels ||
    shape.length % registerCount !== 0
  ) {
    throw new Error(
      "replaceLtx2ConnectorPaddingWithRegisters: sequence length must be divisible by register count.",
    );
  }
  return { batch: shape.batch, length: shape.length, registerCount };
}

function validConnectorTokenIndices(
  maskValues: ArrayLike<number>,
  batch: number,
  length: number,
): number[] {
  const validIndices: number[] = [];
  for (let token = 0; token < length; token += 1) {
    if (Number(maskValues[batch * length + token] ?? 0) > 0) {
      validIndices.push(token);
    }
  }
  return validIndices;
}

function fillConnectorRegisterPlan(
  plan: { gatherIndices: Int32Array; registerIndices: Int32Array; registerMask: number[] },
  validIndices: readonly number[],
  batch: number,
  length: number,
  registerCount: number,
): void {
  for (let token = 0; token < length; token += 1) {
    const sourceToken = validIndices[token];
    const offset = batch * length + token;
    if (sourceToken !== undefined) {
      plan.gatherIndices[offset] = sourceToken;
    } else {
      plan.registerMask[offset] = 1;
      plan.gatherIndices[offset] = 0;
      plan.registerIndices[offset] = (token - validIndices.length) % registerCount;
    }
  }
}

function connectorRegisterPlan(
  attentionMask: MxArrayType,
  batch: number,
  length: number,
  registerCount: number,
): { gatherIndices: Int32Array; registerIndices: Int32Array; registerMask: number[] } {
  const maskValues = attentionMask.toTypedArray();
  const plan = {
    gatherIndices: new Int32Array(batch * length),
    registerIndices: new Int32Array(batch * length),
    registerMask: new Array<number>(batch * length).fill(0),
  };
  for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
    fillConnectorRegisterPlan(
      plan,
      validConnectorTokenIndices(maskValues, batchIndex, length),
      batchIndex,
      length,
      registerCount,
    );
  }
  return plan;
}

/** Feed-forward sublayer used by LTX-2 connector blocks. */
export class Ltx2ConnectorFeedForward extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(hiddenSize: number) {
    super();
    this.linear1 = new Linear(hiddenSize, hiddenSize * 4);
    this.linear2 = new Linear(hiddenSize * 4, hiddenSize);
  }

  forward(x: MxArrayType): MxArrayType {
    using hidden = this.linear1.forward(x);
    using activated = geluApprox(hidden);
    return this.linear2.forward(activated);
  }
}

/** Transformer block used by the LTX-2 text connector. */
export class Ltx2ConnectorBlock1d extends Module {
  attn1: Ltx2ConnectorAttention;
  ff: Ltx2ConnectorFeedForward;
  #hiddenSize: number;

  constructor(options: {
    hiddenSize: number;
    heads: number;
    headDim: number;
    gatedAttention: boolean;
    ropeType: Ltx2RopeType;
  }) {
    super();
    this.attn1 = new Ltx2ConnectorAttention({
      hiddenSize: options.hiddenSize,
      heads: options.heads,
      headDim: options.headDim,
      gatedAttention: options.gatedAttention,
      ropeType: options.ropeType,
    });
    this.ff = new Ltx2ConnectorFeedForward(options.hiddenSize);
    this.#hiddenSize = options.hiddenSize;
  }

  forward(hiddenStates: MxArrayType): MxArrayType {
    return this.run(hiddenStates);
  }

  /** Run connector self-attention and feed-forward residuals. */
  run(hiddenStates: MxArrayType, mask?: MxArrayType, rotary?: LtxRotaryEmbeddings): MxArrayType {
    const shape = assertSequence3d(hiddenStates, "Ltx2ConnectorBlock1d.run hiddenStates");
    if (shape.channels !== this.#hiddenSize) {
      throw new Error("Ltx2ConnectorBlock1d.run: hidden size mismatch.");
    }
    const attentionOptions: { mask?: MxArrayType; rotary?: LtxRotaryEmbeddings } = {};
    if (mask !== undefined) {
      attentionOptions.mask = mask;
    }
    if (rotary !== undefined) {
      attentionOptions.rotary = rotary;
    }
    using normAttention = fastRmsNorm(hiddenStates, undefined, { eps: 1e-6 });
    using attention = this.attn1.run(normAttention, attentionOptions);
    using attentionResidual = add(hiddenStates, attention);
    using normFeedForward = fastRmsNorm(attentionResidual, undefined, { eps: 1e-6 });
    using feedForward = this.ff.forward(normFeedForward);
    return add(attentionResidual, feedForward);
  }
}

/** Replace connector padding tokens with learned register tokens. */
export function replaceLtx2ConnectorPaddingWithRegisters(
  hiddenStates: MxArrayType,
  attentionMask: MxArrayType,
  learnableRegisters: MxArrayType,
): MxArrayType {
  const shape = assertRegisterReplacementShape(hiddenStates, attentionMask, learnableRegisters);
  const plan = connectorRegisterPlan(attentionMask, shape.batch, shape.length, shape.registerCount);
  using sequenceIndices = MxArray.fromData(
    plan.gatherIndices,
    [shape.batch, shape.length, 1],
    "int32",
  );
  using broadcastIndices = broadcastTo(sequenceIndices, [...hiddenStates.shape]);
  using compacted = takeAlongAxis(hiddenStates, broadcastIndices, 1);
  using registerIndexTensor = MxArray.fromData(
    plan.registerIndices,
    [shape.batch, shape.length],
    "int32",
  );
  using registerValues = takeAxis(learnableRegisters, registerIndexTensor, 0);
  using mask = MxArray.fromData(plan.registerMask, [shape.batch, shape.length, 1], "bool");
  return where(mask, registerValues, compacted);
}

export type Ltx2ConnectorTransformerOutput = {
  hiddenStates: MxArrayType;
  attentionMask: MxArrayType;
};

/** LTX-2 modality-specific 1D text connector transformer. */
export class Ltx2ConnectorTransformer1d extends Module {
  learnableRegisters: MxArrayType | null;
  transformerBlocks: Ltx2ConnectorBlock1d[];
  #hiddenSize: number;
  #heads: number;
  #ropeTheta: number;
  #ropeBaseSeqLen: number;
  #ropeType: Ltx2RopeType;

  constructor(options: {
    heads: number;
    headDim: number;
    numLayers: number;
    numLearnableRegisters: number | null;
    ropeBaseSeqLen: number;
    ropeTheta: number;
    ropeType: Ltx2RopeType;
    gatedAttention: boolean;
  }) {
    super();
    this.#heads = options.heads;
    this.#hiddenSize = options.heads * options.headDim;
    this.learnableRegisters =
      options.numLearnableRegisters === null
        ? null
        : randomRegisters(options.numLearnableRegisters, this.#hiddenSize);
    this.transformerBlocks = Array.from(
      { length: options.numLayers },
      () =>
        new Ltx2ConnectorBlock1d({
          hiddenSize: this.#hiddenSize,
          heads: options.heads,
          headDim: options.headDim,
          gatedAttention: options.gatedAttention,
          ropeType: options.ropeType,
        }),
    );
    this.#ropeTheta = options.ropeTheta;
    this.#ropeBaseSeqLen = options.ropeBaseSeqLen;
    this.#ropeType = options.ropeType;
  }

  forward(): MxArrayType {
    throw new Error("Ltx2ConnectorTransformer1d.forward: use run() with an attention mask.");
  }

  /** Run connector blocks and return prompt embeddings plus their mask. */
  run(hiddenStates: MxArrayType, attentionMask: MxArrayType): Ltx2ConnectorTransformerOutput {
    const shape = assertSequence3d(hiddenStates, "Ltx2ConnectorTransformer1d.run hiddenStates");
    if (shape.channels !== this.#hiddenSize) {
      throw new Error("Ltx2ConnectorTransformer1d.run: hidden size mismatch.");
    }
    const registeredHidden =
      this.learnableRegisters === null
        ? retainArray(hiddenStates)
        : replaceLtx2ConnectorPaddingWithRegisters(
            hiddenStates,
            attentionMask,
            this.learnableRegisters,
          );
    try {
      let outputMask: MxArrayType | null = null;
      let mask: MxArrayType | null = null;
      let rotary: LtxRotaryEmbeddings | null = null;
      try {
        outputMask = ones([shape.batch, shape.length], "int32");
        if (this.learnableRegisters === null) {
          const inputMask = ltx2ConnectorBinaryMask(attentionMask, shape.batch, shape.length);
          try {
            mask = connectorAttentionMask(inputMask, shape.batch, shape.length);
          } finally {
            inputMask.free();
          }
        }
        rotary = createLtx2ConnectorRotaryEmbeddings({
          batch: shape.batch,
          length: shape.length,
          dim: this.#hiddenSize,
          theta: this.#ropeTheta,
          baseSequenceLength: this.#ropeBaseSeqLen,
          ropeType: this.#ropeType,
          heads: this.#heads,
        });
        let hidden = retainArray(registeredHidden);
        try {
          for (let index = 0; index < this.transformerBlocks.length; index += 1) {
            const block = checkedModule(
              this.transformerBlocks,
              index,
              "Ltx2ConnectorTransformer1d.run transformerBlocks",
            );
            const next = block.run(hidden, mask === null ? undefined : mask, rotary);
            hidden.free();
            hidden = next;
          }
          using normalized = fastRmsNorm(hidden, undefined, { eps: 1e-6 });
          return { hiddenStates: retainArray(normalized), attentionMask: outputMask };
        } finally {
          hidden.free();
        }
      } catch (error) {
        outputMask?.free();
        throw error;
      } finally {
        mask?.free();
        rotary?.cos.free();
        rotary?.sin.free();
      }
    } finally {
      registeredHidden.free();
    }
  }
}

/** Dispose tensors returned by `Ltx2ConnectorTransformer1d.run`. */
export function disposeLtx2ConnectorTransformerOutput(
  output: Ltx2ConnectorTransformerOutput,
): void {
  output.hiddenStates.free();
  output.attentionMask.free();
}
