import type { MxArray } from "@mlxts/core";
import {
  add,
  divide,
  equal,
  expandDims,
  geluApprox,
  maximum,
  mean,
  minimum,
  multiply,
  reshape,
  retainArray,
  split,
  sqrt,
  square,
  subtract,
  where,
} from "@mlxts/core";
import { Linear, Module, silu } from "@mlxts/nn";

import { QwenImageJointAttention } from "./attention";
import { assertSequence3d, freeArrays, sliceAxis } from "./tensor-utils";

type QwenImageModulationTriplet = {
  shift: MxArray;
  scale: MxArray;
  gate: MxArray;
};

type QwenImageBlockModulation = {
  attention: QwenImageModulationTriplet;
  feedForward: QwenImageModulationTriplet;
};

export type QwenImageBlockOutput = {
  image: MxArray;
  text: MxArray;
};

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

function disposeTriplet(triplet: QwenImageModulationTriplet): void {
  triplet.shift.free();
  triplet.scale.free();
  triplet.gate.free();
}

function disposeModulation(modulation: QwenImageBlockModulation): void {
  disposeTriplet(modulation.attention);
  disposeTriplet(modulation.feedForward);
}

function affineFreeLayerNorm(x: MxArray, dims: number, owner: string): MxArray {
  const shape = assertSequence3d(x, owner);
  if (shape.channels !== dims) {
    throw new Error(`${owner}: expected last dimension ${dims}, got ${shape.channels}.`);
  }
  using center = mean(x, -1, true);
  using centered = subtract(x, center);
  using squared = square(centered);
  using variance = mean(squared, -1, true);
  using stabilized = add(variance, 1e-6);
  using denominator = sqrt(stabilized);
  return divide(centered, denominator);
}

function applyModulation(x: MxArray, modulation: QwenImageModulationTriplet): MxArray {
  using scaled = add(modulation.scale, 1);
  using multiplied = multiply(x, scaled);
  return add(multiplied, modulation.shift);
}

function reshapeModulation(part: MxArray, batch: number, hiddenSize: number): MxArray {
  return reshape(part, [batch, 1, hiddenSize]);
}

function assertModulationIndex(
  index: MxArray,
  batch: number,
  owner: string,
): readonly [number, number] {
  const [indexBatch, indexLength] = index.shape;
  if (index.shape.length !== 2 || indexBatch === undefined || indexLength === undefined) {
    throw new Error(`${owner}: modulateIndex must have shape [batch, imageLength].`);
  }
  if (batch !== indexBatch * 2) {
    throw new Error(`${owner}: zero_cond_t modulation requires two timestep embeddings per batch.`);
  }
  return [indexBatch, indexLength];
}

function indexedModulationPart(
  part: MxArray,
  index: MxArray,
  batch: number,
  hiddenSize: number,
  owner: string,
): MxArray {
  const [indexBatch] = assertModulationIndex(index, batch, owner);
  using targetPart = sliceAxis(part, 0, 0, indexBatch);
  using referencePart = sliceAxis(part, 0, indexBatch, batch);
  using target = reshapeModulation(targetPart, indexBatch, hiddenSize);
  using reference = reshapeModulation(referencePart, indexBatch, hiddenSize);
  using targetSelector = equal(index, 0);
  using selector = expandDims(targetSelector, 2);
  return where(selector, target, reference);
}

function modulationPart(
  part: MxArray,
  batch: number,
  hiddenSize: number,
  owner: string,
  index?: MxArray,
): MxArray {
  if (index === undefined) {
    return reshapeModulation(part, batch, hiddenSize);
  }
  return indexedModulationPart(part, index, batch, hiddenSize, owner);
}

function clipFloat16(x: MxArray): MxArray {
  if (x.dtype !== "float16") {
    return retainArray(x);
  }
  using upper = minimum(x, 65504);
  return maximum(upper, -65504);
}

function validateImageModulateIndex(index: MxArray | undefined, image: MxArray): void {
  if (index === undefined) {
    return;
  }
  const { batch, length } = assertSequence3d(image, "QwenImageTransformerBlock.run image");
  if (index.shape.length !== 2 || index.shape[0] !== batch || index.shape[1] !== length) {
    throw new Error("QwenImageTransformerBlock.run: modulateIndex must match image tokens.");
  }
}

/** Gated modulation projection used by Qwen-Image dual-stream blocks. */
export class QwenImageModulation extends Module {
  linear: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number) {
    super();
    this.linear = new Linear(hiddenSize, hiddenSize * 6);
    this.#hiddenSize = hiddenSize;
  }

  forward(vec: MxArray): MxArray {
    const [batch, channels] = vec.shape;
    if (vec.shape.length !== 2 || batch === undefined || channels !== this.#hiddenSize) {
      throw new Error("QwenImageModulation.forward: vec must have shape [batch, hiddenSize].");
    }
    using activated = silu(vec);
    return this.linear.forward(activated);
  }

  modulate(vec: MxArray, index?: MxArray): QwenImageBlockModulation {
    const [batch] = vec.shape;
    if (batch === undefined) {
      throw new Error("QwenImageModulation.modulate: vec must include a batch dimension.");
    }
    using projected = this.forward(vec);
    const parts = split(projected, 6, -1);
    try {
      return {
        attention: {
          shift: modulationPart(
            partAt(parts, 0, "QwenImageModulation.modulate"),
            batch,
            this.#hiddenSize,
            "QwenImageModulation.modulate attention shift",
            index,
          ),
          scale: modulationPart(
            partAt(parts, 1, "QwenImageModulation.modulate"),
            batch,
            this.#hiddenSize,
            "QwenImageModulation.modulate attention scale",
            index,
          ),
          gate: modulationPart(
            partAt(parts, 2, "QwenImageModulation.modulate"),
            batch,
            this.#hiddenSize,
            "QwenImageModulation.modulate attention gate",
            index,
          ),
        },
        feedForward: {
          shift: modulationPart(
            partAt(parts, 3, "QwenImageModulation.modulate"),
            batch,
            this.#hiddenSize,
            "QwenImageModulation.modulate feedForward shift",
            index,
          ),
          scale: modulationPart(
            partAt(parts, 4, "QwenImageModulation.modulate"),
            batch,
            this.#hiddenSize,
            "QwenImageModulation.modulate feedForward scale",
            index,
          ),
          gate: modulationPart(
            partAt(parts, 5, "QwenImageModulation.modulate"),
            batch,
            this.#hiddenSize,
            "QwenImageModulation.modulate feedForward gate",
            index,
          ),
        },
      };
    } finally {
      freeArrays(parts);
    }
  }
}

/** GELU feed-forward layer used by Qwen-Image blocks. */
export class QwenImageFeedForward extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(hiddenSize: number) {
    super();
    this.linear1 = new Linear(hiddenSize, hiddenSize * 4);
    this.linear2 = new Linear(hiddenSize * 4, hiddenSize);
  }

  /** Run Diffusers GELU-approximate feed-forward projection. */
  forward(x: MxArray): MxArray {
    using hidden = this.linear1.forward(x);
    using activated = geluApprox(hidden);
    return this.linear2.forward(activated);
  }
}

/** Final AdaLayerNormContinuous projection used by Qwen-Image. */
export class QwenImageAdaptiveLayerNormContinuous extends Module {
  linear: Linear;
  #hiddenSize: number;

  constructor(hiddenSize: number) {
    super();
    this.linear = new Linear(hiddenSize, hiddenSize * 2);
    this.#hiddenSize = hiddenSize;
  }

  /** Apply conditioning-scale and shift after affine-free layer normalization. */
  forward(hidden: MxArray, conditioning: MxArray): MxArray {
    using activated = silu(conditioning);
    using projected = this.linear.forward(activated);
    const parts = split(projected, 2, -1);
    try {
      const scale = reshapeModulation(
        partAt(parts, 0, "QwenImageAdaptiveLayerNormContinuous.forward"),
        conditioning.shape[0] ?? 0,
        this.#hiddenSize,
      );
      const shift = reshapeModulation(
        partAt(parts, 1, "QwenImageAdaptiveLayerNormContinuous.forward"),
        conditioning.shape[0] ?? 0,
        this.#hiddenSize,
      );
      try {
        using normalized = affineFreeLayerNorm(
          hidden,
          this.#hiddenSize,
          "QwenImageAdaptiveLayerNormContinuous.forward",
        );
        using scaled = add(scale, 1);
        using modulated = multiply(normalized, scaled);
        return add(modulated, shift);
      } finally {
        scale.free();
        shift.free();
      }
    } finally {
      freeArrays(parts);
    }
  }
}

/** One Qwen-Image dual-stream transformer block. */
export class QwenImageTransformerBlock extends Module {
  imgMod: QwenImageModulation;
  imgNorm1: null;
  attn: QwenImageJointAttention;
  imgNorm2: null;
  imgMlp: QwenImageFeedForward;
  txtMod: QwenImageModulation;
  txtNorm1: null;
  txtNorm2: null;
  txtMlp: QwenImageFeedForward;
  #hiddenSize: number;

  constructor(options: { hiddenSize: number; numHeads: number; headDim: number }) {
    super();
    this.imgMod = new QwenImageModulation(options.hiddenSize);
    this.imgNorm1 = null;
    this.attn = new QwenImageJointAttention(options.hiddenSize, options.numHeads, options.headDim);
    this.imgNorm2 = null;
    this.imgMlp = new QwenImageFeedForward(options.hiddenSize);
    this.txtMod = new QwenImageModulation(options.hiddenSize);
    this.txtNorm1 = null;
    this.txtNorm2 = null;
    this.txtMlp = new QwenImageFeedForward(options.hiddenSize);
    this.#hiddenSize = options.hiddenSize;
  }

  forward(_image: MxArray): MxArray {
    throw new Error("QwenImageTransformerBlock.forward: use run() inside the Qwen-Image model.");
  }

  /** Run one Qwen-Image block over image and text streams. */
  run(
    image: MxArray,
    text: MxArray,
    imageVector: MxArray,
    textVector: MxArray,
    rope: MxArray,
    textMask?: MxArray,
    modulateIndex?: MxArray,
  ): QwenImageBlockOutput {
    validateImageModulateIndex(modulateIndex, image);
    const imageModulation = this.imgMod.modulate(imageVector, modulateIndex);
    const textModulation = this.txtMod.modulate(textVector);
    try {
      using imageNormed = affineFreeLayerNorm(
        image,
        this.#hiddenSize,
        "QwenImageTransformerBlock.run image",
      );
      using textNormed = affineFreeLayerNorm(
        text,
        this.#hiddenSize,
        "QwenImageTransformerBlock.run text",
      );
      using imageAttentionInput = applyModulation(imageNormed, imageModulation.attention);
      using textAttentionInput = applyModulation(textNormed, textModulation.attention);
      const attention = this.attn.run(imageAttentionInput, textAttentionInput, rope, textMask);
      try {
        using gatedImageAttention = multiply(imageModulation.attention.gate, attention.image);
        using gatedTextAttention = multiply(textModulation.attention.gate, attention.text);
        using imageAttentionResidual = add(image, gatedImageAttention);
        using textAttentionResidual = add(text, gatedTextAttention);
        using imageOut = this.#feedForwardResidual(
          imageAttentionResidual,
          imageModulation.feedForward,
          this.imgMlp,
          "QwenImageTransformerBlock.run image mlp",
        );
        using textOut = this.#feedForwardResidual(
          textAttentionResidual,
          textModulation.feedForward,
          this.txtMlp,
          "QwenImageTransformerBlock.run text mlp",
        );
        using clippedImage = clipFloat16(imageOut);
        using clippedText = clipFloat16(textOut);
        return { image: retainArray(clippedImage), text: retainArray(clippedText) };
      } finally {
        attention.image.free();
        attention.text.free();
      }
    } finally {
      disposeModulation(imageModulation);
      disposeModulation(textModulation);
    }
  }

  #feedForwardResidual(
    hidden: MxArray,
    modulation: QwenImageModulationTriplet,
    feedForward: QwenImageFeedForward,
    owner: string,
  ): MxArray {
    using normalized = affineFreeLayerNorm(hidden, this.#hiddenSize, owner);
    using modulated = applyModulation(normalized, modulation);
    using output = feedForward.forward(modulated);
    using gated = multiply(modulation.gate, output);
    return add(hidden, gated);
  }

  get hiddenSize(): number {
    return this.#hiddenSize;
  }
}
