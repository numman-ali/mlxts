import { add, fastRmsNorm, geluApprox, type MxArray, multiply, retainArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import { Ltx2Attention, type Ltx2AttentionOptions } from "./attention-ltx2";
import {
  disposeBlockModulation,
  disposeBlockOutput,
  disposeCrossModulation,
  type Ltx2BlockModulation,
  type Ltx2CrossModulation,
  modParams,
  partAt,
  scaledNormal,
} from "./blocks-ltx2-modulation";
import type { Ltx2VideoTransformerConfig } from "./config";
import type { LtxRotaryEmbeddings } from "./embeddings";
import { applyScaleShift, assertSequence3d, freeArrays, sliceAxis } from "./tensor-utils";

export type Ltx2VideoTransformerBlockInput = {
  hiddenStates: MxArray;
  audioHiddenStates: MxArray;
  encoderHiddenStates: MxArray;
  audioEncoderHiddenStates: MxArray;
  temb: MxArray;
  tembAudio: MxArray;
  tembCaScaleShift: MxArray;
  tembCaAudioScaleShift: MxArray;
  tembCaGate: MxArray;
  tembCaAudioGate: MxArray;
  videoRotaryEmbeddings: LtxRotaryEmbeddings;
  audioRotaryEmbeddings: LtxRotaryEmbeddings;
  caVideoRotaryEmbeddings: LtxRotaryEmbeddings;
  caAudioRotaryEmbeddings: LtxRotaryEmbeddings;
  encoderAttentionMask?: MxArray;
  audioEncoderAttentionMask?: MxArray;
  useA2vCrossAttention?: boolean;
  useV2aCrossAttention?: boolean;
};

export type Ltx2VideoTransformerBlockOutput = {
  hiddenStates: MxArray;
  audioHiddenStates: MxArray;
};

/** GELU feed-forward layer used by LTX-2 video and audio blocks. */
export class Ltx2FeedForward extends Module {
  linear1: Linear;
  linear2: Linear;

  constructor(hiddenSize: number) {
    super();
    this.linear1 = new Linear(hiddenSize, hiddenSize * 4);
    this.linear2 = new Linear(hiddenSize * 4, hiddenSize);
  }

  forward(x: MxArray): MxArray {
    using hidden = this.linear1.forward(x);
    using activated = geluApprox(hidden);
    return this.linear2.forward(activated);
  }
}

/** One LTX-2 audio-video transformer block. */
export class Ltx2VideoTransformerBlock extends Module {
  norm1: null;
  attn1: Ltx2Attention;
  audioNorm1: null;
  audioAttn1: Ltx2Attention;
  norm2: null;
  attn2: Ltx2Attention;
  audioNorm2: null;
  audioAttn2: Ltx2Attention;
  audioToVideoNorm: null;
  audioToVideoAttn: Ltx2Attention;
  videoToAudioNorm: null;
  videoToAudioAttn: Ltx2Attention;
  norm3: null;
  ff: Ltx2FeedForward;
  audioNorm3: null;
  audioFf: Ltx2FeedForward;
  scaleShiftTable: MxArray;
  audioScaleShiftTable: MxArray;
  videoA2vCrossAttnScaleShiftTable: MxArray;
  audioA2vCrossAttnScaleShiftTable: MxArray;
  #hiddenSize: number;
  #audioHiddenSize: number;
  #normEps: number;

  constructor(config: Ltx2VideoTransformerConfig) {
    super();
    this.norm1 = null;
    this.attn1 = new Ltx2Attention({
      queryDim: config.hiddenSize,
      heads: config.numAttentionHeads,
      headDim: config.attentionHeadDim,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
      ropeType: config.ropeType,
      normEps: config.normEps,
      gatedAttention: config.gatedAttn,
    });
    this.audioNorm1 = null;
    this.audioAttn1 = new Ltx2Attention({
      queryDim: config.audioHiddenSize,
      heads: config.audioNumAttentionHeads,
      headDim: config.audioAttentionHeadDim,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
      ropeType: config.ropeType,
      normEps: config.normEps,
      gatedAttention: config.audioGatedAttn,
    });
    this.norm2 = null;
    this.attn2 = new Ltx2Attention({
      queryDim: config.hiddenSize,
      heads: config.numAttentionHeads,
      headDim: config.attentionHeadDim,
      crossAttentionDim: config.crossAttentionDim,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
      ropeType: config.ropeType,
      normEps: config.normEps,
      gatedAttention: config.gatedAttn,
    });
    this.audioNorm2 = null;
    this.audioAttn2 = new Ltx2Attention({
      queryDim: config.audioHiddenSize,
      heads: config.audioNumAttentionHeads,
      headDim: config.audioAttentionHeadDim,
      crossAttentionDim: config.audioCrossAttentionDim,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
      ropeType: config.ropeType,
      normEps: config.normEps,
      gatedAttention: config.audioGatedAttn,
    });
    this.audioToVideoNorm = null;
    this.audioToVideoAttn = new Ltx2Attention({
      queryDim: config.hiddenSize,
      heads: config.audioNumAttentionHeads,
      headDim: config.audioAttentionHeadDim,
      crossAttentionDim: config.audioHiddenSize,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
      ropeType: config.ropeType,
      normEps: config.normEps,
      gatedAttention: config.gatedAttn,
    });
    this.videoToAudioNorm = null;
    this.videoToAudioAttn = new Ltx2Attention({
      queryDim: config.audioHiddenSize,
      heads: config.audioNumAttentionHeads,
      headDim: config.audioAttentionHeadDim,
      crossAttentionDim: config.hiddenSize,
      attentionBias: config.attentionBias,
      attentionOutBias: config.attentionOutBias,
      ropeType: config.ropeType,
      normEps: config.normEps,
      gatedAttention: config.audioGatedAttn,
    });
    this.norm3 = null;
    this.ff = new Ltx2FeedForward(config.hiddenSize);
    this.audioNorm3 = null;
    this.audioFf = new Ltx2FeedForward(config.audioHiddenSize);
    this.scaleShiftTable = scaledNormal([6, config.hiddenSize], config.hiddenSize ** -0.5);
    this.audioScaleShiftTable = scaledNormal(
      [6, config.audioHiddenSize],
      config.audioHiddenSize ** -0.5,
    );
    this.videoA2vCrossAttnScaleShiftTable = scaledNormal([5, config.hiddenSize], 1);
    this.audioA2vCrossAttnScaleShiftTable = scaledNormal([5, config.audioHiddenSize], 1);
    this.#hiddenSize = config.hiddenSize;
    this.#audioHiddenSize = config.audioHiddenSize;
    this.#normEps = config.normEps;
  }

  forward(): MxArray {
    throw new Error("Ltx2VideoTransformerBlock.forward: use run() inside the LTX-2 transformer.");
  }

  /** Run one LTX-2 video/audio block with text and cross-modality attention. */
  run(input: Ltx2VideoTransformerBlockInput): Ltx2VideoTransformerBlockOutput {
    const videoShape = assertSequence3d(input.hiddenStates, "Ltx2VideoTransformerBlock.run video");
    const audioShape = assertSequence3d(
      input.audioHiddenStates,
      "Ltx2VideoTransformerBlock.run audio",
    );
    if (videoShape.channels !== this.#hiddenSize || audioShape.channels !== this.#audioHiddenSize) {
      throw new Error("Ltx2VideoTransformerBlock.run: hidden size mismatch.");
    }
    const videoModulation = this.#blockModulation(
      this.scaleShiftTable,
      input.temb,
      videoShape.batch,
      this.#hiddenSize,
      "Ltx2VideoTransformerBlock.videoModulation",
    );
    const audioModulation = this.#blockModulation(
      this.audioScaleShiftTable,
      input.tembAudio,
      audioShape.batch,
      this.#audioHiddenSize,
      "Ltx2VideoTransformerBlock.audioModulation",
    );
    const videoCrossModulation = this.#crossModulation(
      this.videoA2vCrossAttnScaleShiftTable,
      input.tembCaScaleShift,
      input.tembCaGate,
      videoShape.batch,
      this.#hiddenSize,
      "Ltx2VideoTransformerBlock.videoCrossModulation",
    );
    const audioCrossModulation = this.#crossModulation(
      this.audioA2vCrossAttnScaleShiftTable,
      input.tembCaAudioScaleShift,
      input.tembCaAudioGate,
      audioShape.batch,
      this.#audioHiddenSize,
      "Ltx2VideoTransformerBlock.audioCrossModulation",
    );
    try {
      using videoSelf = this.#videoSelfAttention(
        input.hiddenStates,
        videoModulation,
        input.videoRotaryEmbeddings,
      );
      using audioSelf = this.#audioSelfAttention(
        input.audioHiddenStates,
        audioModulation,
        input.audioRotaryEmbeddings,
      );
      using videoText = this.#videoTextAttention(
        videoSelf,
        input.encoderHiddenStates,
        input.encoderAttentionMask,
      );
      using audioText = this.#audioTextAttention(
        audioSelf,
        input.audioEncoderHiddenStates,
        input.audioEncoderAttentionMask,
      );
      const crossed = this.#crossModal({
        video: videoText,
        audio: audioText,
        videoCrossModulation,
        audioCrossModulation,
        caVideoRotaryEmbeddings: input.caVideoRotaryEmbeddings,
        caAudioRotaryEmbeddings: input.caAudioRotaryEmbeddings,
        useA2vCrossAttention: input.useA2vCrossAttention ?? true,
        useV2aCrossAttention: input.useV2aCrossAttention ?? true,
      });
      try {
        using videoOutput = this.#videoFeedForward(crossed.hiddenStates, videoModulation);
        using audioOutput = this.#audioFeedForward(crossed.audioHiddenStates, audioModulation);
        return {
          hiddenStates: retainArray(videoOutput),
          audioHiddenStates: retainArray(audioOutput),
        };
      } finally {
        disposeBlockOutput(crossed);
      }
    } finally {
      disposeBlockModulation(videoModulation);
      disposeBlockModulation(audioModulation);
      disposeCrossModulation(videoCrossModulation);
      disposeCrossModulation(audioCrossModulation);
    }
  }

  #blockModulation(
    table: MxArray,
    temb: MxArray,
    batch: number,
    hiddenSize: number,
    owner: string,
  ): Ltx2BlockModulation {
    const parts = modParams(table, temb, batch, hiddenSize, 6, owner);
    try {
      return {
        shiftMsa: retainArray(partAt(parts, 0, owner)),
        scaleMsa: retainArray(partAt(parts, 1, owner)),
        gateMsa: retainArray(partAt(parts, 2, owner)),
        shiftMlp: retainArray(partAt(parts, 3, owner)),
        scaleMlp: retainArray(partAt(parts, 4, owner)),
        gateMlp: retainArray(partAt(parts, 5, owner)),
      };
    } finally {
      freeArrays(parts);
    }
  }

  #crossModulation(
    table: MxArray,
    scaleShiftTemb: MxArray,
    gateTemb: MxArray,
    batch: number,
    hiddenSize: number,
    owner: string,
  ): Ltx2CrossModulation {
    using scaleShiftTable = sliceAxis(table, 0, 0, 4);
    using gateTable = sliceAxis(table, 0, 4, 5);
    const scaleShiftParts = modParams(scaleShiftTable, scaleShiftTemb, batch, hiddenSize, 4, owner);
    const gateParts = modParams(gateTable, gateTemb, batch, hiddenSize, 1, owner);
    try {
      return {
        a2vScale: retainArray(partAt(scaleShiftParts, 0, owner)),
        a2vShift: retainArray(partAt(scaleShiftParts, 1, owner)),
        v2aScale: retainArray(partAt(scaleShiftParts, 2, owner)),
        v2aShift: retainArray(partAt(scaleShiftParts, 3, owner)),
        gate: retainArray(partAt(gateParts, 0, owner)),
      };
    } finally {
      freeArrays(scaleShiftParts);
      freeArrays(gateParts);
    }
  }

  #videoSelfAttention(
    hiddenStates: MxArray,
    modulation: Ltx2BlockModulation,
    rotaryEmbeddings: LtxRotaryEmbeddings,
  ): MxArray {
    using normalized = fastRmsNorm(hiddenStates, undefined, { eps: this.#normEps });
    using modulated = applyScaleShift(normalized, modulation.shiftMsa, modulation.scaleMsa);
    using attention = this.attn1.run(modulated, { queryRotaryEmbeddings: rotaryEmbeddings });
    using gated = multiply(attention, modulation.gateMsa);
    return add(hiddenStates, gated);
  }

  #audioSelfAttention(
    hiddenStates: MxArray,
    modulation: Ltx2BlockModulation,
    rotaryEmbeddings: LtxRotaryEmbeddings,
  ): MxArray {
    using normalized = fastRmsNorm(hiddenStates, undefined, { eps: this.#normEps });
    using modulated = applyScaleShift(normalized, modulation.shiftMsa, modulation.scaleMsa);
    using attention = this.audioAttn1.run(modulated, { queryRotaryEmbeddings: rotaryEmbeddings });
    using gated = multiply(attention, modulation.gateMsa);
    return add(hiddenStates, gated);
  }

  #videoTextAttention(
    hiddenStates: MxArray,
    encoderHiddenStates: MxArray,
    attentionMask?: MxArray,
  ): MxArray {
    using normalized = fastRmsNorm(hiddenStates, undefined, { eps: this.#normEps });
    const options: Ltx2AttentionOptions = { encoderHiddenStates };
    if (attentionMask !== undefined) {
      options.attentionMask = attentionMask;
    }
    using attention = this.attn2.run(normalized, options);
    return add(hiddenStates, attention);
  }

  #audioTextAttention(
    hiddenStates: MxArray,
    encoderHiddenStates: MxArray,
    attentionMask?: MxArray,
  ): MxArray {
    using normalized = fastRmsNorm(hiddenStates, undefined, { eps: this.#normEps });
    const options: Ltx2AttentionOptions = { encoderHiddenStates };
    if (attentionMask !== undefined) {
      options.attentionMask = attentionMask;
    }
    using attention = this.audioAttn2.run(normalized, options);
    return add(hiddenStates, attention);
  }

  #crossModal(input: {
    video: MxArray;
    audio: MxArray;
    videoCrossModulation: Ltx2CrossModulation;
    audioCrossModulation: Ltx2CrossModulation;
    caVideoRotaryEmbeddings: LtxRotaryEmbeddings;
    caAudioRotaryEmbeddings: LtxRotaryEmbeddings;
    useA2vCrossAttention: boolean;
    useV2aCrossAttention: boolean;
  }): Ltx2VideoTransformerBlockOutput {
    using normVideo = fastRmsNorm(input.video, undefined, { eps: this.#normEps });
    using normAudio = fastRmsNorm(input.audio, undefined, { eps: this.#normEps });
    let videoOutput: MxArray | null = retainArray(input.video);
    let audioOutput: MxArray | null = retainArray(input.audio);
    try {
      if (input.useA2vCrossAttention) {
        using videoQuery = applyScaleShift(
          normVideo,
          input.videoCrossModulation.a2vShift,
          input.videoCrossModulation.a2vScale,
        );
        using audioKeyValue = applyScaleShift(
          normAudio,
          input.audioCrossModulation.a2vShift,
          input.audioCrossModulation.a2vScale,
        );
        using attention = this.audioToVideoAttn.run(videoQuery, {
          encoderHiddenStates: audioKeyValue,
          queryRotaryEmbeddings: input.caVideoRotaryEmbeddings,
          keyRotaryEmbeddings: input.caAudioRotaryEmbeddings,
        });
        using gated = multiply(attention, input.videoCrossModulation.gate);
        const nextVideo = add(input.video, gated);
        videoOutput.free();
        videoOutput = nextVideo;
      }
      if (input.useV2aCrossAttention) {
        using videoKeyValue = applyScaleShift(
          normVideo,
          input.videoCrossModulation.v2aShift,
          input.videoCrossModulation.v2aScale,
        );
        using audioQuery = applyScaleShift(
          normAudio,
          input.audioCrossModulation.v2aShift,
          input.audioCrossModulation.v2aScale,
        );
        using attention = this.videoToAudioAttn.run(audioQuery, {
          encoderHiddenStates: videoKeyValue,
          queryRotaryEmbeddings: input.caAudioRotaryEmbeddings,
          keyRotaryEmbeddings: input.caVideoRotaryEmbeddings,
        });
        using gated = multiply(attention, input.audioCrossModulation.gate);
        const nextAudio = add(input.audio, gated);
        audioOutput.free();
        audioOutput = nextAudio;
      }
      return { hiddenStates: videoOutput, audioHiddenStates: audioOutput };
    } catch (error) {
      videoOutput?.free();
      audioOutput?.free();
      throw error;
    }
  }

  #videoFeedForward(hiddenStates: MxArray, modulation: Ltx2BlockModulation): MxArray {
    using normalized = fastRmsNorm(hiddenStates, undefined, { eps: this.#normEps });
    using modulated = applyScaleShift(normalized, modulation.shiftMlp, modulation.scaleMlp);
    using projected = this.ff.forward(modulated);
    using gated = multiply(projected, modulation.gateMlp);
    return add(hiddenStates, gated);
  }

  #audioFeedForward(hiddenStates: MxArray, modulation: Ltx2BlockModulation): MxArray {
    using normalized = fastRmsNorm(hiddenStates, undefined, { eps: this.#normEps });
    using modulated = applyScaleShift(normalized, modulation.shiftMlp, modulation.scaleMlp);
    using projected = this.audioFf.forward(modulated);
    using gated = multiply(projected, modulation.gateMlp);
    return add(hiddenStates, gated);
  }
}

/** Dispose tensors returned by `Ltx2VideoTransformerBlock.run`. */
export function disposeLtx2VideoTransformerBlockOutput(
  output: Ltx2VideoTransformerBlockOutput,
): void {
  disposeBlockOutput(output);
}
