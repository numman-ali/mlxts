import { type MxArray, multiply, retainArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import type { Ltx2TextConnectorsConfig } from "./config";
import { ltx2PerLayerMaskedMeanNorm, ltx2PerTokenRmsNorm } from "./connectors-ltx2-normalization";
import {
  disposeLtx2ConnectorTransformerOutput,
  Ltx2ConnectorTransformer1d,
} from "./connectors-ltx2-transformer";

export type Ltx2TextConnectorOutput = {
  videoPromptEmbeds: MxArray;
  audioPromptEmbeds: MxArray;
  attentionMask: MxArray;
};

/** Diffusers-compatible LTX-2 text connector stack for video and audio branches. */
export class Ltx2TextConnectors extends Module {
  textProjIn: Linear | null;
  videoTextProjIn: Linear | null;
  audioTextProjIn: Linear | null;
  videoConnector: Ltx2ConnectorTransformer1d;
  audioConnector: Ltx2ConnectorTransformer1d;
  #config: Ltx2TextConnectorsConfig;

  constructor(config: Ltx2TextConnectorsConfig) {
    super();
    this.#config = config;
    this.textProjIn = config.perModalityProjections
      ? null
      : new Linear(config.textEncoderDim, config.captionChannels, config.projBias);
    this.videoTextProjIn = config.perModalityProjections
      ? new Linear(config.textEncoderDim, config.videoHiddenDim, config.projBias)
      : null;
    this.audioTextProjIn = config.perModalityProjections
      ? new Linear(config.textEncoderDim, config.audioHiddenDim, config.projBias)
      : null;
    this.videoConnector = new Ltx2ConnectorTransformer1d({
      heads: config.videoConnectorNumAttentionHeads,
      headDim: config.videoConnectorAttentionHeadDim,
      numLayers: config.videoConnectorNumLayers,
      numLearnableRegisters: config.videoConnectorNumLearnableRegisters,
      ropeBaseSeqLen: config.connectorRopeBaseSeqLen,
      ropeTheta: config.ropeTheta,
      ropeType: config.ropeType,
      gatedAttention: config.videoGatedAttn,
    });
    this.audioConnector = new Ltx2ConnectorTransformer1d({
      heads: config.audioConnectorNumAttentionHeads,
      headDim: config.audioConnectorAttentionHeadDim,
      numLayers: config.audioConnectorNumLayers,
      numLearnableRegisters: config.audioConnectorNumLearnableRegisters,
      ropeBaseSeqLen: config.connectorRopeBaseSeqLen,
      ropeTheta: config.ropeTheta,
      ropeType: config.ropeType,
      gatedAttention: config.audioGatedAttn,
    });
  }

  forward(): MxArray {
    throw new Error("Ltx2TextConnectors.forward: use run() with an attention mask.");
  }

  /** Produce separate video/audio prompt embeddings from a Gemma hidden-state stack. */
  run(textEncoderHiddenStates: MxArray, attentionMask: MxArray): Ltx2TextConnectorOutput {
    const { videoInput, audioInput } = this.#projectInputs(textEncoderHiddenStates, attentionMask);
    try {
      const video = this.videoConnector.run(videoInput, attentionMask);
      try {
        const audio = this.audioConnector.run(audioInput, attentionMask);
        try {
          return {
            videoPromptEmbeds: retainArray(video.hiddenStates),
            audioPromptEmbeds: retainArray(audio.hiddenStates),
            attentionMask: retainArray(video.attentionMask),
          };
        } finally {
          disposeLtx2ConnectorTransformerOutput(audio);
        }
      } finally {
        disposeLtx2ConnectorTransformerOutput(video);
      }
    } finally {
      videoInput.free();
      audioInput.free();
    }
  }

  get config(): Ltx2TextConnectorsConfig {
    return this.#config;
  }

  #projectInputs(
    textEncoderHiddenStates: MxArray,
    attentionMask: MxArray,
  ): { videoInput: MxArray; audioInput: MxArray } {
    if (this.#config.perModalityProjections) {
      using normalized = ltx2PerTokenRmsNorm(textEncoderHiddenStates, attentionMask, this.#config);
      using videoScaled = multiply(
        normalized,
        Math.sqrt(this.#config.videoHiddenDim / this.#config.captionChannels),
      );
      using audioScaled = multiply(
        normalized,
        Math.sqrt(this.#config.audioHiddenDim / this.#config.captionChannels),
      );
      const videoProjection = this.videoTextProjIn;
      const audioProjection = this.audioTextProjIn;
      if (videoProjection === null || audioProjection === null) {
        throw new Error("Ltx2TextConnectors: missing per-modality projections.");
      }
      return {
        videoInput: videoProjection.forward(videoScaled),
        audioInput: audioProjection.forward(audioScaled),
      };
    }
    using normalized = ltx2PerLayerMaskedMeanNorm(
      textEncoderHiddenStates,
      attentionMask,
      this.#config,
    );
    const projection = this.textProjIn;
    if (projection === null) {
      throw new Error("Ltx2TextConnectors: missing shared text projection.");
    }
    const projected = projection.forward(normalized);
    try {
      return { videoInput: retainArray(projected), audioInput: retainArray(projected) };
    } finally {
      projected.free();
    }
  }
}

/** Dispose tensors returned by `Ltx2TextConnectors.run`. */
export function disposeLtx2TextConnectorOutput(output: Ltx2TextConnectorOutput): void {
  output.videoPromptEmbeds.free();
  output.audioPromptEmbeds.free();
  output.attentionMask.free();
}
