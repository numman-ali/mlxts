import { formatShape, MxArray, multiply, random, retainArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import { disposeLtx2VideoTransformerBlockOutput, Ltx2VideoTransformerBlock } from "./blocks-ltx2";
import { LtxVideoCaptionProjection } from "./conditioning";
import { disposeLtx2AdaLayerNormSingleOutput, Ltx2AdaLayerNormSingle } from "./conditioning-ltx2";
import type { Ltx2VideoTransformerConfig } from "./config";
import { createLtx2RotaryEmbeddings, type LtxRotaryEmbeddings } from "./embeddings";
import type { Ltx2DenoiserInput, Ltx2DenoiserOutput } from "./pipeline-ltx2-types";
import { assertSequence3d, checkedModule } from "./tensor-utils";
import { projectLtx2FinalOutput } from "./transformer-ltx2-final";
import {
  disposeRotaryEmbeddings,
  disposeTiming,
  disposeTransformerEmbeddings,
  encoderAttentionMask,
  type Ltx2TransformerEmbeddings,
  type Ltx2TransformerTiming,
  temporalVideoCoords,
  timestepLike,
  validateLtx2VideoTransformerConfig,
} from "./transformer-ltx2-state";

/** Diffusers-compatible LTX-2 `LTX2VideoTransformer3DModel` prepared tensor path. */
export class Ltx2VideoTransformer3DModel extends Module {
  projIn: Linear;
  audioProjIn: Linear;
  captionProjection: LtxVideoCaptionProjection | null;
  audioCaptionProjection: LtxVideoCaptionProjection | null;
  timeEmbed: Ltx2AdaLayerNormSingle;
  audioTimeEmbed: Ltx2AdaLayerNormSingle;
  avCrossAttnVideoScaleShift: Ltx2AdaLayerNormSingle;
  avCrossAttnAudioScaleShift: Ltx2AdaLayerNormSingle;
  avCrossAttnVideoA2vGate: Ltx2AdaLayerNormSingle;
  avCrossAttnAudioV2aGate: Ltx2AdaLayerNormSingle;
  scaleShiftTable: MxArray;
  audioScaleShiftTable: MxArray;
  transformerBlocks: Ltx2VideoTransformerBlock[];
  normOut: null;
  projOut: Linear;
  audioNormOut: null;
  audioProjOut: Linear;
  #config: Ltx2VideoTransformerConfig;

  constructor(config: Ltx2VideoTransformerConfig) {
    super();
    validateLtx2VideoTransformerConfig(config);
    this.#config = config;
    this.projIn = new Linear(config.inChannels, config.hiddenSize);
    this.audioProjIn = new Linear(config.audioInChannels, config.audioHiddenSize);
    this.captionProjection = config.usePromptEmbeddings
      ? new LtxVideoCaptionProjection(config.captionChannels, config.hiddenSize)
      : null;
    this.audioCaptionProjection = config.usePromptEmbeddings
      ? new LtxVideoCaptionProjection(config.captionChannels, config.audioHiddenSize)
      : null;
    this.timeEmbed = new Ltx2AdaLayerNormSingle(config.hiddenSize, 6);
    this.audioTimeEmbed = new Ltx2AdaLayerNormSingle(config.audioHiddenSize, 6);
    this.avCrossAttnVideoScaleShift = new Ltx2AdaLayerNormSingle(config.hiddenSize, 4);
    this.avCrossAttnAudioScaleShift = new Ltx2AdaLayerNormSingle(config.audioHiddenSize, 4);
    this.avCrossAttnVideoA2vGate = new Ltx2AdaLayerNormSingle(config.hiddenSize, 1);
    this.avCrossAttnAudioV2aGate = new Ltx2AdaLayerNormSingle(config.audioHiddenSize, 1);
    this.scaleShiftTable = this.#scaledNormal([2, config.hiddenSize], config.hiddenSize ** -0.5);
    this.audioScaleShiftTable = this.#scaledNormal(
      [2, config.audioHiddenSize],
      config.audioHiddenSize ** -0.5,
    );
    this.transformerBlocks = Array.from(
      { length: config.numLayers },
      () => new Ltx2VideoTransformerBlock(config),
    );
    this.normOut = null;
    this.projOut = new Linear(config.hiddenSize, config.outChannels);
    this.audioNormOut = null;
    this.audioProjOut = new Linear(config.audioHiddenSize, config.audioOutChannels);
  }

  forward(input: Ltx2DenoiserInput): Ltx2DenoiserOutput;
  forward(...args: MxArray[]): MxArray;
  /** Run an LTX-2 denoising prediction over packed video and audio latents. */
  forward(inputOrTensor: Ltx2DenoiserInput | MxArray): Ltx2DenoiserOutput | MxArray {
    if (inputOrTensor instanceof MxArray || !("hiddenStates" in inputOrTensor)) {
      throw new Error("Ltx2VideoTransformer3DModel.forward: expected an Ltx2DenoiserInput object.");
    }
    const input = inputOrTensor;
    const videoShape = assertSequence3d(
      input.hiddenStates,
      "Ltx2VideoTransformer3DModel.forward hiddenStates",
    );
    const audioShape = assertSequence3d(
      input.audioHiddenStates,
      "Ltx2VideoTransformer3DModel.forward audioHiddenStates",
    );
    const textShape = assertSequence3d(
      input.encoderHiddenStates,
      "Ltx2VideoTransformer3DModel.forward encoderHiddenStates",
    );
    const audioTextShape = assertSequence3d(
      input.audioEncoderHiddenStates,
      "Ltx2VideoTransformer3DModel.forward audioEncoderHiddenStates",
    );
    this.#validateInput(input, videoShape, audioShape, textShape, audioTextShape);

    const embeddings = this.#rotaryEmbeddings(input);
    const timing = this.#timing(input, input.hiddenStates.dtype);
    const encoderMask = encoderAttentionMask(
      input.encoderAttentionMask,
      videoShape.batch,
      textShape.length,
    );
    try {
      const audioEncoderMask = encoderAttentionMask(
        input.audioEncoderAttentionMask,
        audioShape.batch,
        audioTextShape.length,
      );
      try {
        using videoProjected = this.projIn.forward(input.hiddenStates);
        using audioProjected = this.audioProjIn.forward(input.audioHiddenStates);
        const projectedText = this.#projectText(input.encoderHiddenStates, "video");
        try {
          const projectedAudioText = this.#projectText(input.audioEncoderHiddenStates, "audio");
          try {
            const blockOutput = this.#runBlocks({
              video: videoProjected,
              audio: audioProjected,
              text: projectedText,
              audioText: projectedAudioText,
              encoderMask,
              audioEncoderMask,
              embeddings,
              timing,
            });
            try {
              const video = projectLtx2FinalOutput({
                hiddenStates: blockOutput.hiddenStates,
                embeddedTimestep: timing.video.embeddedTimestep,
                scaleShiftTable: this.scaleShiftTable,
                projection: this.projOut,
                hiddenSize: this.#config.hiddenSize,
                name: "video",
              });
              try {
                const audio = projectLtx2FinalOutput({
                  hiddenStates: blockOutput.audioHiddenStates,
                  embeddedTimestep: timing.audio.embeddedTimestep,
                  scaleShiftTable: this.audioScaleShiftTable,
                  projection: this.audioProjOut,
                  hiddenSize: this.#config.audioHiddenSize,
                  name: "audio",
                });
                return { video, audio };
              } catch (error) {
                video.free();
                throw error;
              }
            } finally {
              disposeLtx2VideoTransformerBlockOutput(blockOutput);
            }
          } finally {
            projectedAudioText.free();
          }
        } finally {
          projectedText.free();
        }
      } finally {
        audioEncoderMask.free();
      }
    } finally {
      encoderMask.free();
      disposeTiming(timing);
      disposeTransformerEmbeddings(embeddings);
    }
  }

  get config(): Ltx2VideoTransformerConfig {
    return this.#config;
  }

  #scaledNormal(shape: number[], scale: number): MxArray {
    using values = random.normal(shape);
    return multiply(values, scale);
  }

  #validateInput(
    input: Ltx2DenoiserInput,
    videoShape: { batch: number; length: number; channels: number },
    audioShape: { batch: number; length: number; channels: number },
    textShape: { batch: number; length: number; channels: number },
    audioTextShape: { batch: number; length: number; channels: number },
  ): void {
    if (videoShape.channels !== this.#config.inChannels) {
      throw new Error("Ltx2VideoTransformer3DModel.forward: hiddenStates channel mismatch.");
    }
    if (
      audioShape.batch !== videoShape.batch ||
      audioShape.channels !== this.#config.audioInChannels
    ) {
      throw new Error("Ltx2VideoTransformer3DModel.forward: audioHiddenStates shape mismatch.");
    }
    const expectedTextChannels = this.#config.usePromptEmbeddings
      ? this.#config.captionChannels
      : this.#config.crossAttentionDim;
    const expectedAudioTextChannels = this.#config.usePromptEmbeddings
      ? this.#config.captionChannels
      : this.#config.audioCrossAttentionDim;
    if (textShape.batch !== videoShape.batch || textShape.channels !== expectedTextChannels) {
      throw new Error("Ltx2VideoTransformer3DModel.forward: encoderHiddenStates shape mismatch.");
    }
    if (
      audioTextShape.batch !== videoShape.batch ||
      audioTextShape.length !== textShape.length ||
      audioTextShape.channels !== expectedAudioTextChannels
    ) {
      throw new Error(
        "Ltx2VideoTransformer3DModel.forward: audioEncoderHiddenStates shape mismatch.",
      );
    }
    if (input.timestep.shape.length !== 1 || input.timestep.shape[0] !== videoShape.batch) {
      throw new Error(
        `Ltx2VideoTransformer3DModel.forward: timestep must have shape [${videoShape.batch}], got ${formatShape(
          input.timestep.shape,
        )}.`,
      );
    }
    if (input.sigma.shape.length !== 1 || input.sigma.shape[0] !== videoShape.batch) {
      throw new Error(
        `Ltx2VideoTransformer3DModel.forward: sigma must have shape [${videoShape.batch}], got ${formatShape(
          input.sigma.shape,
        )}.`,
      );
    }
    this.#validateCoords(input.videoCoords, videoShape.batch, 3, videoShape.length, "videoCoords");
    this.#validateCoords(input.audioCoords, audioShape.batch, 1, audioShape.length, "audioCoords");
  }

  #validateCoords(
    coords: MxArray,
    batch: number,
    axes: number,
    length: number,
    name: string,
  ): void {
    if (
      coords.shape.length !== 4 ||
      coords.shape[0] !== batch ||
      coords.shape[1] !== axes ||
      coords.shape[2] !== length ||
      coords.shape[3] !== 2
    ) {
      throw new Error(
        `Ltx2VideoTransformer3DModel.forward: ${name} must have shape [${batch}, ${axes}, ${length}, 2], got ${formatShape(
          coords.shape,
        )}.`,
      );
    }
  }

  #rotaryEmbeddings(input: Ltx2DenoiserInput): Ltx2TransformerEmbeddings {
    const videoTemporalCoords = temporalVideoCoords(input.videoCoords);
    let videoRotary: LtxRotaryEmbeddings | null = null;
    let audioRotary: LtxRotaryEmbeddings | null = null;
    let caVideoRotary: LtxRotaryEmbeddings | null = null;
    let caAudioRotary: LtxRotaryEmbeddings | null = null;
    try {
      videoRotary = createLtx2RotaryEmbeddings({
        coords: input.videoCoords,
        dim: this.#config.hiddenSize,
        modality: "video",
        ropeType: this.#config.ropeType,
        theta: this.#config.ropeTheta,
        baseNumFrames: this.#config.posEmbedMaxPos,
        baseHeight: this.#config.baseHeight,
        baseWidth: this.#config.baseWidth,
        numAttentionHeads: this.#config.numAttentionHeads,
      });
      audioRotary = createLtx2RotaryEmbeddings({
        coords: input.audioCoords,
        dim: this.#config.audioHiddenSize,
        modality: "audio",
        ropeType: this.#config.ropeType,
        theta: this.#config.ropeTheta,
        baseNumFrames: this.#config.audioPosEmbedMaxPos,
        numAttentionHeads: this.#config.audioNumAttentionHeads,
      });
      caVideoRotary = createLtx2RotaryEmbeddings({
        coords: videoTemporalCoords,
        dim: this.#config.audioCrossAttentionDim,
        modality: "video",
        ropeType: this.#config.ropeType,
        theta: this.#config.ropeTheta,
        baseNumFrames: Math.max(this.#config.posEmbedMaxPos, this.#config.audioPosEmbedMaxPos),
        numAttentionHeads: this.#config.audioNumAttentionHeads,
      });
      caAudioRotary = createLtx2RotaryEmbeddings({
        coords: input.audioCoords,
        dim: this.#config.audioCrossAttentionDim,
        modality: "audio",
        ropeType: this.#config.ropeType,
        theta: this.#config.ropeTheta,
        baseNumFrames: Math.max(this.#config.posEmbedMaxPos, this.#config.audioPosEmbedMaxPos),
        numAttentionHeads: this.#config.audioNumAttentionHeads,
      });
      return { videoRotary, audioRotary, caVideoRotary, caAudioRotary };
    } catch (error) {
      if (videoRotary !== null) {
        disposeRotaryEmbeddings(videoRotary);
      }
      if (audioRotary !== null) {
        disposeRotaryEmbeddings(audioRotary);
      }
      if (caVideoRotary !== null) {
        disposeRotaryEmbeddings(caVideoRotary);
      }
      if (caAudioRotary !== null) {
        disposeRotaryEmbeddings(caAudioRotary);
      }
      throw error;
    } finally {
      videoTemporalCoords.free();
    }
  }

  #timing(input: Ltx2DenoiserInput, dtype: MxArray["dtype"]): Ltx2TransformerTiming {
    const videoCrossBase = timestepLike(input, input.useCrossTimestep);
    const audioCrossBase = timestepLike(input, input.useCrossTimestep);
    let video: ReturnType<Ltx2AdaLayerNormSingle["embed"]> | null = null;
    let audio: ReturnType<Ltx2AdaLayerNormSingle["embed"]> | null = null;
    let videoCrossScaleShift: ReturnType<Ltx2AdaLayerNormSingle["embed"]> | null = null;
    let audioCrossScaleShift: ReturnType<Ltx2AdaLayerNormSingle["embed"]> | null = null;
    let videoCrossGate: ReturnType<Ltx2AdaLayerNormSingle["embed"]> | null = null;
    let audioCrossGate: ReturnType<Ltx2AdaLayerNormSingle["embed"]> | null = null;
    try {
      using videoCrossGateTimestep = multiply(
        videoCrossBase,
        this.#config.crossAttnTimestepScaleMultiplier / this.#config.timestepScaleMultiplier,
      );
      using audioCrossGateTimestep = multiply(
        audioCrossBase,
        this.#config.crossAttnTimestepScaleMultiplier / this.#config.timestepScaleMultiplier,
      );
      video = this.timeEmbed.embed(input.timestep, dtype);
      audio = this.audioTimeEmbed.embed(input.timestep, dtype);
      videoCrossScaleShift = this.avCrossAttnVideoScaleShift.embed(videoCrossBase, dtype);
      audioCrossScaleShift = this.avCrossAttnAudioScaleShift.embed(audioCrossBase, dtype);
      videoCrossGate = this.avCrossAttnVideoA2vGate.embed(videoCrossGateTimestep, dtype);
      audioCrossGate = this.avCrossAttnAudioV2aGate.embed(audioCrossGateTimestep, dtype);
      return {
        video,
        audio,
        videoCrossScaleShift,
        audioCrossScaleShift,
        videoCrossGate,
        audioCrossGate,
      };
    } catch (error) {
      if (video !== null) {
        disposeLtx2AdaLayerNormSingleOutput(video);
      }
      if (audio !== null) {
        disposeLtx2AdaLayerNormSingleOutput(audio);
      }
      if (videoCrossScaleShift !== null) {
        disposeLtx2AdaLayerNormSingleOutput(videoCrossScaleShift);
      }
      if (audioCrossScaleShift !== null) {
        disposeLtx2AdaLayerNormSingleOutput(audioCrossScaleShift);
      }
      if (videoCrossGate !== null) {
        disposeLtx2AdaLayerNormSingleOutput(videoCrossGate);
      }
      if (audioCrossGate !== null) {
        disposeLtx2AdaLayerNormSingleOutput(audioCrossGate);
      }
      throw error;
    } finally {
      videoCrossBase.free();
      audioCrossBase.free();
    }
  }

  #projectText(hiddenStates: MxArray, modality: "video" | "audio"): MxArray {
    if (!this.#config.usePromptEmbeddings) {
      return retainArray(hiddenStates);
    }
    const projection = modality === "video" ? this.captionProjection : this.audioCaptionProjection;
    if (projection === null) {
      throw new Error("Ltx2VideoTransformer3DModel: missing prompt projection.");
    }
    return projection.forward(hiddenStates);
  }

  #runBlocks(input: {
    video: MxArray;
    audio: MxArray;
    text: MxArray;
    audioText: MxArray;
    encoderMask: MxArray;
    audioEncoderMask: MxArray;
    embeddings: Ltx2TransformerEmbeddings;
    timing: Ltx2TransformerTiming;
  }): { hiddenStates: MxArray; audioHiddenStates: MxArray } {
    let video = retainArray(input.video);
    let audio = retainArray(input.audio);
    try {
      for (let index = 0; index < this.transformerBlocks.length; index += 1) {
        const block = checkedModule(
          this.transformerBlocks,
          index,
          "Ltx2VideoTransformer3DModel.forward transformerBlocks",
        );
        const output = block.run({
          hiddenStates: video,
          audioHiddenStates: audio,
          encoderHiddenStates: input.text,
          audioEncoderHiddenStates: input.audioText,
          temb: input.timing.video.modulation,
          tembAudio: input.timing.audio.modulation,
          tembCaScaleShift: input.timing.videoCrossScaleShift.modulation,
          tembCaAudioScaleShift: input.timing.audioCrossScaleShift.modulation,
          tembCaGate: input.timing.videoCrossGate.modulation,
          tembCaAudioGate: input.timing.audioCrossGate.modulation,
          videoRotaryEmbeddings: input.embeddings.videoRotary,
          audioRotaryEmbeddings: input.embeddings.audioRotary,
          caVideoRotaryEmbeddings: input.embeddings.caVideoRotary,
          caAudioRotaryEmbeddings: input.embeddings.caAudioRotary,
          encoderAttentionMask: input.encoderMask,
          audioEncoderAttentionMask: input.audioEncoderMask,
        });
        video.free();
        audio.free();
        video = output.hiddenStates;
        audio = output.audioHiddenStates;
      }
      return { hiddenStates: video, audioHiddenStates: audio };
    } catch (error) {
      video.free();
      audio.free();
      throw error;
    }
  }
}
