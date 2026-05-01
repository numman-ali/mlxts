/**
 * Whisper encoder-decoder model.
 * @module
 */

import { add, arange, formatShape, isIntegerDType, type MxArray, retainArray } from "@mlxts/core";
import { Conv1d, Embedding, gelu, LayerNorm, Module } from "@mlxts/nn";

import { WhisperDecoderBlock, WhisperEncoderBlock } from "./block";
import type {
  WhisperConditionalGenerationOutput,
  WhisperConfig,
  WhisperDecoderOutput,
  WhisperEncoderOutput,
  WhisperModelOutput,
  WhisperRunOptions,
} from "./types";

function disposeHiddenStates(hiddenStates: MxArray[] | undefined): void {
  if (hiddenStates === undefined) {
    return;
  }
  for (const hiddenState of hiddenStates) {
    hiddenState.free();
  }
}

/** Dispose arrays returned by `WhisperModel.run`. */
export function disposeWhisperModelOutput(output: WhisperModelOutput): void {
  output.lastHiddenState.free();
  output.encoderLastHiddenState.free();
  disposeHiddenStates(output.decoderHiddenStates);
  disposeHiddenStates(output.encoderHiddenStates);
}

/** Dispose arrays returned by `WhisperForConditionalGeneration.run`. */
export function disposeWhisperConditionalGenerationOutput(
  output: WhisperConditionalGenerationOutput,
): void {
  output.logits.free();
  disposeWhisperModelOutput(output);
}

/** Whisper audio encoder over channel-last log-mel features. */
export class WhisperAudioEncoder extends Module {
  conv1: Conv1d;
  conv2: Conv1d;
  positionEmbedding: Embedding;
  layers: WhisperEncoderBlock[];
  layerNorm: LayerNorm;
  #numMelBins: number;
  #dModel: number;
  #maxSourcePositions: number;

  constructor(config: WhisperConfig) {
    super();
    this.conv1 = new Conv1d(config.numMelBins, config.dModel, 3, 1, 1);
    this.conv2 = new Conv1d(config.dModel, config.dModel, 3, 2, 1);
    this.positionEmbedding = new Embedding(config.maxSourcePositions, config.dModel);
    this.layers = Array.from(
      { length: config.encoderLayers },
      () => new WhisperEncoderBlock(config),
    );
    this.layerNorm = new LayerNorm(config.dModel);
    this.#numMelBins = config.numMelBins;
    this.#dModel = config.dModel;
    this.#maxSourcePositions = config.maxSourcePositions;
  }

  forward(inputFeatures: MxArray): MxArray {
    const output = this.run(inputFeatures);
    disposeHiddenStates(output.hiddenStates);
    return output.lastHiddenState;
  }

  run(inputFeatures: MxArray, options: WhisperRunOptions = {}): WhisperEncoderOutput {
    this.assertInputFeatures(inputFeatures);
    using conv1Output = this.conv1.forward(inputFeatures);
    using conv1Activated = gelu(conv1Output);
    let hidden = this.conv2.forward(conv1Activated);
    let lastHiddenState: MxArray | null = null;
    const hiddenStates: MxArray[] | undefined = options.outputHiddenStates ? [] : undefined;

    try {
      this.assertEncodedShape(hidden);
      using positionIds = arange(0, this.#maxSourcePositions, 1, "int32");
      using positionEmbeddings = this.positionEmbedding.forward(positionIds);
      const embedded = add(hidden, positionEmbeddings);
      hidden.free();
      hidden = embedded;
      if (hiddenStates !== undefined) {
        hiddenStates.push(retainArray(hidden));
      }

      for (const layer of this.layers) {
        const nextHidden = layer.run(hidden);
        hidden.free();
        hidden = nextHidden;
        if (hiddenStates !== undefined) {
          hiddenStates.push(retainArray(hidden));
        }
      }

      lastHiddenState = this.layerNorm.forward(hidden);
      if (hiddenStates !== undefined) {
        hiddenStates.push(retainArray(lastHiddenState));
      }
      const output: WhisperEncoderOutput = { lastHiddenState };
      if (hiddenStates !== undefined) {
        output.hiddenStates = hiddenStates;
      }
      return output;
    } catch (error) {
      lastHiddenState?.free();
      disposeHiddenStates(hiddenStates);
      throw error;
    } finally {
      hidden.free();
    }
  }

  private assertInputFeatures(inputFeatures: MxArray): void {
    const [batch, frames, melBins] = inputFeatures.shape;
    const expectedFrames = this.#maxSourcePositions * 2;
    if (
      batch === undefined ||
      frames !== expectedFrames ||
      melBins !== this.#numMelBins ||
      inputFeatures.shape.length !== 3
    ) {
      throw new Error(
        `WhisperAudioEncoder.run: expected [batch, ${expectedFrames}, ${this.#numMelBins}], got ${formatShape(inputFeatures.shape)}.`,
      );
    }
  }

  private assertEncodedShape(hidden: MxArray): void {
    const [batch, frames, hiddenSize] = hidden.shape;
    if (
      batch === undefined ||
      frames !== this.#maxSourcePositions ||
      hiddenSize !== this.#dModel ||
      hidden.shape.length !== 3
    ) {
      throw new Error(
        `WhisperAudioEncoder.run: expected encoded shape [batch, ${this.#maxSourcePositions}, ${this.#dModel}], got ${formatShape(hidden.shape)}.`,
      );
    }
  }
}

/** Whisper autoregressive text decoder cross-attending to audio features. */
export class WhisperTextDecoder extends Module {
  tokenEmbedding: Embedding;
  positionEmbedding: Embedding;
  layers: WhisperDecoderBlock[];
  layerNorm: LayerNorm;
  #dModel: number;
  #maxTargetPositions: number;

  constructor(config: WhisperConfig) {
    super();
    this.tokenEmbedding = new Embedding(config.vocabSize, config.dModel);
    this.positionEmbedding = new Embedding(config.maxTargetPositions, config.dModel);
    this.layers = Array.from(
      { length: config.decoderLayers },
      () => new WhisperDecoderBlock(config),
    );
    this.layerNorm = new LayerNorm(config.dModel);
    this.#dModel = config.dModel;
    this.#maxTargetPositions = config.maxTargetPositions;
  }

  forward(inputIds: MxArray, encoderHiddenStates: MxArray): MxArray {
    const output = this.run(inputIds, encoderHiddenStates);
    disposeHiddenStates(output.hiddenStates);
    return output.lastHiddenState;
  }

  run(
    inputIds: MxArray,
    encoderHiddenStates: MxArray,
    options: WhisperRunOptions = {},
  ): WhisperDecoderOutput {
    this.assertInputIds(inputIds);
    this.assertEncoderHiddenStates(encoderHiddenStates);
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === undefined) {
      throw new Error("WhisperTextDecoder.run: input ids are missing a sequence axis.");
    }

    using tokenEmbeddings = this.tokenEmbedding.forward(inputIds);
    using positionIds = arange(0, sequenceLength, 1, "int32");
    using positionEmbeddings = this.positionEmbedding.forward(positionIds);
    let hidden = add(tokenEmbeddings, positionEmbeddings);
    let lastHiddenState: MxArray | null = null;
    const hiddenStates: MxArray[] | undefined = options.outputHiddenStates
      ? [retainArray(hidden)]
      : undefined;

    try {
      for (const layer of this.layers) {
        const nextHidden = layer.run(hidden, encoderHiddenStates);
        hidden.free();
        hidden = nextHidden;
        if (hiddenStates !== undefined) {
          hiddenStates.push(retainArray(hidden));
        }
      }

      lastHiddenState = this.layerNorm.forward(hidden);
      if (hiddenStates !== undefined) {
        hiddenStates.push(retainArray(lastHiddenState));
      }
      const output: WhisperDecoderOutput = { lastHiddenState };
      if (hiddenStates !== undefined) {
        output.hiddenStates = hiddenStates;
      }
      return output;
    } catch (error) {
      lastHiddenState?.free();
      disposeHiddenStates(hiddenStates);
      throw error;
    } finally {
      hidden.free();
    }
  }

  projectLogits(hiddenStates: MxArray): MxArray {
    return this.tokenEmbedding.asLinear(hiddenStates);
  }

  private assertInputIds(inputIds: MxArray): void {
    const [batch, sequenceLength] = inputIds.shape;
    if (
      batch === undefined ||
      sequenceLength === undefined ||
      inputIds.shape.length !== 2 ||
      !isIntegerDType(inputIds.dtype)
    ) {
      throw new Error(
        `WhisperTextDecoder.run: expected rank-2 integer token ids, got ${formatShape(inputIds.shape)} with dtype ${inputIds.dtype}.`,
      );
    }
    if (sequenceLength > this.#maxTargetPositions) {
      throw new Error(
        `WhisperTextDecoder.run: sequence length ${sequenceLength} exceeds max_target_positions ${this.#maxTargetPositions}.`,
      );
    }
  }

  private assertEncoderHiddenStates(encoderHiddenStates: MxArray): void {
    const [batch, sequenceLength, hiddenSize] = encoderHiddenStates.shape;
    if (
      batch === undefined ||
      sequenceLength === undefined ||
      hiddenSize !== this.#dModel ||
      encoderHiddenStates.shape.length !== 3
    ) {
      throw new Error(
        `WhisperTextDecoder.run: expected encoder states [batch, seq, ${this.#dModel}], got ${formatShape(encoderHiddenStates.shape)}.`,
      );
    }
  }
}

/** Whisper base encoder-decoder model without the tied vocabulary projection wrapper. */
export class WhisperModel extends Module {
  encoder: WhisperAudioEncoder;
  decoder: WhisperTextDecoder;

  constructor(config: WhisperConfig) {
    super();
    this.encoder = new WhisperAudioEncoder(config);
    this.decoder = new WhisperTextDecoder(config);
  }

  forward(inputFeatures: MxArray, decoderInputIds: MxArray): MxArray {
    const output = this.run(inputFeatures, decoderInputIds);
    output.encoderLastHiddenState.free();
    disposeHiddenStates(output.encoderHiddenStates);
    disposeHiddenStates(output.decoderHiddenStates);
    return output.lastHiddenState;
  }

  run(
    inputFeatures: MxArray,
    decoderInputIds: MxArray,
    options: WhisperRunOptions = {},
  ): WhisperModelOutput {
    const encoderOutput = this.encoder.run(inputFeatures, options);
    let decoderOutput: WhisperDecoderOutput | null = null;
    try {
      decoderOutput = this.decoder.run(decoderInputIds, encoderOutput.lastHiddenState, options);
      const output: WhisperModelOutput = {
        lastHiddenState: decoderOutput.lastHiddenState,
        encoderLastHiddenState: encoderOutput.lastHiddenState,
      };
      if (encoderOutput.hiddenStates !== undefined) {
        output.encoderHiddenStates = encoderOutput.hiddenStates;
      }
      if (decoderOutput.hiddenStates !== undefined) {
        output.decoderHiddenStates = decoderOutput.hiddenStates;
      }
      return output;
    } catch (error) {
      decoderOutput?.lastHiddenState.free();
      disposeHiddenStates(decoderOutput?.hiddenStates);
      encoderOutput.lastHiddenState.free();
      disposeHiddenStates(encoderOutput.hiddenStates);
      throw error;
    }
  }
}

/** Whisper conditional-generation model with tied decoder embedding projection. */
export class WhisperForConditionalGeneration extends Module {
  model: WhisperModel;
  readonly config: WhisperConfig;

  constructor(config: WhisperConfig) {
    super();
    this.config = config;
    this.model = new WhisperModel(config);
  }

  forward(inputFeatures: MxArray, decoderInputIds: MxArray): MxArray {
    const output = this.run(inputFeatures, decoderInputIds);
    output.lastHiddenState.free();
    output.encoderLastHiddenState.free();
    disposeHiddenStates(output.encoderHiddenStates);
    disposeHiddenStates(output.decoderHiddenStates);
    return output.logits;
  }

  run(
    inputFeatures: MxArray,
    decoderInputIds: MxArray,
    options: WhisperRunOptions = {},
  ): WhisperConditionalGenerationOutput {
    const output = this.model.run(inputFeatures, decoderInputIds, options);
    let logits: MxArray | null = null;
    try {
      logits = this.model.decoder.projectLogits(output.lastHiddenState);
      return {
        ...output,
        logits,
      };
    } catch (error) {
      logits?.free();
      disposeWhisperModelOutput(output);
      throw error;
    }
  }
}
