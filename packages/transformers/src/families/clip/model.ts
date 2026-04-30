/**
 * CLIP text encoder model.
 * @module
 */

import {
  add,
  arange,
  argmax,
  asType,
  broadcastTo,
  equal,
  expandDims,
  formatShape,
  type MxArray,
  retainArray,
  squeeze,
  takeAlongAxis,
} from "@mlxts/core";
import { Embedding, LayerNorm, Linear, Module } from "@mlxts/nn";

import { CLIPEncoderBlock } from "./block";
import type {
  CLIPTextConfig,
  CLIPTextModelOptions,
  CLIPTextModelOutput,
  CLIPTextProjectionOutput,
} from "./types";

function disposeHiddenStates(hiddenStates: MxArray[] | undefined): void {
  if (hiddenStates === undefined) {
    return;
  }
  for (const hiddenState of hiddenStates) {
    hiddenState.free();
  }
}

/** Dispose arrays returned by `CLIPTextModel.run`. */
export function disposeCLIPTextModelOutput(output: CLIPTextModelOutput): void {
  output.lastHiddenState.free();
  output.pooledOutput.free();
  disposeHiddenStates(output.hiddenStates);
}

/** Dispose arrays returned by `CLIPTextModelWithProjection.run`. */
export function disposeCLIPTextProjectionOutput(output: CLIPTextProjectionOutput): void {
  output.textEmbeds.free();
  disposeCLIPTextModelOutput(output);
}

/** Token and position embeddings for CLIP text input ids. */
export class CLIPTextEmbeddings extends Module {
  tokenEmbedding: Embedding;
  positionEmbedding: Embedding;
  #maxPositionEmbeddings: number;
  #hiddenSize: number;

  constructor(config: CLIPTextConfig) {
    super();
    this.tokenEmbedding = new Embedding(config.vocabSize, config.hiddenSize);
    this.positionEmbedding = new Embedding(config.maxPositionEmbeddings, config.hiddenSize);
    this.#maxPositionEmbeddings = config.maxPositionEmbeddings;
    this.#hiddenSize = config.hiddenSize;
  }

  forward(inputIds: MxArray): MxArray {
    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(
        `CLIPTextEmbeddings.forward: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`,
      );
    }
    if (sequenceLength > this.#maxPositionEmbeddings) {
      throw new Error(
        `CLIPTextEmbeddings.forward: sequence length ${sequenceLength} exceeds max_position_embeddings ${this.#maxPositionEmbeddings}.`,
      );
    }

    using tokenEmbeddings = this.tokenEmbedding.forward(inputIds);
    using positionIds = arange(0, sequenceLength, 1, "int32");
    using positionEmbeddings = this.positionEmbedding.forward(positionIds);
    const embeddings = add(tokenEmbeddings, positionEmbeddings);
    const hiddenSize = embeddings.shape[2];
    if (hiddenSize !== this.#hiddenSize) {
      embeddings.free();
      throw new Error(
        `CLIPTextEmbeddings.forward: expected hidden size ${this.#hiddenSize}, got ${hiddenSize ?? "undefined"}.`,
      );
    }
    return embeddings;
  }
}

/** CLIP text transformer encoder without the projection head. */
export class CLIPTextModel extends Module {
  embeddings: CLIPTextEmbeddings;
  layers: CLIPEncoderBlock[];
  finalLayerNorm: LayerNorm;
  #eosTokenId: number | null;
  #hiddenSize: number;

  constructor(config: CLIPTextConfig) {
    super();
    this.embeddings = new CLIPTextEmbeddings(config);
    this.layers = Array.from(
      { length: config.numHiddenLayers },
      () => new CLIPEncoderBlock(config),
    );
    this.finalLayerNorm = new LayerNorm(config.hiddenSize, config.layerNormEps);
    this.#eosTokenId = config.eosTokenId;
    this.#hiddenSize = config.hiddenSize;
  }

  forward(inputIds: MxArray): MxArray {
    const output = this.run(inputIds);
    output.pooledOutput.free();
    return output.lastHiddenState;
  }

  run(inputIds: MxArray, options: CLIPTextModelOptions = {}): CLIPTextModelOutput {
    this.assertInputIds(inputIds, "CLIPTextModel.run");
    let hidden = this.embeddings.forward(inputIds);
    const hiddenStates = options.outputHiddenStates ? [retainArray(hidden)] : undefined;
    let lastHiddenState: MxArray | null = null;

    try {
      for (const layer of this.layers) {
        const nextHidden = layer.run(hidden);
        hidden.free();
        hidden = nextHidden;
        if (hiddenStates !== undefined) {
          hiddenStates.push(retainArray(hidden));
        }
      }

      lastHiddenState = this.finalLayerNorm.forward(hidden);
      const pooledOutput = this.pool(lastHiddenState, inputIds);
      const output: CLIPTextModelOutput = {
        lastHiddenState,
        pooledOutput,
      };
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

  private assertInputIds(inputIds: MxArray, context: string): void {
    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(`${context}: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`);
    }
  }

  private pool(lastHiddenState: MxArray, inputIds: MxArray): MxArray {
    const [batch, sequenceLength, hiddenSize] = lastHiddenState.shape;
    if (
      batch === undefined ||
      sequenceLength === undefined ||
      hiddenSize !== this.#hiddenSize ||
      lastHiddenState.shape.length !== 3
    ) {
      throw new Error(
        `CLIPTextModel.pool: expected [batch, seq, ${this.#hiddenSize}], got ${formatShape(lastHiddenState.shape)}.`,
      );
    }

    using eosPositions = this.eosPositions(inputIds);
    using positionColumn = expandDims(eosPositions, 1);
    using positionCube = expandDims(positionColumn, 2);
    using gatherIndices = broadcastTo(positionCube, [batch, 1, this.#hiddenSize]);
    using gathered = takeAlongAxis(lastHiddenState, gatherIndices, 1);
    return squeeze(gathered, 1);
  }

  private eosPositions(inputIds: MxArray): MxArray {
    if (this.#eosTokenId === null || this.#eosTokenId === 2) {
      return argmax(inputIds, -1);
    }
    using eosMask = equal(inputIds, this.#eosTokenId);
    using eosMaskInt = asType(eosMask, "int32");
    return argmax(eosMaskInt, -1);
  }
}

/** CLIP text transformer encoder with the pooled projection head. */
export class CLIPTextModelWithProjection extends Module {
  textModel: CLIPTextModel;
  textProjection: Linear;

  constructor(config: CLIPTextConfig) {
    super();
    if (config.projectionDim === null) {
      throw new Error("CLIPTextModelWithProjection: projection_dim is required.");
    }
    this.textModel = new CLIPTextModel(config);
    this.textProjection = new Linear(config.hiddenSize, config.projectionDim, false);
  }

  forward(inputIds: MxArray): MxArray {
    const output = this.run(inputIds);
    output.lastHiddenState.free();
    output.pooledOutput.free();
    disposeHiddenStates(output.hiddenStates);
    return output.textEmbeds;
  }

  run(inputIds: MxArray, options: CLIPTextModelOptions = {}): CLIPTextProjectionOutput {
    const textOutput = this.textModel.run(inputIds, options);
    let textEmbeds: MxArray | null = null;
    try {
      textEmbeds = this.textProjection.forward(textOutput.pooledOutput);
      return {
        ...textOutput,
        textEmbeds,
      };
    } catch (error) {
      textEmbeds?.free();
      disposeCLIPTextModelOutput(textOutput);
      throw error;
    }
  }
}
