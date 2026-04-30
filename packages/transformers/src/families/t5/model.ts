/**
 * T5 encoder model.
 * @module
 */

import { formatShape, isIntegerDType, type MxArray, retainArray } from "@mlxts/core";
import { Embedding, Module, RMSNorm } from "@mlxts/nn";

import { T5EncoderBlock } from "./block";
import type { T5EncoderConfig, T5EncoderModelOptions, T5EncoderModelOutput } from "./types";

function disposeHiddenStates(hiddenStates: MxArray[] | undefined): void {
  if (hiddenStates === undefined) {
    return;
  }
  for (const hiddenState of hiddenStates) {
    hiddenState.free();
  }
}

/** Dispose arrays returned by `T5EncoderModel.run`. */
export function disposeT5EncoderModelOutput(output: T5EncoderModelOutput): void {
  output.lastHiddenState.free();
  disposeHiddenStates(output.hiddenStates);
}

/** Encoder-only T5 model for text conditioning and encoder-decoder proof paths. */
export class T5EncoderModel extends Module {
  tokenEmbedding: Embedding;
  layers: T5EncoderBlock[];
  finalLayerNorm: RMSNorm;
  #dModel: number;

  constructor(config: T5EncoderConfig) {
    super();
    this.tokenEmbedding = new Embedding(config.vocabSize, config.dModel);
    this.layers = Array.from(
      { length: config.numLayers },
      (_, index) => new T5EncoderBlock(config, index === 0),
    );
    this.finalLayerNorm = new RMSNorm(config.dModel, config.layerNormEps);
    this.#dModel = config.dModel;
  }

  forward(inputIds: MxArray): MxArray {
    const output = this.run(inputIds);
    disposeHiddenStates(output.hiddenStates);
    return output.lastHiddenState;
  }

  run(inputIds: MxArray, options: T5EncoderModelOptions = {}): T5EncoderModelOutput {
    this.assertInputIds(inputIds, "T5EncoderModel.run");
    let hidden = this.tokenEmbedding.forward(inputIds);
    const hiddenStates: MxArray[] | undefined = options.outputHiddenStates ? [] : undefined;
    let lastHiddenState: MxArray | null = null;
    const sequenceLength = inputIds.shape[1];
    if (sequenceLength === undefined) {
      hidden.free();
      throw new Error("T5EncoderModel.run: input ids are missing a sequence axis.");
    }

    const positionBias =
      this.layers[0] === undefined
        ? null
        : this.layers[0].selfAttention.attention.positionBias(sequenceLength, sequenceLength);

    try {
      for (const layer of this.layers) {
        if (hiddenStates !== undefined) {
          hiddenStates.push(retainArray(hidden));
        }
        if (positionBias === null) {
          throw new Error(
            "T5EncoderModel.run: position bias is required when encoder layers exist.",
          );
        }
        const nextHidden = layer.run(hidden, positionBias);
        hidden.free();
        hidden = nextHidden;
      }

      lastHiddenState = this.finalLayerNorm.forward(hidden);
      if (hiddenStates !== undefined) {
        hiddenStates.push(retainArray(lastHiddenState));
      }
      const output: T5EncoderModelOutput = { lastHiddenState };
      if (hiddenStates !== undefined) {
        output.hiddenStates = hiddenStates;
      }
      return output;
    } catch (error) {
      lastHiddenState?.free();
      disposeHiddenStates(hiddenStates);
      throw error;
    } finally {
      positionBias?.free();
      hidden.free();
    }
  }

  private assertInputIds(inputIds: MxArray, context: string): void {
    const [batch, sequenceLength] = inputIds.shape;
    if (batch === undefined || sequenceLength === undefined || inputIds.shape.length !== 2) {
      throw new Error(`${context}: expected rank-2 token ids, got ${formatShape(inputIds.shape)}.`);
    }
    if (!isIntegerDType(inputIds.dtype)) {
      throw new Error(`${context}: expected integer token ids, got dtype ${inputIds.dtype}.`);
    }
  }
}
