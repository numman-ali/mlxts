/**
 * Whisper feed-forward layers.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { gelu, Linear, Module } from "@mlxts/nn";

import type { WhisperConfig } from "./types";

/** Feed-forward network used by Whisper transformer blocks. */
export class WhisperMLP extends Module {
  fc1: Linear;
  fc2: Linear;

  constructor(inputDims: number, intermediateDims: number) {
    super();
    this.fc1 = new Linear(inputDims, intermediateDims, true);
    this.fc2 = new Linear(intermediateDims, inputDims, true);
  }

  static encoder(config: WhisperConfig): WhisperMLP {
    return new WhisperMLP(config.dModel, config.encoderFfnDim);
  }

  static decoder(config: WhisperConfig): WhisperMLP {
    return new WhisperMLP(config.dModel, config.decoderFfnDim);
  }

  forward(hiddenStates: MxArray): MxArray {
    using projected = this.fc1.forward(hiddenStates);
    using activated = gelu(projected);
    return this.fc2.forward(activated);
  }
}
