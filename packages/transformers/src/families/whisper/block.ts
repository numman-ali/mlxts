/**
 * Whisper transformer blocks.
 * @module
 */

import { add, type MxArray } from "@mlxts/core";
import { LayerNorm, Module } from "@mlxts/nn";

import { WhisperAttention } from "./attention";
import { WhisperMLP } from "./mlp";
import type { WhisperConfig } from "./types";

/** Transformer encoder block used by Whisper audio encoders. */
export class WhisperEncoderBlock extends Module {
  selfAttention: WhisperAttention;
  selfAttentionLayerNorm: LayerNorm;
  mlp: WhisperMLP;
  finalLayerNorm: LayerNorm;

  constructor(config: WhisperConfig) {
    super();
    this.selfAttention = WhisperAttention.encoder(config);
    this.selfAttentionLayerNorm = new LayerNorm(config.dModel);
    this.mlp = WhisperMLP.encoder(config);
    this.finalLayerNorm = new LayerNorm(config.dModel);
  }

  forward(hiddenStates: MxArray): MxArray {
    return this.run(hiddenStates);
  }

  run(hiddenStates: MxArray): MxArray {
    using normalizedAttentionInput = this.selfAttentionLayerNorm.forward(hiddenStates);
    using attentionOutput = this.selfAttention.run(normalizedAttentionInput);
    const attended = add(hiddenStates, attentionOutput);
    using residual = attended;
    using normalizedMlpInput = this.finalLayerNorm.forward(residual);
    using mlpOutput = this.mlp.forward(normalizedMlpInput);
    return add(residual, mlpOutput);
  }
}

/** Transformer decoder block used by Whisper text decoders. */
export class WhisperDecoderBlock extends Module {
  selfAttention: WhisperAttention;
  selfAttentionLayerNorm: LayerNorm;
  crossAttention: WhisperAttention;
  crossAttentionLayerNorm: LayerNorm;
  mlp: WhisperMLP;
  finalLayerNorm: LayerNorm;

  constructor(config: WhisperConfig) {
    super();
    this.selfAttention = WhisperAttention.decoder(config);
    this.selfAttentionLayerNorm = new LayerNorm(config.dModel);
    this.crossAttention = WhisperAttention.decoder(config);
    this.crossAttentionLayerNorm = new LayerNorm(config.dModel);
    this.mlp = WhisperMLP.decoder(config);
    this.finalLayerNorm = new LayerNorm(config.dModel);
  }

  forward(hiddenStates: MxArray, encoderHiddenStates: MxArray): MxArray {
    return this.run(hiddenStates, encoderHiddenStates);
  }

  run(hiddenStates: MxArray, encoderHiddenStates: MxArray): MxArray {
    using normalizedSelfInput = this.selfAttentionLayerNorm.forward(hiddenStates);
    using selfAttentionOutput = this.selfAttention.run(normalizedSelfInput, { causal: true });
    const selfAttended = add(hiddenStates, selfAttentionOutput);
    using selfResidual = selfAttended;
    using normalizedCrossInput = this.crossAttentionLayerNorm.forward(selfResidual);
    using crossAttentionOutput = this.crossAttention.run(normalizedCrossInput, {
      keyValueStates: encoderHiddenStates,
    });
    const crossAttended = add(selfResidual, crossAttentionOutput);
    using crossResidual = crossAttended;
    using normalizedMlpInput = this.finalLayerNorm.forward(crossResidual);
    using mlpOutput = this.mlp.forward(normalizedMlpInput);
    return add(crossResidual, mlpOutput);
  }
}
