/**
 * LLaMA-style decoder MLP blocks.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { split } from "@mlxts/core";
import { Linear, Module, swiglu } from "@mlxts/nn";

import { gegluApprox } from "../../infrastructure/gated-activations";
import type { LlamaLikeConfig } from "./types";

/** Shared gated MLP used by the supported LLaMA-like families. */
export class LlamaLikeMLP extends Module {
  gateProjection: Linear | null;
  upProjection: Linear | null;
  gateUpProjection: Linear | null;
  downProjection: Linear;
  #activation: LlamaLikeConfig["mlpActivation"];
  #intermediateSize: number;

  constructor(config: LlamaLikeConfig) {
    super();
    this.#intermediateSize = config.intermediateSize;
    if (config.mlpProjectionLayout === "packed_gate_up") {
      this.gateProjection = null;
      this.upProjection = null;
      this.gateUpProjection = new Linear(config.hiddenSize, config.intermediateSize * 2, false);
    } else {
      this.gateProjection = new Linear(config.hiddenSize, config.intermediateSize, false);
      this.upProjection = new Linear(config.hiddenSize, config.intermediateSize, false);
      this.gateUpProjection = null;
    }
    this.downProjection = new Linear(config.intermediateSize, config.hiddenSize, false);
    this.#activation = config.mlpActivation;
  }

  forward(x: MxArray): MxArray {
    const projected = this.projectGateAndValue(x);

    try {
      using gate = projected.gate;
      using value = projected.value;
      if (this.#activation === "swiglu") {
        using activated = swiglu(gate, value);
        return this.downProjection.forward(activated);
      }
      using activated = gegluApprox(gate, value);
      return this.downProjection.forward(activated);
    } finally {
      projected.gate.free();
      projected.value.free();
    }
  }

  private projectGateAndValue(x: MxArray): { gate: MxArray; value: MxArray } {
    if (
      this.gateProjection !== null &&
      this.upProjection !== null &&
      this.gateUpProjection === null
    ) {
      return {
        gate: this.gateProjection.forward(x),
        value: this.upProjection.forward(x),
      };
    }

    if (this.gateUpProjection === null) {
      throw new Error("LlamaLikeMLP: expected either split or packed gate/value projections.");
    }

    using packed = this.gateUpProjection.forward(x);
    const parts = split(packed, 2, packed.shape.length - 1);
    const gate = parts[0];
    const value = parts[1];
    if (gate === undefined || value === undefined || parts.length !== 2) {
      for (const part of parts) {
        part.free();
      }
      throw new Error(
        `LlamaLikeMLP: expected packed gate_up projection to split into 2 tensors of width ${this.#intermediateSize}.`,
      );
    }
    return { gate, value };
  }
}
