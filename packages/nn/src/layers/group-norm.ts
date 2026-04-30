/**
 * Group normalization over channel-last tensors.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  add,
  divide,
  formatShape,
  mean,
  multiply,
  ones,
  reshape,
  sqrt,
  square,
  subtract,
  zeros,
} from "@mlxts/core";
import { Module } from "../module";

function normalizationAxes(groupedRank: number): number[] {
  const groupAxis = groupedRank - 2;
  return Array.from({ length: groupedRank - 1 }, (_, index) => index + 1).filter(
    (axis) => axis !== groupAxis,
  );
}

/** Group normalization for channel-last tensors. */
export class GroupNorm extends Module {
  weight: MxArray;
  bias: MxArray;
  #groups: number;
  #channels: number;
  #eps: number;

  /**
   * @param groups - Number of independent channel groups.
   * @param channels - Number of channels in the last input dimension.
   * @param eps - Small constant for numerical stability. Defaults to 1e-5.
   */
  constructor(groups: number, channels: number, eps = 1e-5) {
    super();
    if (groups <= 0) {
      throw new Error(`GroupNorm: groups must be > 0, got ${groups}`);
    }
    if (channels <= 0) {
      throw new Error(`GroupNorm: channels must be > 0, got ${channels}`);
    }
    if (channels % groups !== 0) {
      throw new Error("GroupNorm: groups must divide channels.");
    }
    if (eps <= 0) {
      throw new Error(`GroupNorm: eps must be > 0, got ${eps}`);
    }
    this.#groups = groups;
    this.#channels = channels;
    this.#eps = eps;
    this.weight = ones([channels]);
    this.bias = zeros([channels]);
  }

  get groups(): number {
    return this.#groups;
  }

  get channels(): number {
    return this.#channels;
  }

  forward(x: MxArray): MxArray {
    const channels = x.shape[x.shape.length - 1];
    if (x.shape.length < 2 || channels === undefined) {
      throw new Error(`GroupNorm.forward: expected rank >= 2 input, got ${formatShape(x.shape)}.`);
    }
    if (channels !== this.#channels) {
      throw new Error(
        `GroupNorm.forward: expected last dimension ${this.#channels}, got ${channels}.`,
      );
    }
    this.#validateParameters();

    const channelsPerGroup = channels / this.#groups;
    const groupedShape = [...x.shape.slice(0, -1), this.#groups, channelsPerGroup];
    const axes = normalizationAxes(groupedShape.length);
    using grouped = reshape(x, groupedShape);
    using groupMean = mean(grouped, axes, true);
    using centered = subtract(grouped, groupMean);
    using squared = square(centered);
    using variance = mean(squared, axes, true);
    using stabilized = add(variance, this.#eps);
    using denominator = sqrt(stabilized);
    using normalizedGrouped = divide(centered, denominator);
    using normalized = reshape(normalizedGrouped, [...x.shape]);
    using weighted = multiply(normalized, this.weight);
    return add(weighted, this.bias);
  }

  #validateParameters(): void {
    if (this.weight.shape.length !== 1 || this.weight.shape[0] !== this.#channels) {
      throw new Error(
        `GroupNorm.forward: expected weight shape [${this.#channels}], got ${formatShape(this.weight.shape)}.`,
      );
    }
    if (this.bias.shape.length !== 1 || this.bias.shape[0] !== this.#channels) {
      throw new Error(
        `GroupNorm.forward: expected bias shape [${this.#channels}], got ${formatShape(this.bias.shape)}.`,
      );
    }
  }
}
