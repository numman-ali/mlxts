/**
 * Qwen 3.5 dense and MoE feed-forward config parsing.
 * @module
 */

import {
  expectInteger,
  optionalInteger,
  optionalNumber,
} from "../../infrastructure/config-parsing";
import { ConfigParseError } from "../../types";
import { expectPositiveInteger } from "./config-helpers";
import type { Qwen3_5TextModelType } from "./types";

function requirePositiveOptionalInteger(
  config: Record<string, unknown>,
  key: string,
  context: string,
): number {
  return expectPositiveInteger(expectInteger(config, key, context), `${context}.${key}`);
}

/**
 * Parse the dense or MoE feed-forward fields for a Qwen text config.
 */
export function parseQwen3_5FeedForward(
  config: Record<string, unknown>,
  context: string,
  modelType: Qwen3_5TextModelType,
): {
  feedForwardKind: "dense" | "moe";
  intermediateSize: number;
  moeIntermediateSize: number | null;
  sharedExpertIntermediateSize: number | null;
  numExperts: number | null;
  numExpertsPerToken: number | null;
  routerAuxLossCoef: number | null;
} {
  if (modelType === "qwen3_5_moe_text") {
    const sharedExpertIntermediateSize = expectPositiveInteger(
      optionalInteger(config, "shared_expert_intermediate_size", context) ?? 512,
      `${context}.shared_expert_intermediate_size`,
    );
    const numExperts = expectPositiveInteger(
      optionalInteger(config, "num_experts", context) ?? 256,
      `${context}.num_experts`,
    );
    const numExpertsPerToken = expectPositiveInteger(
      optionalInteger(config, "num_experts_per_tok", context) ?? 8,
      `${context}.num_experts_per_tok`,
    );
    if (numExpertsPerToken > numExperts) {
      throw new ConfigParseError(
        `${context}.num_experts_per_tok must be <= num_experts (${numExperts}), got ${numExpertsPerToken}.`,
      );
    }
    return {
      feedForwardKind: "moe",
      intermediateSize: expectPositiveInteger(
        optionalInteger(config, "intermediate_size", context) ?? sharedExpertIntermediateSize,
        `${context}.intermediate_size`,
      ),
      moeIntermediateSize: expectPositiveInteger(
        optionalInteger(config, "moe_intermediate_size", context) ?? 512,
        `${context}.moe_intermediate_size`,
      ),
      sharedExpertIntermediateSize,
      numExperts,
      numExpertsPerToken,
      routerAuxLossCoef: optionalNumber(config, "router_aux_loss_coef", context) ?? 0.001,
    };
  }

  return {
    feedForwardKind: "dense",
    intermediateSize: requirePositiveOptionalInteger(config, "intermediate_size", context),
    moeIntermediateSize: null,
    sharedExpertIntermediateSize: null,
    numExperts: null,
    numExpertsPerToken: null,
    routerAuxLossCoef: null,
  };
}
