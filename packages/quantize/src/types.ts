import type { QuantizationMode } from "@mlxts/core";
import type { Embedding, Linear, Module, QuantizedEmbedding, QuantizedLinear } from "@mlxts/nn";

/** Fully resolved quantization parameters for one target module. */
export type QuantizationParameters = {
  groupSize: number;
  bits: number;
  mode: QuantizationMode;
};

/** Partial quantization parameters used by call sites and checkpoint metadata. */
export type QuantizationParameterOverrides = {
  groupSize?: number;
  bits?: number;
  mode?: QuantizationMode;
};

/** Selection result for live module quantization. */
export type QuantizeSelectionResult = boolean | QuantizationParameterOverrides;

/** Options for quantizing dense linear and embedding modules in place. */
export type QuantizeModuleOptions = QuantizationParameterOverrides & {
  select?: (path: string, layer: Embedding | Linear) => QuantizeSelectionResult;
};

/** One module that was quantized or prepared for quantized loading. */
export type QuantizedModuleTarget = {
  path: string;
  params: QuantizationParameters;
};

/** One module that was intentionally left dense. */
export type QuantizedModuleSkip = {
  path: string;
  reason: string;
};

/** Result of a module-tree quantization transform. */
export type QuantizedModuleResult = {
  targets: QuantizedModuleTarget[];
  skipped: QuantizedModuleSkip[];
};

/** One explicit checkpoint quantization rule. */
export type QuantizedCheckpointRule = {
  path: string;
  enabled: boolean;
  params?: QuantizationParameterOverrides;
};

/** Parsed quantization metadata from a pretrained config payload. */
export type QuantizedCheckpointPlan = {
  provider: string;
  sourceKey: "quantization" | "quantization_config";
  defaults: QuantizationParameters;
  explicitOnly: boolean;
  rules: QuantizedCheckpointRule[];
};

/** Provider hook for parsing quantization metadata from model configs. */
export interface QuantizedCheckpointProvider {
  readonly name: string;
  resolve(config: Record<string, unknown>): QuantizedCheckpointPlan | null;
}

/** One direct child module slot discovered during traversal. */
export type ModuleChildSlot = {
  path: string;
  parent: Module;
  key: string;
  child: Module;
};

/** One direct child linear slot discovered during traversal. */
export type LinearChildSlot = {
  path: string;
  parent: Module;
  key: string;
  child: Linear | QuantizedLinear;
};

/** One direct child embedding slot discovered during traversal. */
export type EmbeddingChildSlot = {
  path: string;
  parent: Module;
  key: string;
  child: Embedding | QuantizedEmbedding;
};
