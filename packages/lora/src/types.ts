import type { FlatEntry } from "@mlxts/core";
import type { Linear, LoRALinear, Module, QuantizedLinear } from "@mlxts/nn";

/** Shared LoRA adapter hyperparameters. */
export type LoRAConfig = {
  rank?: number;
  alpha?: number;
  dropout?: number;
};

/** Options for applying LoRA wrappers across a module tree. */
export type ApplyLoRAOptions = LoRAConfig & {
  lastLayers?: number;
  keys?: readonly string[];
  paths?: readonly string[];
  select?: (path: string, layer: Linear | QuantizedLinear) => boolean;
};

/** One direct child module slot discovered during traversal. */
export type ModuleChildSlot = {
  path: string;
  parent: Module;
  key: string;
  child: Module;
};

/** One LoRA-targetable linear slot. */
export type LoRATargetSlot = {
  path: string;
  parent: Module;
  key: string;
  child: Linear | QuantizedLinear;
};

/** One existing LoRA wrapper slot. */
export type LoRAWrapperSlot = {
  path: string;
  parent: Module;
  key: string;
  child: LoRALinear;
};

/** Result of applying, merging, or removing LoRA wrappers. */
export type LoRAModuleResult = {
  targets: string[];
  skipped: Array<{ path: string; reason: string }>;
};

/** One saved LoRA adapter target. */
export type LoRAAdapterTarget = {
  path: string;
  rank: number;
  alpha: number;
  dropout: number;
};

/** Serialized LoRA adapter configuration file. */
export type LoRAAdapterConfig = {
  format: "mlxts-lora";
  version: 1;
  targets: LoRAAdapterTarget[];
};

/** Saved LoRA adapter tensors keyed by module path. */
export type LoRAAdapterTensorEntry = FlatEntry;
