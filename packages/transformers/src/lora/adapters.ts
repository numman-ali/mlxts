import {
  type FlatEntry,
  loadSafetensors,
  type MxArray,
  saveSafetensors,
  treeUnflatten,
} from "@mlxts/core";
import type { LoRAConfig } from "@mlxts/lora";
import { applyLoRAToModule, loadLoRAAdapters, saveLoRAAdapters } from "@mlxts/lora";
import { mkdirSync } from "fs";

import { resolveFamily } from "../registry";
import type { CausalLM } from "../types";
import { collectLoRAWrapperStates, expectCausalLMModule } from "./module-traversal";

/** Supported on-disk adapter formats for transformer-owned causal LM adapter I/O. */
export type CausalLMAdapterFormat = "mlxts" | "peft";

/** Options for saving active causal LM LoRA adapters. */
export type SaveCausalLMAdaptersOptions = {
  format?: CausalLMAdapterFormat;
  baseModelNameOrPath?: string;
};

/** Options for loading causal LM LoRA adapters with optional format auto-detection. */
export type LoadCausalLMAdaptersOptions = {
  format?: "auto" | CausalLMAdapterFormat;
};

type SavedTarget = {
  path: string;
  rank: number;
  alpha: number;
  dropout: number;
};

type PeftLoRAConfig = {
  peft_type: "LORA";
  task_type: "CAUSAL_LM";
  base_model_name_or_path: string | null;
  revision: string | null;
  inference_mode: boolean;
  r: number;
  lora_alpha: number;
  lora_dropout: number;
  bias: "none";
  fan_in_fan_out: boolean;
  use_rslora: boolean;
  target_modules: string[];
};

const configPath = (directory: string): string => `${directory}/adapter_config.json`;
const peftWeightsPath = (directory: string): string => `${directory}/adapter_model.safetensors`;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function tensorEntries(tensors: Record<string, MxArray>): FlatEntry[] {
  return Object.entries(tensors).map(([path, tensor]) => [path.split("."), tensor]);
}

function wrapperTargets(model: CausalLM): SavedTarget[] {
  return collectLoRAWrapperStates(model).map((slot) => ({
    path: slot.path,
    rank: slot.child.rank,
    alpha: slot.child.alpha,
    dropout: slot.child.dropoutProbability,
  }));
}

function uniformLoRAConfig(targets: readonly SavedTarget[]): LoRAConfig {
  const first = targets[0];
  if (first === undefined) {
    throw new Error("transformers: found no active LoRA wrappers.");
  }

  for (const target of targets) {
    if (
      target.rank !== first.rank ||
      target.alpha !== first.alpha ||
      Math.abs(target.dropout - first.dropout) > 1e-9
    ) {
      throw new Error(
        "transformers.saveCausalLMAdapters: PEFT export currently requires one shared rank, alpha, and dropout across active adapters.",
      );
    }
  }

  return {
    rank: first.rank,
    alpha: first.alpha,
    dropout: first.dropout,
  };
}

function checkpointPrefix(modelType: string): string {
  if (modelType === "mistral3") {
    return "language_model.";
  }

  if (modelType === "gemma4") {
    return "model.language_model.";
  }

  return "";
}

function checkpointModulePath(path: string): string {
  if (path === "lmHead") {
    return "lm_head";
  }

  if (path === "model.perLayerModelProjection") {
    return "model.per_layer_model_projection";
  }

  return path
    .replace(/^model\.layers\.(\d+)\.selfAttention\./, "model.layers.$1.self_attn.")
    .replace(/^model\.layers\.(\d+)\.mlp\./, "model.layers.$1.mlp.")
    .replace(/^model\.layers\.(\d+)\.perLayerInputGate$/, "model.layers.$1.per_layer_input_gate")
    .replace(/^model\.layers\.(\d+)\.perLayerProjection$/, "model.layers.$1.per_layer_projection")
    .replace(/^model\.embedTokens$/, "model.embed_tokens")
    .replace(/^model\.embedTokensPerLayer$/, "model.embed_tokens_per_layer")
    .replace(/qProjection$/, "q_proj")
    .replace(/kProjection$/, "k_proj")
    .replace(/vProjection$/, "v_proj")
    .replace(/outputProjection$/, "o_proj")
    .replace(/qkvProjection$/, "qkv_proj")
    .replace(/gateProjection$/, "gate_proj")
    .replace(/upProjection$/, "up_proj")
    .replace(/downProjection$/, "down_proj")
    .replace(/gateUpProjection$/, "gate_up_proj");
}

function peftTensorPrefix(model: CausalLM, path: string): string {
  return `base_model.model.${checkpointPrefix(model.config.modelType)}${checkpointModulePath(path)}`;
}

function readPeftTargetModules(targets: readonly SavedTarget[]): string[] {
  return [
    ...new Set(
      targets
        .map((target) => checkpointModulePath(target.path).split(".").at(-1))
        .filter((value): value is string => value !== undefined),
    ),
  ];
}

function peftConfig(
  targets: readonly SavedTarget[],
  options: SaveCausalLMAdaptersOptions,
): PeftLoRAConfig {
  const config = uniformLoRAConfig(targets);
  return {
    peft_type: "LORA",
    task_type: "CAUSAL_LM",
    base_model_name_or_path: options.baseModelNameOrPath ?? null,
    revision: null,
    inference_mode: true,
    r: config.rank ?? 8,
    lora_alpha: config.alpha ?? config.rank ?? 8,
    lora_dropout: config.dropout ?? 0,
    bias: "none",
    fan_in_fan_out: false,
    use_rslora: false,
    target_modules: readPeftTargetModules(targets),
  };
}

function peftTensorRecord(model: CausalLM): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const slot of collectLoRAWrapperStates(model)) {
    const prefix = peftTensorPrefix(model, slot.path);
    tensors[`${prefix}.lora_A.weight`] = slot.child.loraA;
    tensors[`${prefix}.lora_B.weight`] = slot.child.loraB;
  }
  return tensors;
}

const ALLOWED_PEFT_CONFIG_KEYS = new Set([
  "peft_type",
  "task_type",
  "base_model_name_or_path",
  "revision",
  "inference_mode",
  "r",
  "lora_alpha",
  "lora_dropout",
  "bias",
  "fan_in_fan_out",
  "use_rslora",
  "target_modules",
  "modules_to_save",
  "rank_pattern",
  "alpha_pattern",
  "use_dora",
  "alora_invocation_tokens",
  "lora_bias",
  "target_parameters",
  "layer_replication",
  "trainable_token_indices",
  "runtime_config",
  "auto_mapping",
]);

function assertAllowedPeftKeys(payload: Record<string, unknown>): void {
  for (const key of Object.keys(payload)) {
    if (!ALLOWED_PEFT_CONFIG_KEYS.has(key)) {
      throw new Error(
        `transformers.loadCausalLMAdapters: unsupported PEFT config key "${key}" in first-pass causal LM LoRA interop.`,
      );
    }
  }
}

function assertPeftTask(payload: Record<string, unknown>): void {
  if (payload.peft_type !== "LORA") {
    throw new Error(
      'transformers.loadCausalLMAdapters: PEFT adapter config must set peft_type to "LORA".',
    );
  }
  if (payload.task_type !== "CAUSAL_LM") {
    throw new Error(
      'transformers.loadCausalLMAdapters: PEFT adapter config must set task_type to "CAUSAL_LM".',
    );
  }
}

function assertSupportedPeftFeatureSubset(payload: Record<string, unknown>): void {
  if (payload.modules_to_save !== undefined) {
    throw new Error(
      "transformers.loadCausalLMAdapters: PEFT modules_to_save is not supported in this first-pass causal LM adapter loader.",
    );
  }
  if (payload.trainable_token_indices !== undefined) {
    throw new Error(
      "transformers.loadCausalLMAdapters: PEFT trainable_token_indices is not supported in this first-pass causal LM adapter loader.",
    );
  }
  if (payload.rank_pattern !== undefined || payload.alpha_pattern !== undefined) {
    throw new Error(
      "transformers.loadCausalLMAdapters: PEFT rank_pattern and alpha_pattern are not supported in this first-pass causal LM adapter loader.",
    );
  }
  if (
    payload.use_dora === true ||
    payload.alora_invocation_tokens !== undefined ||
    payload.lora_bias === true ||
    payload.target_parameters !== undefined ||
    payload.layer_replication !== undefined
  ) {
    throw new Error(
      "transformers.loadCausalLMAdapters: this first-pass causal LM adapter loader supports standard LoRA only.",
    );
  }
}

function readPositiveIntegerField(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(`transformers.loadCausalLMAdapters: PEFT ${name} must be a positive integer.`);
  }
  return value;
}

function readFiniteNumberField(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`transformers.loadCausalLMAdapters: PEFT ${name} must be a finite number.`);
  }
  return value;
}

function readDropoutField(value: unknown): number {
  if (typeof value !== "number" || value < 0 || value >= 1) {
    throw new Error(
      "transformers.loadCausalLMAdapters: PEFT lora_dropout must satisfy 0 <= p < 1.",
    );
  }
  return value;
}

function readTargetModulesField(value: unknown): string[] {
  if (typeof value === "string") {
    return [value];
  }
  if (Array.isArray(value) && value.every((entry) => typeof entry === "string")) {
    return value;
  }
  throw new Error(
    "transformers.loadCausalLMAdapters: PEFT target_modules must be a string or an array of strings.",
  );
}

const readNullableStringField = (value: unknown): string | null =>
  value === null || typeof value === "string" ? value : null;

function parsePeftConfig(payload: unknown): PeftLoRAConfig {
  if (!isRecord(payload)) {
    throw new Error(
      "transformers.loadCausalLMAdapters: PEFT adapter config must be a JSON object.",
    );
  }

  assertAllowedPeftKeys(payload);
  assertPeftTask(payload);
  assertSupportedPeftFeatureSubset(payload);

  if (payload.bias !== "none") {
    throw new Error('transformers.loadCausalLMAdapters: only PEFT bias="none" is supported.');
  }
  if (payload.fan_in_fan_out === true) {
    throw new Error(
      "transformers.loadCausalLMAdapters: PEFT fan_in_fan_out=true is not supported in this first-pass causal LM adapter loader.",
    );
  }
  if (payload.use_rslora === true) {
    throw new Error(
      "transformers.loadCausalLMAdapters: PEFT use_rslora=true is not supported yet by mlxts LoRA scaling.",
    );
  }

  return {
    peft_type: "LORA",
    task_type: "CAUSAL_LM",
    base_model_name_or_path: readNullableStringField(payload.base_model_name_or_path),
    revision: readNullableStringField(payload.revision),
    inference_mode: payload.inference_mode === true,
    r: readPositiveIntegerField(payload.r, "r"),
    lora_alpha: readFiniteNumberField(payload.lora_alpha, "lora_alpha"),
    lora_dropout: readDropoutField(payload.lora_dropout),
    bias: "none",
    fan_in_fan_out: false,
    use_rslora: false,
    target_modules: readTargetModulesField(payload.target_modules),
  };
}

function detectAdapterFormat(payload: unknown): CausalLMAdapterFormat {
  if (!isRecord(payload)) {
    throw new Error("transformers.loadCausalLMAdapters: adapter config must be a JSON object.");
  }

  if (payload.format === "mlxts-lora") {
    return "mlxts";
  }

  if (payload.peft_type === "LORA") {
    return "peft";
  }

  throw new Error(
    'transformers.loadCausalLMAdapters: could not detect adapter format. Expected mlxts "format" or PEFT "peft_type".',
  );
}

function translatedPeftPath(model: CausalLM, peftTensorName: string): string | null {
  const match = peftTensorName.match(/^(?:base_model\.model\.)(.+)\.(lora_A|lora_B)\.weight$/);
  if (match === null) {
    return null;
  }

  const checkpointModule = match[1];
  const tensorKind = match[2];
  if (checkpointModule === undefined || tensorKind === undefined) {
    return null;
  }

  const checkpointWeightName = `${checkpointModule}.weight`;
  const registration = resolveFamily(model.config.modelType);
  const internalWeightPath = registration.sanitizeWeight(model.config, checkpointWeightName);
  if (internalWeightPath === null || !internalWeightPath.endsWith(".weight")) {
    return null;
  }

  const modulePath = internalWeightPath.slice(0, -".weight".length);
  return `${modulePath}.${tensorKind === "lora_A" ? "loraA" : "loraB"}`;
}

function expectedPeftTensorNames(model: CausalLM): Set<string> {
  return new Set(
    collectLoRAWrapperStates(model).flatMap((slot) => {
      const prefix = peftTensorPrefix(model, slot.path);
      return [`${prefix}.lora_A.weight`, `${prefix}.lora_B.weight`];
    }),
  );
}

async function loadPeftAdapters(model: CausalLM, directory: string): Promise<void> {
  const config = parsePeftConfig(JSON.parse(await Bun.file(configPath(directory)).text()));
  const loaded = await loadSafetensors(peftWeightsPath(directory));
  const translatedTensors: Record<string, MxArray> = {};

  try {
    for (const [name, tensor] of Object.entries(loaded.tensors)) {
      const translated = translatedPeftPath(model, name);
      if (translated === null) {
        throw new Error(
          `transformers.loadCausalLMAdapters: unexpected PEFT adapter tensor "${name}".`,
        );
      }
      if (translated in translatedTensors) {
        throw new Error(
          `transformers.loadCausalLMAdapters: duplicate translated PEFT adapter tensor "${translated}".`,
        );
      }
      translatedTensors[translated] = tensor;
    }

    const loraPaths = [
      ...new Set(
        [...Object.keys(translatedTensors)].map((name) => name.replace(/\.(loraA|loraB)$/, "")),
      ),
    ];
    for (const path of loraPaths) {
      applyLoRAToModule(expectCausalLMModule(model), {
        paths: [path],
        rank: config.r,
        alpha: config.lora_alpha,
        dropout: config.lora_dropout,
      });
    }

    const expectedNames = expectedPeftTensorNames(model);
    for (const name of Object.keys(loaded.tensors)) {
      if (!expectedNames.has(name)) {
        throw new Error(
          `transformers.loadCausalLMAdapters: unexpected PEFT adapter tensor "${name}".`,
        );
      }
    }
    for (const name of expectedNames) {
      if (!(name in loaded.tensors)) {
        throw new Error(
          `transformers.loadCausalLMAdapters: missing PEFT adapter tensor "${name}".`,
        );
      }
    }

    expectCausalLMModule(model).update(treeUnflatten(tensorEntries(translatedTensors)));
  } catch (error) {
    for (const tensor of Object.values(loaded.tensors)) {
      tensor.free();
    }
    throw error;
  }
}

/** Save active LoRA adapters for a loaded decoder model. */
export async function saveCausalLMAdapters(
  model: CausalLM,
  directory: string,
  options: SaveCausalLMAdaptersOptions = {},
): Promise<void> {
  const format = options.format ?? "mlxts";
  if (format === "mlxts") {
    await saveLoRAAdapters(expectCausalLMModule(model), directory);
    return;
  }

  const targets = wrapperTargets(model);
  if (targets.length === 0) {
    throw new Error("transformers.saveCausalLMAdapters: found no active LoRA wrappers.");
  }

  mkdirSync(directory, { recursive: true });
  await Bun.write(
    configPath(directory),
    `${JSON.stringify(peftConfig(targets, options), null, 2)}\n`,
  );
  await saveSafetensors(peftTensorRecord(model), peftWeightsPath(directory));
}

/** Load LoRA adapters for a loaded decoder model from either mlxts-native or PEFT format. */
export async function loadCausalLMAdapters(
  model: CausalLM,
  directory: string,
  options: LoadCausalLMAdaptersOptions = {},
): Promise<void> {
  const requestedFormat = options.format ?? "auto";
  if (requestedFormat === "mlxts") {
    await loadLoRAAdapters(expectCausalLMModule(model), directory);
    return;
  }
  if (requestedFormat === "peft") {
    await loadPeftAdapters(model, directory);
    return;
  }

  const configPayload = JSON.parse(await Bun.file(configPath(directory)).text());
  const detectedFormat = detectAdapterFormat(configPayload);
  if (detectedFormat === "mlxts") {
    await loadLoRAAdapters(expectCausalLMModule(model), directory);
    return;
  }

  await loadPeftAdapters(model, directory);
}
