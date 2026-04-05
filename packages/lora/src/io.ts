import type { FlatEntry, MxArray } from "@mlxts/core";
import { loadSafetensors, saveSafetensors, treeUnflatten } from "@mlxts/core";
import type { Module } from "@mlxts/nn";
import { mkdirSync } from "fs";

import { applyLoRAToModule } from "./apply-module";
import { collectLoRAWrapperSlots } from "./traversal";
import type { LoRAAdapterConfig, LoRAAdapterTarget } from "./types";

function configPath(directory: string): string {
  return `${directory}/adapter_config.json`;
}

function weightsPath(directory: string): string {
  return `${directory}/adapters.safetensors`;
}

function adapterTargets(module: Module): LoRAAdapterTarget[] {
  return collectLoRAWrapperSlots(module).map((slot) => ({
    path: slot.path,
    rank: slot.child.rank,
    alpha: slot.child.alpha,
    dropout: slot.child.dropoutProbability,
  }));
}

function adapterConfig(module: Module): LoRAAdapterConfig {
  return {
    format: "mlxts-lora",
    version: 1,
    targets: adapterTargets(module),
  };
}

function adapterTensorRecord(module: Module): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const slot of collectLoRAWrapperSlots(module)) {
    tensors[`${slot.path}.loraA`] = slot.child.loraA;
    tensors[`${slot.path}.loraB`] = slot.child.loraB;
  }
  return tensors;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readTarget(value: unknown, index: number): LoRAAdapterTarget {
  if (!isRecord(value)) {
    throw new Error(`lora: adapter target ${index} must be an object.`);
  }
  const path = value.path;
  const rank = value.rank;
  const alpha = value.alpha;
  const dropout = value.dropout;
  if (typeof path !== "string" || path === "") {
    throw new Error(`lora: adapter target ${index}.path must be a non-empty string.`);
  }
  if (!Number.isInteger(rank) || typeof rank !== "number" || rank <= 0) {
    throw new Error(`lora: adapter target ${index}.rank must be a positive integer.`);
  }
  if (typeof alpha !== "number" || !Number.isFinite(alpha)) {
    throw new Error(`lora: adapter target ${index}.alpha must be a finite number.`);
  }
  if (typeof dropout !== "number" || dropout < 0 || dropout >= 1) {
    throw new Error(`lora: adapter target ${index}.dropout must satisfy 0 <= p < 1.`);
  }
  return {
    path,
    rank,
    alpha,
    dropout,
  };
}

function parseAdapterConfig(payload: unknown): LoRAAdapterConfig {
  if (!isRecord(payload)) {
    throw new Error("lora: adapter config must be a JSON object.");
  }
  if (payload.format !== "mlxts-lora") {
    throw new Error('lora: adapter config format must be "mlxts-lora".');
  }
  if (payload.version !== 1) {
    throw new Error("lora: adapter config version must be 1.");
  }
  const rawTargets = payload.targets;
  if (!Array.isArray(rawTargets)) {
    throw new Error("lora: adapter config targets must be an array.");
  }
  return {
    format: "mlxts-lora",
    version: 1,
    targets: rawTargets.map((value, index) => readTarget(value, index)),
  };
}

function expectedTensorNames(config: LoRAAdapterConfig): Set<string> {
  return new Set(
    config.targets.flatMap((target) => [`${target.path}.loraA`, `${target.path}.loraB`]),
  );
}

function tensorEntries(tensors: Record<string, MxArray>): FlatEntry[] {
  return Object.entries(tensors).map(([path, tensor]) => [path.split("."), tensor]);
}

/** Save the active LoRA adapter weights and config to a directory. */
export async function saveLoRAAdapters(module: Module, directory: string): Promise<void> {
  const config = adapterConfig(module);
  if (config.targets.length === 0) {
    throw new Error("lora: saveLoRAAdapters found no active LoRA wrappers.");
  }

  mkdirSync(directory, { recursive: true });
  await Bun.write(configPath(directory), `${JSON.stringify(config, null, 2)}\n`);
  await saveSafetensors(adapterTensorRecord(module), weightsPath(directory));
}

/** Load LoRA adapters from a directory and apply them to a model tree. */
export async function loadLoRAAdapters(module: Module, directory: string): Promise<void> {
  const config = parseAdapterConfig(JSON.parse(await Bun.file(configPath(directory)).text()));

  for (const target of config.targets) {
    applyLoRAToModule(module, {
      paths: [target.path],
      rank: target.rank,
      alpha: target.alpha,
      dropout: target.dropout,
    });
  }

  const loaded = await loadSafetensors(weightsPath(directory));
  const expectedNames = expectedTensorNames(config);

  try {
    for (const name of Object.keys(loaded.tensors)) {
      if (!expectedNames.has(name)) {
        throw new Error(`lora: unexpected adapter tensor "${name}".`);
      }
    }
    for (const name of expectedNames) {
      if (!(name in loaded.tensors)) {
        throw new Error(`lora: missing adapter tensor "${name}".`);
      }
    }

    module.update(treeUnflatten(tensorEntries(loaded.tensors)));
  } catch (error) {
    for (const tensor of Object.values(loaded.tensors)) {
      tensor.free();
    }
    throw error;
  }
}
