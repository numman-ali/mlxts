/**
 * Explicit Qwen 3.5 / 3.6 multimodal checkpoint loading.
 * @module
 */

import { loadPreparedCausalLM, type PreparedModel } from "../../load";
import type { LoadCausalLMOptions } from "../../types";
import { Qwen3_5ForConditionalGeneration } from "./conditional";
import { qwen3_5ConditionalFamily } from "./config";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readConfigRecord(source: string): Promise<Record<string, unknown> | null> {
  try {
    const value: unknown = await Bun.file(`${source.replace(/\/$/, "")}/config.json`).json();
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function hasArchitecture(config: Record<string, unknown>, architecture: string): boolean {
  const architectures = config.architectures;
  return Array.isArray(architectures) && architectures.includes(architecture);
}

function prepareQwen3_5ConditionalModel(
  configRecord: Record<string, unknown>,
): PreparedModel<Qwen3_5ForConditionalGeneration> {
  const config = qwen3_5ConditionalFamily.parseConfig(configRecord);
  const model = new Qwen3_5ForConditionalGeneration(config);
  return {
    registration: qwen3_5ConditionalFamily,
    config,
    model,
  };
}

/** Load the full Qwen 3.5 / 3.6 multimodal wrapper with its vision tower. */
export async function loadQwen3_5ForConditionalGeneration(
  source: string,
  options: LoadCausalLMOptions = {},
): Promise<Qwen3_5ForConditionalGeneration> {
  return loadPreparedCausalLM(source, options, prepareQwen3_5ConditionalModel);
}

/** True when a resolved checkpoint advertises the Qwen 3.5 conditional multimodal wrapper. */
export async function shouldLoadQwen3_5ForConditionalGeneration(source: string): Promise<boolean> {
  const config = await readConfigRecord(source);
  if (config === null) {
    return false;
  }
  return (
    config.model_type === "qwen3_5" &&
    isRecord(config.vision_config) &&
    hasArchitecture(config, "Qwen3_5ForConditionalGeneration")
  );
}
