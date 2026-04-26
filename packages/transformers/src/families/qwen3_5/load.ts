/**
 * Explicit Qwen 3.5 / 3.6 multimodal checkpoint loading.
 * @module
 */

import { loadPreparedCausalLM, type PreparedModel } from "../../load";
import type { LoadCausalLMOptions } from "../../types";
import { Qwen3_5ForConditionalGeneration } from "./conditional";
import { qwen3_5ConditionalFamily } from "./config";

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
