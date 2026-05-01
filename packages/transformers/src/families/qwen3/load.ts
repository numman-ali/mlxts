/**
 * Explicit dense Qwen3 text checkpoint loading.
 * @module
 */

import { loadPreparedCausalLM, type PreparedModel } from "../../load";
import type { LoadCausalLMOptions } from "../../types";
import { LlamaLikeCausalLM } from "../llama-like/model";
import { qwen3Family } from "./config";

/** Dense Qwen3 text model using the shared LLaMA-like causal LM runtime. */
export type Qwen3TextCausalLM = LlamaLikeCausalLM;

function prepareQwen3Model(
  configRecord: Record<string, unknown>,
): PreparedModel<LlamaLikeCausalLM> {
  const config = qwen3Family.parseConfig(configRecord);
  const model = new LlamaLikeCausalLM(config);
  return {
    registration: qwen3Family,
    config,
    model,
  };
}

/** Load a dense Qwen3 text model from a local checkpoint or Hugging Face repo. */
export async function loadQwen3CausalLM(
  source: string,
  options: LoadCausalLMOptions = {},
): Promise<Qwen3TextCausalLM> {
  return loadPreparedCausalLM(source, options, prepareQwen3Model);
}
