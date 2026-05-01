/**
 * Explicit Qwen2-family text checkpoint loading.
 * @module
 */

import { loadPreparedCausalLM, type PreparedModel } from "../../load";
import type { LoadCausalLMOptions } from "../../types";
import { LlamaLikeCausalLM } from "../llama-like/model";
import { qwen2Family } from "./config";

/** Qwen2-family text model using the shared LLaMA-like causal LM runtime. */
export type Qwen2TextCausalLM = LlamaLikeCausalLM;

function prepareQwen2Model(
  configRecord: Record<string, unknown>,
): PreparedModel<LlamaLikeCausalLM> {
  const config = qwen2Family.parseConfig(configRecord);
  const model = new LlamaLikeCausalLM(config);
  return {
    registration: qwen2Family,
    config,
    model,
  };
}

/** Load a Qwen2 text model from a local checkpoint or Hugging Face repo. */
export async function loadQwen2CausalLM(
  source: string,
  options: LoadCausalLMOptions = {},
): Promise<Qwen2TextCausalLM> {
  return loadPreparedCausalLM(source, options, prepareQwen2Model);
}

/** Load the text-decoder portion of a Qwen2.5-VL conditional-generation checkpoint. */
export async function loadQwen2_5VLTextEncoder(
  source: string,
  options: LoadCausalLMOptions = {},
): Promise<Qwen2TextCausalLM> {
  return loadPreparedCausalLM(source, options, prepareQwen2Model);
}
