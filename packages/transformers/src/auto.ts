/**
 * Thin convenience aliases for the canonical loading functions.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";

import { loadCausalLM, loadPretrainedTokenizer } from "./load";
import type { CausalLM, LoadCausalLMOptions, LoadPretrainedTokenizerOptions } from "./types";

/** Convenience alias over `loadCausalLM()`. */
export const AutoModel = {
  fromPretrained(source: string, options?: LoadCausalLMOptions): Promise<CausalLM> {
    return loadCausalLM(source, options);
  },
} as const;

/** Convenience alias over `loadPretrainedTokenizer()`. */
export const AutoTokenizer = {
  fromPretrained(source: string, options?: LoadPretrainedTokenizerOptions): Promise<Tokenizer> {
    return loadPretrainedTokenizer(source, options);
  },
} as const;
