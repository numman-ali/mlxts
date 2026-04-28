import { resolveFamily } from "../../registry";
import type { CausalLM } from "../../types";

export function rawConfigForFamily(
  modelType: "llama" | "mistral" | "gemma" | "phi3",
): Record<string, unknown> {
  const base = {
    vocab_size: 5,
    hidden_size: 8,
    intermediate_size: 16,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 2,
    max_position_embeddings: 32,
    rope_theta: 10_000,
    rms_norm_eps: 1e-6,
    attention_bias: false,
  };

  if (modelType === "phi3") {
    return {
      ...base,
      model_type: "phi3",
      tie_word_embeddings: false,
      hidden_act: "silu",
      partial_rotary_factor: 1.0,
      sliding_window: 8,
    };
  }

  if (modelType === "gemma") {
    return {
      ...base,
      model_type: "gemma",
      tie_word_embeddings: false,
      head_dim: 4,
      hidden_act: "gelu_pytorch_tanh",
    };
  }

  if (modelType === "mistral") {
    return {
      ...base,
      model_type: "mistral",
      tie_word_embeddings: false,
      sliding_window: 8,
    };
  }

  return {
    ...base,
    model_type: "llama",
    tie_word_embeddings: false,
  };
}

export function createModel(rawConfig: Record<string, unknown>): CausalLM {
  const modelType = rawConfig.model_type;
  if (typeof modelType !== "string") {
    throw new Error("expected rawConfig.model_type to be a string");
  }
  const registration = resolveFamily(modelType);
  return registration.createModel(registration.parseConfig(rawConfig));
}
