import type { DiffusionModelIndexComponentSpec } from "./model-index";

export const LTX_VIDEO_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "transformer",
    role: "backbone",
    allowed: [["diffusers", "LTXVideoTransformer3DModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKLLTXVideo"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "scheduler",
    role: "scheduler",
    allowed: [["diffusers", "FlowMatchEulerDiscreteScheduler"]],
    requiresConfig: true,
  },
  {
    name: "text_encoder",
    role: "text-encoder",
    allowed: [["transformers", "T5EncoderModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [
      ["transformers", "T5Tokenizer"],
      ["transformers", "T5TokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
];

export const LTX2_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "transformer",
    role: "backbone",
    allowed: [["diffusers", "LTX2VideoTransformer3DModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKLLTX2Video"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "audio_vae",
    role: "audio-vae",
    allowed: [["diffusers", "AutoencoderKLLTX2Audio"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "scheduler",
    role: "scheduler",
    allowed: [["diffusers", "FlowMatchEulerDiscreteScheduler"]],
    requiresConfig: true,
  },
  {
    name: "text_encoder",
    role: "text-encoder",
    allowed: [["transformers", "Gemma3ForConditionalGeneration"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [
      ["transformers", "GemmaTokenizer"],
      ["transformers", "GemmaTokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
  {
    name: "connectors",
    role: "connector",
    allowed: [["ltx2", "LTX2TextConnectors"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "vocoder",
    role: "vocoder",
    allowed: [["ltx2", "LTX2Vocoder"]],
    requiresConfig: true,
    requiresWeights: true,
  },
];
