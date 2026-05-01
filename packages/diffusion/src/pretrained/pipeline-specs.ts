import { LTX_VIDEO_COMPONENTS, LTX2_COMPONENTS } from "./ltx-pipeline-specs";
import type {
  DiffusersPipelineClassName,
  DiffusionModelIndexComponentSpec,
  DiffusionPipelineKind,
} from "./model-index";

export type PipelineSpec = {
  kind: DiffusionPipelineKind;
  components: readonly DiffusionModelIndexComponentSpec[];
};

const STABLE_DIFFUSION_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKL"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "text_encoder",
    role: "text-encoder",
    allowed: [["transformers", "CLIPTextModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [
      ["transformers", "CLIPTokenizer"],
      ["transformers", "CLIPTokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
  {
    name: "unet",
    role: "backbone",
    allowed: [["diffusers", "UNet2DConditionModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "scheduler",
    role: "scheduler",
    allowed: [
      ["diffusers", "DDIMScheduler"],
      ["diffusers", "EulerDiscreteScheduler"],
    ],
    requiresConfig: true,
  },
  {
    name: "safety_checker",
    role: "safety",
    optional: true,
    allowed: [["stable_diffusion", "StableDiffusionSafetyChecker"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "feature_extractor",
    role: "image-processor",
    optional: true,
    allowed: [["transformers", "CLIPImageProcessor"]],
    requiresConfig: true,
  },
  {
    name: "image_encoder",
    role: "image-encoder",
    optional: true,
    allowed: [["transformers", "CLIPVisionModelWithProjection"]],
    requiresConfig: true,
    requiresWeights: true,
  },
];

const STABLE_DIFFUSION_XL_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKL"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "text_encoder",
    role: "text-encoder",
    allowed: [["transformers", "CLIPTextModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "text_encoder_2",
    role: "text-encoder",
    allowed: [["transformers", "CLIPTextModelWithProjection"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [
      ["transformers", "CLIPTokenizer"],
      ["transformers", "CLIPTokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
  {
    name: "tokenizer_2",
    role: "tokenizer",
    allowed: [
      ["transformers", "CLIPTokenizer"],
      ["transformers", "CLIPTokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
  {
    name: "unet",
    role: "backbone",
    allowed: [["diffusers", "UNet2DConditionModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "scheduler",
    role: "scheduler",
    allowed: [
      ["diffusers", "DDIMScheduler"],
      ["diffusers", "EulerDiscreteScheduler"],
    ],
    requiresConfig: true,
  },
  {
    name: "feature_extractor",
    role: "image-processor",
    optional: true,
    allowed: [["transformers", "CLIPImageProcessor"]],
    requiresConfig: true,
  },
  {
    name: "image_encoder",
    role: "image-encoder",
    optional: true,
    allowed: [["transformers", "CLIPVisionModelWithProjection"]],
    requiresConfig: true,
    requiresWeights: true,
  },
];

const STABLE_DIFFUSION_3_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "transformer",
    role: "backbone",
    allowed: [["diffusers", "SD3Transformer2DModel"]],
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
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKL"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "text_encoder",
    role: "text-encoder",
    allowed: [["transformers", "CLIPTextModelWithProjection"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [
      ["transformers", "CLIPTokenizer"],
      ["transformers", "CLIPTokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
  {
    name: "text_encoder_2",
    role: "text-encoder",
    allowed: [["transformers", "CLIPTextModelWithProjection"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer_2",
    role: "tokenizer",
    allowed: [
      ["transformers", "CLIPTokenizer"],
      ["transformers", "CLIPTokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
  {
    name: "text_encoder_3",
    role: "text-encoder",
    allowed: [["transformers", "T5EncoderModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer_3",
    role: "tokenizer",
    allowed: [["transformers", "T5TokenizerFast"]],
    requiresTokenizerFiles: true,
  },
  {
    name: "image_encoder",
    role: "image-encoder",
    optional: true,
    allowed: [["transformers", "SiglipVisionModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "feature_extractor",
    role: "image-processor",
    optional: true,
    allowed: [["transformers", "SiglipImageProcessor"]],
    requiresConfig: true,
  },
];

const FLUX_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "transformer",
    role: "backbone",
    allowed: [["diffusers", "FluxTransformer2DModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKL"]],
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
    allowed: [["transformers", "CLIPTextModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [["transformers", "CLIPTokenizer"]],
    requiresTokenizerFiles: true,
  },
  {
    name: "text_encoder_2",
    role: "text-encoder",
    allowed: [["transformers", "T5EncoderModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer_2",
    role: "tokenizer",
    allowed: [["transformers", "T5TokenizerFast"]],
    requiresTokenizerFiles: true,
  },
];

const FLUX2_KLEIN_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "transformer",
    role: "backbone",
    allowed: [["diffusers", "Flux2Transformer2DModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKLFlux2"]],
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
    allowed: [["transformers", "Qwen3ForCausalLM"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [["transformers", "Qwen2TokenizerFast"]],
    requiresTokenizerFiles: true,
  },
];

const QWEN_IMAGE_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "transformer",
    role: "backbone",
    allowed: [["diffusers", "QwenImageTransformer2DModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKLQwenImage"]],
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
    allowed: [["transformers", "Qwen2_5_VLForConditionalGeneration"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [
      ["transformers", "Qwen2Tokenizer"],
      ["transformers", "Qwen2TokenizerFast"],
    ],
    requiresTokenizerFiles: true,
  },
];

const Z_IMAGE_COMPONENTS: readonly DiffusionModelIndexComponentSpec[] = [
  {
    name: "transformer",
    role: "backbone",
    allowed: [["diffusers", "ZImageTransformer2DModel"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "vae",
    role: "vae",
    allowed: [["diffusers", "AutoencoderKL"]],
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
    allowed: [["transformers", "Qwen3Model"]],
    requiresConfig: true,
    requiresWeights: true,
  },
  {
    name: "tokenizer",
    role: "tokenizer",
    allowed: [["transformers", "Qwen2Tokenizer"]],
    requiresTokenizerFiles: true,
  },
];

export const PIPELINE_SPECS: Record<DiffusersPipelineClassName, PipelineSpec> = {
  StableDiffusionPipeline: {
    kind: "stable-diffusion",
    components: STABLE_DIFFUSION_COMPONENTS,
  },
  StableDiffusionXLPipeline: {
    kind: "stable-diffusion-xl",
    components: STABLE_DIFFUSION_XL_COMPONENTS,
  },
  StableDiffusion3Pipeline: {
    kind: "stable-diffusion-3",
    components: STABLE_DIFFUSION_3_COMPONENTS,
  },
  FluxPipeline: {
    kind: "flux",
    components: FLUX_COMPONENTS,
  },
  Flux2KleinPipeline: {
    kind: "flux2-klein",
    components: FLUX2_KLEIN_COMPONENTS,
  },
  QwenImagePipeline: {
    kind: "qwen-image",
    components: QWEN_IMAGE_COMPONENTS,
  },
  ZImagePipeline: {
    kind: "z-image",
    components: Z_IMAGE_COMPONENTS,
  },
  LTXPipeline: {
    kind: "ltx-video",
    components: LTX_VIDEO_COMPONENTS,
  },
  LTXConditionPipeline: {
    kind: "ltx-video",
    components: LTX_VIDEO_COMPONENTS,
  },
  LTX2Pipeline: {
    kind: "ltx2",
    components: LTX2_COMPONENTS,
  },
};
