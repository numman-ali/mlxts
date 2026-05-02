import { DiffusionConfigError } from "../errors";
import { PIPELINE_SPECS, type PipelineSpec } from "./pipeline-specs";

/** Diffusers pipeline classes with recognized local generative-media snapshot layout. */
export type DiffusersPipelineClassName =
  | "StableDiffusionPipeline"
  | "StableDiffusionXLPipeline"
  | "StableDiffusion3Pipeline"
  | "FluxPipeline"
  | "Flux2KleinPipeline"
  | "QwenImagePipeline"
  | "QwenImageEditPipeline"
  | "QwenImageEditPlusPipeline"
  | "ZImagePipeline"
  | "LTXPipeline"
  | "LTXConditionPipeline"
  | "LTXLatentUpsamplePipeline"
  | "LTX2Pipeline"
  | "LTX2LatentUpsamplePipeline";

/** Pipeline family represented by a supported Diffusers `model_index.json`. */
export type DiffusionPipelineKind =
  | "stable-diffusion"
  | "stable-diffusion-xl"
  | "stable-diffusion-3"
  | "flux"
  | "flux2-klein"
  | "qwen-image"
  | "qwen-image-edit"
  | "qwen-image-edit-plus"
  | "z-image"
  | "ltx-video"
  | "ltx-video-latent-upsample"
  | "ltx2"
  | "ltx2-latent-upsample";

/** Component folders recognized in Diffusers generative-media snapshots. */
export type DiffusionComponentName =
  | "vae"
  | "audio_vae"
  | "text_encoder"
  | "text_encoder_2"
  | "text_encoder_3"
  | "tokenizer"
  | "tokenizer_2"
  | "tokenizer_3"
  | "unet"
  | "transformer"
  | "scheduler"
  | "safety_checker"
  | "feature_extractor"
  | "processor"
  | "image_encoder"
  | "latent_upsampler"
  | "connectors"
  | "vocoder";

/** Semantic role for a Diffusers component folder. */
export type DiffusionComponentRole =
  | "vae"
  | "audio-vae"
  | "text-encoder"
  | "tokenizer"
  | "backbone"
  | "scheduler"
  | "safety"
  | "image-processor"
  | "image-encoder"
  | "latent-upsampler"
  | "connector"
  | "vocoder";

/** Component entry parsed from `model_index.json`. */
export type DiffusionModelIndexComponent = {
  name: DiffusionComponentName;
  role: DiffusionComponentRole;
  library: string | null;
  className: string | null;
  enabled: boolean;
  optional: boolean;
  subfolder: string;
};

/** Parsed Diffusers pipeline manifest translated into package-owned terms. */
export type ParsedDiffusionModelIndex = {
  kind: DiffusionPipelineKind;
  className: DiffusersPipelineClassName;
  diffusersVersion?: string;
  components: readonly DiffusionModelIndexComponent[];
  pipelineConfig: Record<string, string | number | boolean | null>;
  rawConfig: Record<string, unknown>;
};

export type DiffusionModelIndexComponentSpec = {
  name: DiffusionComponentName;
  role: DiffusionComponentRole;
  optional?: boolean;
  allowed: readonly (readonly [string, string])[];
  requiresConfig?: boolean;
  requiresTokenizerFiles?: boolean;
  requiresWeights?: boolean;
};

const SCALAR_CONFIG_KEYS = new Set([
  "add_watermarker",
  "requires_safety_checker",
  "force_zeros_for_empty_prompt",
  "is_distilled",
]);

function fieldName(context: string, key: string): string {
  return `${context}.${key}`;
}

function valueDescription(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new DiffusionConfigError(`${context} must be a JSON object.`);
  }
  return Object.fromEntries(Object.entries(value));
}

function requiredString(record: Record<string, unknown>, key: string, context: string): string {
  const value = record[key];
  if (typeof value !== "string" || value === "") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a non-empty string, got ${valueDescription(value)}.`,
    );
  }
  return value;
}

function optionalString(
  record: Record<string, unknown>,
  key: string,
  context: string,
): string | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string" || value === "") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a non-empty string when present, got ${valueDescription(
        value,
      )}.`,
    );
  }
  return value;
}

function parsePipelineClassName(
  record: Record<string, unknown>,
  context: string,
): DiffusersPipelineClassName {
  const className = requiredString(record, "_class_name", context);
  if (
    className === "StableDiffusionPipeline" ||
    className === "StableDiffusionXLPipeline" ||
    className === "StableDiffusion3Pipeline" ||
    className === "FluxPipeline" ||
    className === "Flux2KleinPipeline" ||
    className === "QwenImagePipeline" ||
    className === "QwenImageEditPipeline" ||
    className === "QwenImageEditPlusPipeline" ||
    className === "ZImagePipeline" ||
    className === "LTXPipeline" ||
    className === "LTXConditionPipeline" ||
    className === "LTXLatentUpsamplePipeline" ||
    className === "LTX2Pipeline" ||
    className === "LTX2LatentUpsamplePipeline"
  ) {
    return className;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, "_class_name")}="${className}" is not supported yet.`,
  );
}

function parseScalarConfigValue(
  value: unknown,
  key: string,
  context: string,
): string | number | boolean | null {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, key)} must be a scalar config value, got ${valueDescription(value)}.`,
  );
}

function parseComponentTuple(
  rawValue: unknown,
  spec: DiffusionModelIndexComponentSpec,
  context: string,
): DiffusionModelIndexComponent {
  if (!Array.isArray(rawValue) || rawValue.length !== 2) {
    throw new DiffusionConfigError(
      `${fieldName(context, spec.name)} must be a [library, class] pair.`,
    );
  }
  const [library, className] = rawValue;
  if (library === null && className === null) {
    if (spec.optional === true) {
      return {
        name: spec.name,
        role: spec.role,
        library: null,
        className: null,
        enabled: false,
        optional: true,
        subfolder: spec.name,
      };
    }
    throw new DiffusionConfigError(`${fieldName(context, spec.name)} cannot be disabled.`);
  }
  if (
    typeof library !== "string" ||
    library === "" ||
    typeof className !== "string" ||
    className === ""
  ) {
    throw new DiffusionConfigError(
      `${fieldName(context, spec.name)} must contain non-empty library and class strings.`,
    );
  }
  const supported = spec.allowed.some(
    ([allowedLibrary, allowedClass]) => allowedLibrary === library && allowedClass === className,
  );
  if (!supported) {
    throw new DiffusionConfigError(
      `${fieldName(context, spec.name)}=[${library}, ${className}] is not supported yet.`,
    );
  }
  return {
    name: spec.name,
    role: spec.role,
    library,
    className,
    enabled: true,
    optional: spec.optional === true,
    subfolder: spec.name,
  };
}

function parseComponents(
  record: Record<string, unknown>,
  spec: PipelineSpec,
  context: string,
): DiffusionModelIndexComponent[] {
  const components: DiffusionModelIndexComponent[] = [];
  const knownComponents = new Set<string>(spec.components.map((component) => component.name));
  for (const component of spec.components) {
    const value = record[component.name];
    if (value === undefined) {
      if (component.optional === true) {
        continue;
      }
      throw new DiffusionConfigError(`${fieldName(context, component.name)} is required.`);
    }
    components.push(parseComponentTuple(value, component, context));
  }
  for (const [key, value] of Object.entries(record)) {
    if (key.startsWith("_") || knownComponents.has(key) || SCALAR_CONFIG_KEYS.has(key)) {
      continue;
    }
    if (Array.isArray(value)) {
      throw new DiffusionConfigError(`${fieldName(context, key)} is not a supported component.`);
    }
    throw new DiffusionConfigError(`${fieldName(context, key)} is not a supported pipeline field.`);
  }
  return components;
}

function parsePipelineConfig(
  record: Record<string, unknown>,
  context: string,
): Record<string, string | number | boolean | null> {
  const config: Record<string, string | number | boolean | null> = {};
  for (const key of SCALAR_CONFIG_KEYS) {
    if (record[key] !== undefined) {
      config[key] = parseScalarConfigValue(record[key], key, context);
    }
  }
  return config;
}

export function getDiffusionComponentSpec(
  modelIndex: ParsedDiffusionModelIndex,
  component: DiffusionModelIndexComponent,
): DiffusionModelIndexComponentSpec {
  const spec = PIPELINE_SPECS[modelIndex.className].components.find(
    (candidate) => candidate.name === component.name,
  );
  if (spec === undefined) {
    throw new DiffusionConfigError(`model_index.json.${component.name} has no component spec.`);
  }
  return spec;
}

/** Parse a Diffusers `model_index.json` payload into a supported pipeline manifest. */
export function parseDiffusionModelIndex(rawConfig: unknown): ParsedDiffusionModelIndex {
  const context = "model_index.json";
  const record = expectRecord(rawConfig, context);
  const className = parsePipelineClassName(record, context);
  const spec = PIPELINE_SPECS[className];
  const diffusersVersion = optionalString(record, "_diffusers_version", context);
  const parsed: ParsedDiffusionModelIndex = {
    kind: spec.kind,
    className,
    components: parseComponents(record, spec, context),
    pipelineConfig: parsePipelineConfig(record, context),
    rawConfig: record,
  };
  if (diffusersVersion !== undefined) {
    return { ...parsed, diffusersVersion };
  }
  return parsed;
}
