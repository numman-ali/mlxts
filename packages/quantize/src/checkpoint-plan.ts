import type { QuantizationMode } from "@mlxts/core";

import { resolveQuantizationParameters } from "./parameters";
import type {
  QuantizationParameterOverrides,
  QuantizedCheckpointPlan,
  QuantizedCheckpointProvider,
  QuantizedCheckpointRule,
} from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readMode(value: unknown): QuantizationMode | null {
  switch (value) {
    case "affine":
    case "mxfp4":
    case "mxfp8":
    case "nvfp4":
      return value;
    default:
      return null;
  }
}

function readPositiveInteger(value: unknown, label: string): number | null {
  if (!Number.isInteger(value) || typeof value !== "number" || value <= 0) {
    throw new Error(`quantize: ${label} must be a positive integer.`);
  }
  return value;
}

type OverrideParser = (
  entry: unknown,
  context: string,
  overrides: QuantizationParameterOverrides,
) => void;

const OVERRIDE_PARSERS: Record<string, OverrideParser> = {
  group_size(entry, context, overrides) {
    const groupSize = readPositiveInteger(entry, `${context}.group_size`);
    if (groupSize !== null) {
      overrides.groupSize = groupSize;
    }
  },
  groupSize(entry, context, overrides) {
    const groupSize = readPositiveInteger(entry, `${context}.groupSize`);
    if (groupSize !== null) {
      overrides.groupSize = groupSize;
    }
  },
  bits(entry, context, overrides) {
    const bits = readPositiveInteger(entry, `${context}.bits`);
    if (bits !== null) {
      overrides.bits = bits;
    }
  },
  mode(entry, context, overrides) {
    const mode = readMode(entry);
    if (mode === null) {
      throw new Error(`quantize: ${context}.mode must be one of affine, mxfp4, mxfp8, or nvfp4.`);
    }
    overrides.mode = mode;
  },
};

function parseOverrides(
  value: Record<string, unknown>,
  context: string,
): QuantizationParameterOverrides {
  const overrides: QuantizationParameterOverrides = {};
  for (const [key, entry] of Object.entries(value)) {
    const parseOverride = OVERRIDE_PARSERS[key];
    if (parseOverride !== undefined) {
      parseOverride(entry, context, overrides);
    }
  }
  return overrides;
}

function pushRule(
  rules: QuantizedCheckpointRule[],
  path: string,
  enabled: boolean,
  params?: QuantizationParameterOverrides,
): void {
  if (rules.some((rule) => rule.path === path)) {
    throw new Error(`quantize: duplicate quantization rule for "${path}".`);
  }

  if (params === undefined) {
    rules.push({ path, enabled });
    return;
  }

  rules.push({ path, enabled, params });
}

function parseRuleEntries(
  record: Record<string, unknown>,
  sourceKey: string,
): QuantizedCheckpointRule[] {
  const rules: QuantizedCheckpointRule[] = [];
  for (const [key, value] of Object.entries(record)) {
    if (key === "group_size" || key === "groupSize" || key === "bits" || key === "mode") {
      continue;
    }

    if (value === true) {
      pushRule(rules, key, true);
      continue;
    }

    if (value === false) {
      pushRule(rules, key, false);
      continue;
    }

    if (isRecord(value)) {
      pushRule(rules, key, true, parseOverrides(value, `${sourceKey}.${key}`));
    }
  }
  return rules;
}

function buildPlan(
  provider: string,
  sourceKey: "quantization" | "quantization_config",
  defaults: QuantizationParameterOverrides,
  rules: QuantizedCheckpointRule[],
): QuantizedCheckpointPlan {
  return {
    provider,
    sourceKey,
    defaults: resolveQuantizationParameters(defaults),
    explicitOnly: rules.length > 0,
    rules,
  };
}

function resolvePlanRecord(
  sourceKey: "quantization" | "quantization_config",
  value: unknown,
): QuantizedCheckpointPlan | null {
  if (!isRecord(value)) {
    return null;
  }

  if (sourceKey === "quantization_config") {
    const quantMethod = value.quant_method;
    if (quantMethod === "mxfp4") {
      return buildPlan("mlx", sourceKey, { mode: "mxfp4", groupSize: 32, bits: 4 }, []);
    }
    if (quantMethod === "compressed-tensors") {
      return buildPlan("mlx", sourceKey, { mode: "affine", groupSize: 32, bits: 4 }, []);
    }
    if (quantMethod === "awq") {
      return buildPlan("awq", sourceKey, parseOverrides(value, sourceKey), []);
    }
    if (quantMethod === "gptq") {
      return buildPlan("gptq", sourceKey, parseOverrides(value, sourceKey), []);
    }
  }

  const defaults = parseOverrides(value, sourceKey);
  const rules = parseRuleEntries(value, sourceKey);
  if (Object.keys(defaults).length === 0 && rules.length === 0) {
    return null;
  }
  return buildPlan("mlx", sourceKey, defaults, rules);
}

function rootCandidates(
  config: Record<string, unknown>,
): Array<{ sourceKey: "quantization" | "quantization_config"; value: unknown }> {
  const candidates: Array<{ sourceKey: "quantization" | "quantization_config"; value: unknown }> =
    [];
  if ("quantization" in config) {
    candidates.push({ sourceKey: "quantization", value: config.quantization });
  }
  if ("quantization_config" in config) {
    candidates.push({ sourceKey: "quantization_config", value: config.quantization_config });
  }
  const textConfig = config.text_config;
  if (isRecord(textConfig)) {
    if ("quantization" in textConfig && !("quantization" in config)) {
      candidates.push({ sourceKey: "quantization", value: textConfig.quantization });
    }
    if ("quantization_config" in textConfig && !("quantization_config" in config)) {
      candidates.push({
        sourceKey: "quantization_config",
        value: textConfig.quantization_config,
      });
    }
  }
  return candidates;
}

function mlxCheckpointProvider(): QuantizedCheckpointProvider {
  return {
    name: "mlx",
    resolve(config) {
      for (const candidate of rootCandidates(config)) {
        const plan = resolvePlanRecord(candidate.sourceKey, candidate.value);
        if (plan !== null) {
          return plan;
        }
      }
      return null;
    },
  };
}

const providers: QuantizedCheckpointProvider[] = [mlxCheckpointProvider()];

/** Register a custom quantized-checkpoint metadata provider. */
export function registerQuantizedCheckpointProvider(provider: QuantizedCheckpointProvider): void {
  providers.push(provider);
}

/** Parse quantization metadata from a pretrained config payload. */
export function resolveCheckpointQuantizationPlan(
  config: Record<string, unknown>,
): QuantizedCheckpointPlan | null {
  for (const provider of providers) {
    const plan = provider.resolve(config);
    if (plan !== null) {
      return plan;
    }
  }
  return null;
}

/** Translate explicit checkpoint rule paths into another module namespace. */
export function translateCheckpointQuantizationPlanPaths(
  plan: QuantizedCheckpointPlan,
  translatePath: (path: string) => string | null,
): QuantizedCheckpointPlan {
  const translatedRules: QuantizedCheckpointRule[] = [];
  for (const rule of plan.rules) {
    const translated = translatePath(rule.path);
    if (translated === null) {
      if (rule.enabled) {
        throw new Error(
          `quantize: could not translate explicit quantization rule path "${rule.path}".`,
        );
      }
      continue;
    }

    pushRule(translatedRules, translated, rule.enabled, rule.params);
  }

  return {
    provider: plan.provider,
    sourceKey: plan.sourceKey,
    defaults: plan.defaults,
    explicitOnly: plan.explicitOnly,
    rules: translatedRules,
  };
}
