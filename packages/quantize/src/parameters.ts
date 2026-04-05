import type { QuantizationMode } from "@mlxts/core";

import type { QuantizationParameterOverrides, QuantizationParameters } from "./types";

function defaultsForMode(mode: QuantizationMode): QuantizationParameters {
  switch (mode) {
    case "mxfp4":
      return { groupSize: 32, bits: 4, mode };
    case "mxfp8":
      return { groupSize: 32, bits: 8, mode };
    case "nvfp4":
      return { groupSize: 16, bits: 4, mode };
    default:
      return { groupSize: 64, bits: 4, mode };
  }
}

/** Resolve a partial quantization configuration against mode defaults. */
export function resolveQuantizationParameters(
  overrides: QuantizationParameterOverrides = {},
  base?: QuantizationParameters,
): QuantizationParameters {
  const mode = overrides.mode ?? base?.mode ?? "affine";
  const defaults = defaultsForMode(mode);
  const groupSize = overrides.groupSize ?? base?.groupSize ?? defaults.groupSize;
  const bits = overrides.bits ?? base?.bits ?? defaults.bits;
  return {
    groupSize,
    bits,
    mode,
  };
}
