import type {
  Ltx2AudioAutoencoderConfig,
  Ltx2VideoAutoencoderConfig,
} from "./config-ltx2-autoencoders";
import type { Ltx2TextConnectorsConfig, Ltx2VocoderConfig } from "./config-ltx2-media";
import type { Ltx2VideoTransformerConfig } from "./config-ltx2-transformer";

export {
  type Ltx2AudioAutoencoderConfig,
  type Ltx2AudioCausalityAxis,
  type Ltx2AudioNormType,
  type Ltx2SpatialPaddingMode,
  type Ltx2VideoAutoencoderConfig,
  type Ltx2VideoDownBlockType,
  type Ltx2VideoDownsampleType,
  type Ltx2VideoUpsampleType,
  parseLtx2AudioAutoencoderConfig,
  parseLtx2VideoAutoencoderConfig,
} from "./config-ltx2-autoencoders";
export {
  type Ltx2TextConnectorsConfig,
  type Ltx2VocoderActivation,
  type Ltx2VocoderConfig,
  type Ltx2VocoderFinalActivation,
  parseLtx2TextConnectorsConfig,
  parseLtx2VocoderConfig,
} from "./config-ltx2-media";
export {
  type Ltx2QkNorm,
  type Ltx2RopeType,
  type Ltx2VideoTransformerConfig,
  parseLtx2VideoTransformerConfig,
} from "./config-ltx2-transformer";

/** Configs required before LTX-2 model construction can begin. */
export type Ltx2ComponentConfigs = {
  pipelineKind: "ltx2";
  transformer: Ltx2VideoTransformerConfig;
  vae: Ltx2VideoAutoencoderConfig;
  audioVae: Ltx2AudioAutoencoderConfig;
  connectors: Ltx2TextConnectorsConfig;
  vocoder: Ltx2VocoderConfig;
};
