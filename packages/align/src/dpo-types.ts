import type { PreferenceBatch } from "@mlxts/data";
import type { CausalLM } from "@mlxts/transformers";

import type { OptimizerLike } from "./sft-types";

export type DPOTrainOptions = {
  referenceModel: CausalLM;
  optimizer: OptimizerLike;
  batches: readonly PreferenceBatch[];
  beta?: number;
  learningRate?: number;
  maxGradNorm?: number | null;
};
