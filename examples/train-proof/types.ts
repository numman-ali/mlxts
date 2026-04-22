import type { ChatMessage, PreferenceExample, TokenSupervisionExample } from "@mlxts/data";
import type { InteractionProfile, loadPretrainedTokenizer } from "@mlxts/transformers";

import type { TrainingProofArgs } from "./args";

export type MetricPair = {
  before: number;
  after: number;
  delta: number;
};

export type StageReport = {
  stage: string;
  evalLoss?: MetricPair;
  preferenceAccuracy?: MetricPair;
  averageTrainingLoss?: number;
  sampleText?: string;
  notes: string[];
};

export type TrainingProofReport = {
  source: string;
  quantizedOutputDir: string;
  datasetSource: TrainingProofArgs["datasetSource"];
  trainLimit: number;
  evalLimit: number;
  batchSize: number;
  steps: number;
  maxSequenceLength: number;
  seed: number;
  dataNotes: string[];
  stages: StageReport[];
};

export type LoadedAssets = {
  tokenizer: Awaited<ReturnType<typeof loadPretrainedTokenizer>>;
  profile: InteractionProfile;
};

export type PreparedTrainingProofData = {
  supervisionTrain: readonly TokenSupervisionExample[];
  supervisionEval: readonly TokenSupervisionExample[];
  preferenceTrain: readonly PreferenceExample[];
  preferenceEval: readonly PreferenceExample[];
  samplePromptMessages: readonly ChatMessage[];
  notes: string[];
};

export type LoRAStageMode = "lora" | "qlora" | "dpo";

export type AppliedLoRA = {
  preset: "attention" | "all-linear";
  targets: string[];
};
