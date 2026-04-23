import type { ChatMessage, PreferenceExample, TokenSupervisionExample } from "@mlxts/data";
import type { InteractionProfile, loadPretrainedTokenizer } from "@mlxts/transformers";

import type { DPOProofProfile, TrainingProofArgs } from "./args";

export type MetricPair = {
  before: number;
  after: number;
  delta: number;
};

export type StageReport = {
  stage: string;
  evalLoss?: MetricPair;
  rewardAccuracy?: MetricPair;
  rewardMargin?: MetricPair;
  chosenReward?: MetricPair;
  rejectedReward?: MetricPair;
  chosenLogProb?: MetricPair;
  rejectedLogProb?: MetricPair;
  rawPreferenceAccuracy?: MetricPair;
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
  preset: "attention" | "attention+mlp" | "all-linear";
  targets: string[];
};

export type DPOProofConfig = {
  profile: DPOProofProfile;
  preset: AppliedLoRA["preset"];
  lastLayers: number | null;
  rank: number;
  alpha: number;
  dropout: number;
  learningRate: number;
  beta: number;
};
