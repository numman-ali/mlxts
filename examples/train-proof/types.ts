import type { TrainingStepLoss } from "@mlxts/align";
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
  trainingStepLosses?: readonly TrainingStepLoss[];
  sampleText?: string;
  targets?: string[];
  parameterCounts?: {
    total: number;
    trainable: number;
  };
  memory?: {
    peakBytes: number;
  };
  adapterCheck?: {
    directory: string;
    reloadedMergeTargets: string[];
    trainedSampleText: string;
    reloadedSampleText: string;
    reloadedMergedSampleText: string;
  };
  notes: string[];
};

export type TrainingProofVerificationCheck = {
  id: string;
  passed: boolean;
  message: string;
};

export type TrainingProofVerification = {
  passed: boolean;
  checks: TrainingProofVerificationCheck[];
};

export type TrainingProofReport = {
  source: string;
  quantizedOutputDir: string;
  adapterOutputDir: string;
  datasetSource: TrainingProofArgs["datasetSource"];
  trainLimit: number;
  evalLimit: number;
  batchSize: number;
  steps: number;
  maxSequenceLength: number;
  seed: number;
  dataNotes: string[];
  stages: StageReport[];
  verification?: TrainingProofVerification;
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
