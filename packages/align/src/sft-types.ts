import type { MxArray, ParameterTree } from "@mlxts/core";
import type { TokenBatch } from "@mlxts/data";
import type { Module } from "@mlxts/nn";
import type { CausalLM } from "@mlxts/transformers";

export type TrainableCausalLM = CausalLM & Module;

export type OptimizerLike = {
  update(model: Module, gradients: ParameterTree): void;
  stateArrays(): MxArray[];
  setLearningRate?: (lr: number) => void;
};

export type SFTTrainOptions = {
  optimizer: OptimizerLike;
  batches: readonly TokenBatch[];
  learningRate?: number;
  maxGradNorm?: number | null;
};
