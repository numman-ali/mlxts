import {
  type ChatPreparationStats,
  prepareChatPreferenceExamples,
  prepareChatSupervisionExamples,
} from "@mlxts/align";
import type { PreferenceExample, TokenSupervisionExample } from "@mlxts/data";
import type { InteractionProfile } from "@mlxts/transformers";

import type { TrainingProofArgs } from "./args";
import { loadTrainingProofRawDatasets, type TrainingProofRawDatasets } from "./datasets";
import type { LoadedAssets, PreparedTrainingProofData } from "./types";

function requireChatTemplate(
  profile: InteractionProfile,
): NonNullable<InteractionProfile["chatTemplate"]> {
  if (profile.chatTemplate === null) {
    throw new Error("Training proof requires an instruct checkpoint with a chat template.");
  }
  return profile.chatTemplate;
}

function appendPreparationNotes(
  notes: string[],
  label: string,
  kind: "supervision" | "preference",
  stats: ChatPreparationStats,
): void {
  notes.push(`${label}_${kind}_kept=${stats.kept}`);
  notes.push(`${label}_${kind}_skipped_malformed=${stats.skippedMalformed}`);
  notes.push(`${label}_${kind}_skipped_long=${stats.skippedLong}`);
}

function buildPreparedSupervisionExamples(
  rawMessages: readonly TrainingProofRawDatasets["supervisionTrainMessages"],
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  limit: number,
  maxSequenceLength: number,
  label: string,
  notes: string[],
): TokenSupervisionExample[] {
  const template = requireChatTemplate(profile);
  const prepared = prepareChatSupervisionExamples(tokenizer, template, rawMessages, {
    limit,
    maxSequenceLength,
  });
  appendPreparationNotes(notes, label, "supervision", prepared.stats);
  return prepared.examples;
}

function buildPreparedPreferenceExamples(
  rawRows: readonly TrainingProofRawDatasets["preferenceTrainRows"],
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  limit: number,
  maxSequenceLength: number,
  label: string,
  notes: string[],
): PreferenceExample[] {
  const template = requireChatTemplate(profile);
  const prepared = prepareChatPreferenceExamples(tokenizer, template, rawRows, {
    limit,
    maxSequenceLength,
  });
  appendPreparationNotes(notes, label, "preference", prepared.stats);
  return prepared.examples;
}

export async function prepareTrainingProofData(
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  args: TrainingProofArgs,
): Promise<PreparedTrainingProofData> {
  const rawDatasets = await loadTrainingProofRawDatasets(args);
  const notes = [...rawDatasets.notes];

  return {
    supervisionTrain: buildPreparedSupervisionExamples(
      rawDatasets.supervisionTrainMessages,
      tokenizer,
      profile,
      args.trainLimit,
      args.maxSequenceLength,
      "train",
      notes,
    ),
    supervisionEval: buildPreparedSupervisionExamples(
      rawDatasets.supervisionEvalMessages,
      tokenizer,
      profile,
      args.evalLimit,
      args.maxSequenceLength,
      "eval",
      notes,
    ),
    preferenceTrain: buildPreparedPreferenceExamples(
      rawDatasets.preferenceTrainRows,
      tokenizer,
      profile,
      args.trainLimit,
      args.maxSequenceLength,
      "train",
      notes,
    ),
    preferenceEval: buildPreparedPreferenceExamples(
      rawDatasets.preferenceEvalRows,
      tokenizer,
      profile,
      args.evalLimit,
      args.maxSequenceLength,
      "eval",
      notes,
    ),
    samplePromptMessages: rawDatasets.samplePromptMessages,
    notes,
  };
}
