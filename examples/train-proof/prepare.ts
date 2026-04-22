import { buildChatPreferenceExample, buildChatSupervisionExample } from "@mlxts/align";
import type { ChatMessage, PreferenceExample, TokenSupervisionExample } from "@mlxts/data";
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

function buildPreparedSupervisionExamples(
  rawMessages: readonly (readonly ChatMessage[])[],
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  limit: number,
  maxSequenceLength: number,
  label: string,
  notes: string[],
): TokenSupervisionExample[] {
  const template = requireChatTemplate(profile);
  const prepared: TokenSupervisionExample[] = [];
  let skippedMalformed = 0;
  let skippedLong = 0;

  for (const messages of rawMessages) {
    try {
      const example = buildChatSupervisionExample(tokenizer, template, messages);
      if (example.inputIds.length > maxSequenceLength) {
        skippedLong += 1;
        continue;
      }
      prepared.push(example);
      if (prepared.length === limit) {
        break;
      }
    } catch {
      skippedMalformed += 1;
    }
  }

  if (prepared.length < limit) {
    throw new Error(
      `training proof: collected only ${prepared.length} ${label} supervision example(s); expected ${limit}.`,
    );
  }

  notes.push(`${label}_supervision_kept=${prepared.length}`);
  notes.push(`${label}_supervision_skipped_malformed=${skippedMalformed}`);
  notes.push(`${label}_supervision_skipped_long=${skippedLong}`);
  return prepared;
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
  const prepared: PreferenceExample[] = [];
  let skippedMalformed = 0;
  let skippedLong = 0;

  for (const row of rawRows) {
    try {
      const example = buildChatPreferenceExample(
        tokenizer,
        template,
        row.promptMessages,
        row.chosen,
        row.rejected,
      );
      const chosenLength = example.promptIds.length + example.chosenIds.length;
      const rejectedLength = example.promptIds.length + example.rejectedIds.length;
      if (Math.max(chosenLength, rejectedLength) > maxSequenceLength) {
        skippedLong += 1;
        continue;
      }
      prepared.push(example);
      if (prepared.length === limit) {
        break;
      }
    } catch {
      skippedMalformed += 1;
    }
  }

  if (prepared.length < limit) {
    throw new Error(
      `training proof: collected only ${prepared.length} ${label} preference example(s); expected ${limit}.`,
    );
  }

  notes.push(`${label}_preference_kept=${prepared.length}`);
  notes.push(`${label}_preference_skipped_malformed=${skippedMalformed}`);
  notes.push(`${label}_preference_skipped_long=${skippedLong}`);
  return prepared;
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
