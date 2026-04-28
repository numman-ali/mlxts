import { prepareChatSupervisionExamples } from "@mlxts/align";
import {
  type ChatMessage,
  createTrainingProofCorpus,
  loadHuggingFaceRowsDataset,
  loadJsonlDataset,
  parseUltrachatMessagesRow,
  type TokenSupervisionExample,
} from "@mlxts/data";
import type { InteractionProfile } from "@mlxts/transformers";

import { type FinetuneArgs, ULTRACHAT_DATASET } from "./args";
import type { LoadedAssets } from "./types";

function requireChatTemplate(
  profile: InteractionProfile,
): NonNullable<InteractionProfile["chatTemplate"]> {
  if (profile.chatTemplate === null) {
    throw new Error("lora-finetune: source model must provide a chat template.");
  }
  return profile.chatTemplate;
}

export async function loadRawMessages(args: FinetuneArgs): Promise<{
  trainMessages: readonly (readonly ChatMessage[])[];
  evalMessages: readonly (readonly ChatMessage[])[];
  samplePrompt: readonly ChatMessage[];
}> {
  if (args.datasetSource === "tiny") {
    const corpus = createTrainingProofCorpus();
    return {
      trainMessages: Array.from({ length: args.trainLimit }, (_, index) => {
        const example = corpus.supervisionExamples[index % corpus.supervisionExamples.length];
        if (example === undefined) {
          throw new Error("lora-finetune: expected at least one tiny supervision example.");
        }
        return example;
      }),
      evalMessages: Array.from({ length: args.evalLimit }, (_, index) => {
        const example = corpus.supervisionExamples[index % corpus.supervisionExamples.length];
        if (example === undefined) {
          throw new Error("lora-finetune: expected at least one tiny supervision example.");
        }
        return example;
      }),
      samplePrompt: corpus.promptMessages,
    };
  }

  if (args.datasetSource === "jsonl") {
    const dataset = await loadJsonlDataset(args.datasetJsonlPath ?? "", (row) =>
      parseUltrachatMessagesRow(row),
    );
    const items = dataset.items();
    if (items.length < args.trainLimit + args.evalLimit) {
      throw new Error(
        `lora-finetune: jsonl dataset contained only ${items.length} record(s); expected at least ${args.trainLimit + args.evalLimit}.`,
      );
    }
    return {
      trainMessages: items.slice(0, args.trainLimit),
      evalMessages: items.slice(args.trainLimit, args.trainLimit + args.evalLimit),
      samplePrompt: items[0]?.slice(0, -1) ?? createTrainingProofCorpus().promptMessages,
    };
  }

  const trainRows = await loadHuggingFaceRowsDataset({
    dataset: ULTRACHAT_DATASET,
    split: "train_sft",
    length: args.trainLimit,
    parseRow: parseUltrachatMessagesRow,
  });
  const evalRows = await loadHuggingFaceRowsDataset({
    dataset: ULTRACHAT_DATASET,
    split: "test_sft",
    length: args.evalLimit,
    parseRow: parseUltrachatMessagesRow,
  });

  return {
    trainMessages: trainRows.items(),
    evalMessages: evalRows.items(),
    samplePrompt: createTrainingProofCorpus().promptMessages,
  };
}

export function prepareSupervisionExamples(
  rawMessages: readonly (readonly ChatMessage[])[],
  tokenizer: LoadedAssets["tokenizer"],
  profile: InteractionProfile,
  limit: number,
  maxSequenceLength: number,
): TokenSupervisionExample[] {
  const template = requireChatTemplate(profile);
  return prepareChatSupervisionExamples(tokenizer, template, rawMessages, {
    limit,
    maxSequenceLength,
  }).examples;
}
