import type { ChatMessage, PreferenceExample, TokenSupervisionExample } from "@mlxts/data";
import type { Tokenizer } from "@mlxts/tokenizers";

import {
  buildChatPreferenceExample,
  buildChatSupervisionExample,
  type RenderableChatTemplate,
} from "./chat-templates";

/** One prompt plus chosen/rejected assistant replies for preference training. */
export type ChatPreferenceConversation = {
  promptMessages: readonly ChatMessage[];
  chosen: ChatMessage;
  rejected: ChatMessage;
};

/** Counts gathered while normalizing raw chat rows into trainable examples. */
export type ChatPreparationStats = {
  kept: number;
  skippedMalformed: number;
  skippedLong: number;
};

/** One prepared example slice plus the normalization stats that produced it. */
export type PreparedExamplesResult<T> = {
  examples: T[];
  stats: ChatPreparationStats;
};

type ChatPreparationOptions = {
  limit: number;
  maxSequenceLength: number;
};

function validatePreparationOptions(options: ChatPreparationOptions, context: string): void {
  if (!Number.isInteger(options.limit) || options.limit < 1) {
    throw new Error(`${context}: limit must be a positive integer.`);
  }
  if (!Number.isInteger(options.maxSequenceLength) || options.maxSequenceLength < 1) {
    throw new Error(`${context}: maxSequenceLength must be a positive integer.`);
  }
}

function createEmptyPreparationStats(): ChatPreparationStats {
  return {
    kept: 0,
    skippedMalformed: 0,
    skippedLong: 0,
  };
}

function finalizePreparedExamples<T>(
  examples: T[],
  stats: ChatPreparationStats,
  limit: number,
  context: string,
): PreparedExamplesResult<T> {
  if (examples.length < limit) {
    throw new Error(`${context}: collected only ${examples.length} example(s); expected ${limit}.`);
  }
  stats.kept = examples.length;
  return { examples, stats };
}

/** Prepare supervision examples from chat transcripts, skipping malformed or overlong rows. */
export function prepareChatSupervisionExamples(
  tokenizer: Tokenizer,
  template: RenderableChatTemplate,
  rawMessages: readonly (readonly ChatMessage[])[],
  options: ChatPreparationOptions,
): PreparedExamplesResult<TokenSupervisionExample> {
  validatePreparationOptions(options, "align.prepareChatSupervisionExamples");

  const prepared: TokenSupervisionExample[] = [];
  const stats = createEmptyPreparationStats();
  for (const messages of rawMessages) {
    try {
      const example = buildChatSupervisionExample(tokenizer, template, messages);
      if (example.inputIds.length > options.maxSequenceLength) {
        stats.skippedLong += 1;
        continue;
      }
      prepared.push(example);
      if (prepared.length === options.limit) {
        break;
      }
    } catch {
      stats.skippedMalformed += 1;
    }
  }

  return finalizePreparedExamples(
    prepared,
    stats,
    options.limit,
    "align.prepareChatSupervisionExamples",
  );
}

/** Prepare preference examples from prompt plus chosen/rejected assistant replies. */
export function prepareChatPreferenceExamples(
  tokenizer: Tokenizer,
  template: RenderableChatTemplate,
  rawRows: readonly ChatPreferenceConversation[],
  options: ChatPreparationOptions,
): PreparedExamplesResult<PreferenceExample> {
  validatePreparationOptions(options, "align.prepareChatPreferenceExamples");

  const prepared: PreferenceExample[] = [];
  const stats = createEmptyPreparationStats();
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
      if (Math.max(chosenLength, rejectedLength) > options.maxSequenceLength) {
        stats.skippedLong += 1;
        continue;
      }
      prepared.push(example);
      if (prepared.length === options.limit) {
        break;
      }
    } catch {
      stats.skippedMalformed += 1;
    }
  }

  return finalizePreparedExamples(
    prepared,
    stats,
    options.limit,
    "align.prepareChatPreferenceExamples",
  );
}
