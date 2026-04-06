import type { ChatMessage, PreferenceExample, TokenSupervisionExample } from "@mlxts/data";
import type { Tokenizer } from "@mlxts/tokenizers";
import type { ChatTemplate } from "@mlxts/transformers";

export type RenderableChatTemplate =
  | ChatTemplate
  | {
      format(messages: readonly ChatMessage[], options?: { addGenerationPrompt?: boolean }): string;
    };

function renderWithTemplate(
  template: RenderableChatTemplate,
  messages: readonly ChatMessage[],
  addGenerationPrompt: boolean,
): string {
  return template.format(messages, { addGenerationPrompt });
}

function assertAssistantTurn(message: ChatMessage | undefined): ChatMessage {
  if (message === undefined || message.role !== "assistant") {
    throw new Error("align: chat supervision examples must end in an assistant message.");
  }
  return message;
}

function suffixIds(fullIds: readonly number[], prefixIds: readonly number[]): number[] {
  if (fullIds.length < prefixIds.length) {
    throw new Error("align: full chat rendering must be at least as long as the prompt prefix.");
  }
  for (let index = 0; index < prefixIds.length; index += 1) {
    if (fullIds[index] !== prefixIds[index]) {
      throw new Error(
        "align: full chat rendering did not preserve the prompt prefix tokenization.",
      );
    }
  }
  return fullIds.slice(prefixIds.length);
}

type SplitRenderedTokens = {
  promptIds: number[];
  completionIds: number[];
  fullIds: number[];
  completionStartIndex: number;
};

function splitRenderedCompletion(
  tokenizer: Tokenizer,
  promptText: string,
  fullText: string,
): SplitRenderedTokens {
  if (!fullText.startsWith(promptText)) {
    throw new Error("align: full chat rendering did not preserve the prompt prefix text.");
  }

  const boundary = promptText.length;
  const encoding = tokenizer.encodeWithOffsets(fullText, {
    addSpecialTokens: false,
    returnOffsets: true,
  });
  const fullIds = encoding.ids;
  const offsets = encoding.offsets;
  if (offsets === undefined) {
    const promptIds = tokenizer.encode(promptText, { addSpecialTokens: false });
    return {
      promptIds,
      completionIds: suffixIds(fullIds, promptIds),
      fullIds,
      completionStartIndex: promptIds.length,
    };
  }

  const completionStartIndex = offsets.findIndex((offset) => offset.end > boundary);
  if (completionStartIndex === -1) {
    throw new Error("align: assistant completion produced no trainable tokens.");
  }

  return {
    promptIds: fullIds.slice(0, completionStartIndex),
    completionIds: fullIds.slice(completionStartIndex),
    fullIds,
    completionStartIndex,
  };
}

/** Render a chat transcript with a loaded model template. */
export function renderChatMessages(
  template: RenderableChatTemplate,
  messages: readonly ChatMessage[],
  options: { addGenerationPrompt?: boolean } = {},
): string {
  return renderWithTemplate(template, messages, options.addGenerationPrompt ?? true);
}

/** Build a token-level supervision example from a final assistant turn. */
export function buildChatSupervisionExample(
  tokenizer: Tokenizer,
  template: RenderableChatTemplate,
  messages: readonly ChatMessage[],
): TokenSupervisionExample {
  const assistant = assertAssistantTurn(messages.at(-1));
  const promptMessages = messages.slice(0, -1);
  const promptText = renderWithTemplate(template, promptMessages, true);
  const fullText = renderWithTemplate(template, [...promptMessages, assistant], false);
  const { fullIds, completionIds, completionStartIndex } = splitRenderedCompletion(
    tokenizer,
    promptText,
    fullText,
  );
  if (completionIds.length === 0) {
    throw new Error("align: assistant completion produced no trainable tokens.");
  }

  const inputIds = fullIds.slice(0, -1);
  const targetIds = fullIds.slice(1);
  const firstTrainableTargetIndex = Math.max(0, completionStartIndex - 1);
  const lossMask = targetIds.map((_, index) => (index >= firstTrainableTargetIndex ? 1 : 0));
  return {
    inputIds,
    targetIds,
    lossMask,
  };
}

/** Build a token-level preference example from chosen and rejected replies. */
export function buildChatPreferenceExample(
  tokenizer: Tokenizer,
  template: RenderableChatTemplate,
  promptMessages: readonly ChatMessage[],
  chosen: ChatMessage,
  rejected: ChatMessage,
): PreferenceExample {
  if (chosen.role !== "assistant" || rejected.role !== "assistant") {
    throw new Error("align: chosen and rejected replies must both be assistant messages.");
  }

  const promptText = renderWithTemplate(template, promptMessages, true);
  const chosenSplit = splitRenderedCompletion(
    tokenizer,
    promptText,
    renderWithTemplate(template, [...promptMessages, chosen], false),
  );
  const rejectedSplit = splitRenderedCompletion(
    tokenizer,
    promptText,
    renderWithTemplate(template, [...promptMessages, rejected], false),
  );
  if (chosenSplit.promptIds.length !== rejectedSplit.promptIds.length) {
    throw new Error("align: chosen and rejected prompt prefixes diverged after tokenization.");
  }
  return {
    promptIds: chosenSplit.promptIds,
    chosenIds: chosenSplit.completionIds,
    rejectedIds: rejectedSplit.completionIds,
  };
}
