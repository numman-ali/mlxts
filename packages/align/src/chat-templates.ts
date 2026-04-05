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
  const promptIds = tokenizer.encode(renderWithTemplate(template, promptMessages, true), {
    addSpecialTokens: false,
  });
  const fullIds = tokenizer.encode(
    renderWithTemplate(template, [...promptMessages, assistant], false),
    {
      addSpecialTokens: false,
    },
  );
  const completionIds = suffixIds(fullIds, promptIds);
  if (completionIds.length === 0) {
    throw new Error("align: assistant completion produced no trainable tokens.");
  }

  const inputIds = fullIds.slice(0, -1);
  const targetIds = fullIds.slice(1);
  const lossMask = targetIds.map((_, index) => (index >= promptIds.length - 1 ? 1 : 0));
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

  const promptIds = tokenizer.encode(renderWithTemplate(template, promptMessages, true), {
    addSpecialTokens: false,
  });
  const chosenIds = suffixIds(
    tokenizer.encode(renderWithTemplate(template, [...promptMessages, chosen], false), {
      addSpecialTokens: false,
    }),
    promptIds,
  );
  const rejectedIds = suffixIds(
    tokenizer.encode(renderWithTemplate(template, [...promptMessages, rejected], false), {
      addSpecialTokens: false,
    }),
    promptIds,
  );
  return {
    promptIds,
    chosenIds,
    rejectedIds,
  };
}
