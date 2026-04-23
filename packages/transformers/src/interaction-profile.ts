/**
 * Shared prompt compilation for completion and chat-capable checkpoints.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";

import {
  type ChatMessage,
  type ChatTemplate,
  type ChatTool,
  loadChatTemplate,
} from "./chat-template";
import type { LoadSourceOptions } from "./types";

/** A compiled prompt string plus the exact token IDs that will be fed to generation. */
export type PromptCompilation = {
  text: string;
  tokenIds: number[];
};

/** Options for compiling a plain completion-style prompt. */
export type CompileTextPromptOptions = {
  addSpecialTokens?: boolean;
};

/** Options for compiling chat messages through a checkpoint chat template. */
export type CompileChatPromptOptions = CompileTextPromptOptions & {
  addGenerationPrompt?: boolean;
  tools?: readonly ChatTool[];
  enableThinking?: boolean;
  preserveThinking?: boolean;
};

/** Shared prompt-compilation profile for completion and chat-capable checkpoints. */
export type InteractionProfile = {
  readonly kind: "completion" | "chat";
  readonly chatTemplate: ChatTemplate | null;
  compileTextPrompt(
    tokenizer: Tokenizer,
    prompt: string,
    options?: CompileTextPromptOptions,
  ): PromptCompilation;
  compileMessages(
    tokenizer: Tokenizer,
    messages: readonly ChatMessage[],
    options?: CompileChatPromptOptions,
  ): PromptCompilation;
};

function encodePrompt(
  tokenizer: Tokenizer,
  text: string,
  addSpecialTokens: boolean | undefined,
): PromptCompilation {
  return {
    text,
    tokenIds: tokenizer.encode(text, addSpecialTokens === undefined ? {} : { addSpecialTokens }),
  };
}

/** Create an interaction profile from an optional loaded chat template. */
export function createInteractionProfile(chatTemplate: ChatTemplate | null): InteractionProfile {
  return {
    kind: chatTemplate === null ? "completion" : "chat",
    chatTemplate,
    compileTextPrompt(tokenizer, prompt, options = {}) {
      return encodePrompt(tokenizer, prompt, options.addSpecialTokens ?? true);
    },
    compileMessages(tokenizer, messages, options = {}) {
      if (chatTemplate === null) {
        throw new Error(
          "interaction profile: this checkpoint does not define a chat template, so messages cannot be compiled directly.",
        );
      }

      const text = chatTemplate.format(messages, {
        ...(options.addGenerationPrompt === undefined
          ? {}
          : { addGenerationPrompt: options.addGenerationPrompt }),
        ...(options.tools === undefined ? {} : { tools: options.tools }),
        ...(options.enableThinking === undefined ? {} : { enableThinking: options.enableThinking }),
        ...(options.preserveThinking === undefined
          ? {}
          : { preserveThinking: options.preserveThinking }),
      });
      return encodePrompt(tokenizer, text, options.addSpecialTokens ?? true);
    },
  };
}

/** Load the shared prompt-compilation profile for a local model directory or repo id. */
export async function loadInteractionProfile(
  source: string,
  options: LoadSourceOptions = {},
): Promise<InteractionProfile> {
  const chatTemplate = await loadChatTemplate(source, options);
  return createInteractionProfile(chatTemplate);
}
