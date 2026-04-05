import { Template } from "@huggingface/jinja";
import { existsSync, readFileSync } from "fs";

import { inspectSnapshot, resolvePretrainedSnapshot } from "./pretrained/snapshot";
import type { LoadSourceOptions } from "./pretrained/types";

/** A single chat turn used when formatting a model-specific prompt template. */
export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

/** A loaded chat template that can format message history into a model prompt string. */
export type ChatTemplate = {
  readonly template: string;
  format(messages: readonly ChatMessage[], options?: { addGenerationPrompt?: boolean }): string;
};

function tokenString(value: unknown): string | undefined {
  if (typeof value === "string" && value !== "") {
    return value;
  }
  if (
    typeof value === "object" &&
    value !== null &&
    "content" in value &&
    typeof value.content === "string" &&
    value.content !== ""
  ) {
    return value.content;
  }
  return undefined;
}

function chatTemplateString(value: unknown): string | undefined {
  if (typeof value === "string" && value !== "") {
    return value;
  }
  if (Array.isArray(value)) {
    const namedDefault = value.find(
      (entry) =>
        typeof entry === "object" &&
        entry !== null &&
        "name" in entry &&
        entry.name === "default" &&
        "template" in entry &&
        typeof entry.template === "string" &&
        entry.template !== "",
    );
    if (
      namedDefault !== undefined &&
      typeof namedDefault === "object" &&
      namedDefault !== null &&
      "template" in namedDefault &&
      typeof namedDefault.template === "string"
    ) {
      return namedDefault.template;
    }

    const firstTemplate = value.find(
      (entry) =>
        typeof entry === "object" &&
        entry !== null &&
        "template" in entry &&
        typeof entry.template === "string" &&
        entry.template !== "",
    );
    if (
      firstTemplate !== undefined &&
      typeof firstTemplate === "object" &&
      firstTemplate !== null &&
      "template" in firstTemplate &&
      typeof firstTemplate.template === "string"
    ) {
      return firstTemplate.template;
    }
  }
  return undefined;
}

function resolveTemplateText(inspection: ReturnType<typeof inspectSnapshot>): string | null {
  const filePath = inspection.model.chatTemplatePath;
  if (filePath !== undefined && existsSync(filePath)) {
    const template = readFileSync(filePath, "utf8").trim();
    if (template !== "") {
      return template;
    }
  }

  return (
    chatTemplateString(inspection.tokenizerConfig.chat_template) ??
    chatTemplateString(inspection.processorConfig.chat_template) ??
    chatTemplateString(inspection.config.chat_template_jinja) ??
    null
  );
}

/** Load a Hugging Face chat template for a local model directory or repo id. */
export async function loadChatTemplate(
  source: string,
  options: LoadSourceOptions = {},
): Promise<ChatTemplate | null> {
  const snapshot = await resolvePretrainedSnapshot(source, options);
  const inspection = inspectSnapshot(snapshot);
  const templateText = resolveTemplateText(inspection);
  if (templateText === null) {
    return null;
  }

  const bosToken =
    tokenString(inspection.tokenizerConfig.bos_token) ??
    tokenString(inspection.specialTokensMap.bos_token) ??
    "";
  const eosToken =
    tokenString(inspection.tokenizerConfig.eos_token) ??
    tokenString(inspection.specialTokensMap.eos_token) ??
    "";
  const template = new Template(templateText);

  return {
    template: templateText,
    format(messages, renderOptions = {}) {
      return template.render({
        messages,
        bos_token: bosToken,
        eos_token: eosToken,
        add_generation_prompt: renderOptions.addGenerationPrompt ?? true,
      });
    },
  };
}
