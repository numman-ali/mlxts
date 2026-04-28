import type { ChatMessage } from "./chat";

/** Small chat corpus used by proof and fine-tuning smoke paths. */
export type TrainingProofCorpus = {
  supervisionExamples: readonly ChatMessage[][];
  promptMessages: readonly ChatMessage[];
  chosen: ChatMessage;
  rejected: ChatMessage;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function expectObject(value: unknown, context: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`${context} must be an object.`);
  }
  return value;
}

function expectString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context} must be a string.`);
  }
  return value;
}

function parseChatRole(value: unknown, context: string): ChatMessage["role"] {
  const role = expectString(value, context);
  if (role === "system" || role === "user" || role === "assistant") {
    return role;
  }
  throw new Error(`${context} must be one of system, user, or assistant.`);
}

function parseChatMessages(value: unknown, context: string): ChatMessage[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context} must be an array of chat messages.`);
  }

  return value.map((entry, index) => {
    const record = expectObject(entry, `${context}[${index}]`);
    return {
      role: parseChatRole(record.role, `${context}[${index}].role`),
      content: expectString(record.content, `${context}[${index}].content`),
    };
  });
}

function expectAssistant(messages: readonly ChatMessage[], context: string): void {
  if (messages.at(-1)?.role !== "assistant") {
    throw new Error(`${context} must end in an assistant message.`);
  }
}

/** Canonical small chat corpus for tiny local proof and example runners. */
export function createTrainingProofCorpus(): TrainingProofCorpus {
  return {
    supervisionExamples: [
      [
        { role: "system", content: "You are concise, exact, and helpful." },
        { role: "user", content: "Explain LoRA in one sentence." },
        {
          role: "assistant",
          content: "LoRA trains a small low-rank adapter instead of updating the entire model.",
        },
      ],
      [
        { role: "system", content: "You are concise, exact, and helpful." },
        { role: "user", content: "Why does quantization help on Apple Silicon?" },
        {
          role: "assistant",
          content: "It lowers memory use and keeps the unified-memory runtime practical.",
        },
      ],
    ],
    promptMessages: [
      { role: "system", content: "You are concise, exact, and helpful." },
      { role: "user", content: "Write a friendly one-sentence greeting." },
    ],
    chosen: {
      role: "assistant",
      content: "You're welcome. Happy to help.",
    },
    rejected: {
      role: "assistant",
      content: "ok",
    },
  };
}

/** Parse one UltraChat row into a supervision-ready chat transcript. */
export function parseUltrachatMessagesRow(row: unknown): readonly ChatMessage[] {
  const record = expectObject(row, "ultrachat row");
  const messages = parseChatMessages(record.messages, "ultrachat row.messages");
  expectAssistant(messages, "ultrachat row.messages");
  return messages;
}
