/** One conversational message used for chat-format datasets. */
export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

/** One supervised chat example ending in an assistant response. */
export type ChatExample = {
  messages: readonly ChatMessage[];
};
