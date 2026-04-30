/**
 * OpenAI Responses function-call stream helpers.
 * @module
 */

import type { OpenAIChatCompletionStreamToolCall } from "../protocols/openai-chat-completion-streaming";
import type {
  NormalizedOpenAIResponse,
  OpenAIResponseFunctionCallItem,
  OpenAIResponseMessageItem,
  OpenAIResponseOutputItem,
  OpenAIResponseReasoningItem,
} from "../protocols/openai-responses";
import type { StreamObserverChunkKind } from "./runtime";

export type OpenAIResponseOutputState = {
  id: string;
  outputIndex: number;
};

export type OpenAIResponseFunctionCallState = OpenAIResponseOutputState & {
  callId: string;
  name: string;
  arguments: string;
};

export type OpenAIResponseFunctionCallStore = {
  outputCount: number;
  functionCalls: OpenAIResponseFunctionCallState[];
};

export function hasOpenAIResponseStreamingToolOutput(response: NormalizedOpenAIResponse): boolean {
  return (
    (response.request.input.kind === "messages" || response.request.input.kind === "content") &&
    response.request.input.tools !== undefined &&
    response.request.input.tools.length > 0
  );
}

export function openAIResponseFunctionCallItem(
  state: OpenAIResponseFunctionCallState,
): OpenAIResponseFunctionCallItem {
  return {
    id: state.id,
    type: "function_call",
    status: "completed",
    call_id: state.callId,
    name: state.name,
    arguments: state.arguments,
  };
}

export function emitOpenAIResponseFunctionCall(
  store: OpenAIResponseFunctionCallStore,
  emit: (event: string, payload: Record<string, unknown>, kind?: StreamObserverChunkKind) => void,
  options: { id: string },
  toolCall: OpenAIChatCompletionStreamToolCall,
): void {
  const index = store.functionCalls.length;
  const item: OpenAIResponseFunctionCallState = {
    id: `${options.id}-fc-${index + 1}`,
    outputIndex: store.outputCount,
    callId: toolCall.id ?? `call_${index + 1}`,
    name: toolCall.function?.name ?? "",
    arguments: toolCall.function?.arguments ?? "",
  };
  store.outputCount += 1;
  store.functionCalls.push(item);
  emit("response.output_item.added", {
    response_id: options.id,
    output_index: item.outputIndex,
    item: {
      id: item.id,
      type: "function_call",
      status: "in_progress",
      call_id: item.callId,
      name: item.name,
      arguments: "",
    },
  });
  if (item.arguments !== "") {
    emit(
      "response.function_call_arguments.delta",
      {
        response_id: options.id,
        item_id: item.id,
        output_index: item.outputIndex,
        delta: item.arguments,
      },
      "output",
    );
  }
  emit("response.function_call_arguments.done", {
    response_id: options.id,
    item_id: item.id,
    output_index: item.outputIndex,
    call_id: item.callId,
    name: item.name,
    arguments: item.arguments,
  });
  emit("response.output_item.done", {
    response_id: options.id,
    output_index: item.outputIndex,
    item: openAIResponseFunctionCallItem(item),
  });
}

export function finalOpenAIResponseOutputItems(state: {
  reasoningItem?: OpenAIResponseOutputState;
  messageItem?: OpenAIResponseOutputState;
  reasoningText: string;
  visibleText: string;
  functionCalls: OpenAIResponseFunctionCallState[];
}): OpenAIResponseOutputItem[] {
  const items: OpenAIResponseOutputItem[] = [];
  if (state.reasoningItem !== undefined) {
    const reasoning: OpenAIResponseReasoningItem = {
      id: state.reasoningItem.id,
      type: "reasoning",
      status: "completed",
      summary: [],
      content: [{ type: "reasoning_text", text: state.reasoningText }],
    };
    items.push(reasoning);
  }
  if (state.messageItem !== undefined) {
    const message: OpenAIResponseMessageItem = {
      id: state.messageItem.id,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: state.visibleText, annotations: [] }],
    };
    items.push(message);
  }
  items.push(...state.functionCalls.map(openAIResponseFunctionCallItem));
  return items;
}
