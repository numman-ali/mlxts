/**
 * Shared protocol helpers for reasoning-tag normalization.
 * @module
 */

/** Visible assistant content split from optional model-native reasoning text. */
export type ReasoningText = {
  content: string;
  reasoningContent?: string;
};

/** Incremental visible or reasoning text produced while parsing a stream. */
export type ReasoningTextDelta = {
  content?: string;
  reasoningContent?: string;
};

type ReasoningTag = {
  open: string;
  close: string;
};

type ReasoningMarker = {
  tag: ReasoningTag;
  index: number;
};

const REASONING_TAGS: ReasoningTag[] = [
  { open: "<think>", close: "</think>" },
  { open: "<antThinking>", close: "</antThinking>" },
  { open: "<|channel>thought\n", close: "<channel|>" },
];

const MAX_REASONING_TAG_LENGTH = Math.max(
  ...REASONING_TAGS.flatMap((tag) => [tag.open.length, tag.close.length]),
);

function earlierMarker(current: ReasoningMarker | null, marker: ReasoningMarker): ReasoningMarker {
  return current === null || marker.index < current.index ? marker : current;
}

function firstOpenMarker(text: string): ReasoningMarker | null {
  let marker: ReasoningMarker | null = null;
  for (const tag of REASONING_TAGS) {
    const index = text.indexOf(tag.open);
    if (index >= 0) {
      marker = earlierMarker(marker, { tag, index });
    }
  }
  return marker;
}

function firstCloseMarker(text: string): ReasoningMarker | null {
  let marker: ReasoningMarker | null = null;
  for (const tag of REASONING_TAGS) {
    const index = text.indexOf(tag.close);
    if (index >= 0) {
      marker = earlierMarker(marker, { tag, index });
    }
  }
  return marker;
}

function reasoningSplit(content: string, reasoning: string): ReasoningText {
  const reasoningContent = reasoning.trim();
  return reasoningContent === "" ? { content } : { content, reasoningContent };
}

/** Split known model reasoning tags out of visible assistant text. */
export function splitReasoningTags(text: string): ReasoningText {
  const open = firstOpenMarker(text);
  const close = firstCloseMarker(text);
  if (open === null && close === null) {
    return { content: text.trim() };
  }

  if (close !== null && (open === null || close.index < open.index)) {
    const content = text
      .slice(close.index + close.tag.close.length)
      .trimStart()
      .trim();
    return reasoningSplit(content, text.slice(0, close.index));
  }

  if (open === null) {
    return { content: text.trim() };
  }

  const closeIndex = text.indexOf(open.tag.close, open.index + open.tag.open.length);
  if (closeIndex >= 0) {
    const contentPrefix = open.index > 0 ? text.slice(0, open.index).trimEnd() : "";
    const contentSuffix = text.slice(closeIndex + open.tag.close.length).trimStart();
    const content =
      contentPrefix === "" ? contentSuffix.trim() : `${contentPrefix}\n${contentSuffix}`.trim();
    return reasoningSplit(content, text.slice(open.index + open.tag.open.length, closeIndex));
  }

  return reasoningSplit(
    text.slice(0, open.index).trimEnd(),
    text.slice(open.index + open.tag.open.length),
  );
}

/** Split known model reasoning tags out of fallback assistant content. */
export function cleanReasoningFromText(text: string): ReasoningText {
  return splitReasoningTags(text);
}

type ReasoningStreamState = {
  buffer: string;
  mode: "content" | "reasoning";
  activeCloseTag: string | null;
  trimLeadingContent: boolean;
};

function pushDelta(
  deltas: ReasoningTextDelta[],
  key: keyof ReasoningTextDelta,
  text: string,
): void {
  if (text === "") {
    return;
  }
  deltas.push(key === "content" ? { content: text } : { reasoningContent: text });
}

function pushContentDelta(
  state: ReasoningStreamState,
  deltas: ReasoningTextDelta[],
  text: string,
): void {
  if (!state.trimLeadingContent) {
    pushDelta(deltas, "content", text);
    return;
  }

  const trimmed = text.trimStart();
  if (trimmed === "") {
    return;
  }
  state.trimLeadingContent = false;
  pushDelta(deltas, "content", trimmed);
}

function flushTailLength(buffer: string, tagLength: number = MAX_REASONING_TAG_LENGTH): number {
  return Math.max(0, buffer.length - (tagLength - 1));
}

function consumeContentBuffer(state: ReasoningStreamState, deltas: ReasoningTextDelta[]): boolean {
  const open = firstOpenMarker(state.buffer);
  const close = firstCloseMarker(state.buffer);
  if (open === null && close === null) {
    const safeLength = flushTailLength(state.buffer);
    if (safeLength === 0) {
      return true;
    }
    pushContentDelta(state, deltas, state.buffer.slice(0, safeLength));
    state.buffer = state.buffer.slice(safeLength);
    return true;
  }

  if (close !== null && (open === null || close.index < open.index)) {
    pushDelta(deltas, "reasoningContent", state.buffer.slice(0, close.index));
    state.buffer = state.buffer.slice(close.index + close.tag.close.length);
    state.trimLeadingContent = true;
    return false;
  }

  if (open === null) {
    return true;
  }

  pushContentDelta(state, deltas, state.buffer.slice(0, open.index));
  state.buffer = state.buffer.slice(open.index + open.tag.open.length);
  state.mode = "reasoning";
  state.activeCloseTag = open.tag.close;
  return false;
}

function consumeReasoningBuffer(
  state: ReasoningStreamState,
  deltas: ReasoningTextDelta[],
): boolean {
  const closeIndex =
    state.activeCloseTag === null ? -1 : state.buffer.indexOf(state.activeCloseTag);
  if (closeIndex < 0) {
    const safeLength = flushTailLength(
      state.buffer,
      state.activeCloseTag?.length ?? MAX_REASONING_TAG_LENGTH,
    );
    if (safeLength === 0) {
      return true;
    }
    pushDelta(deltas, "reasoningContent", state.buffer.slice(0, safeLength));
    state.buffer = state.buffer.slice(safeLength);
    return true;
  }

  pushDelta(deltas, "reasoningContent", state.buffer.slice(0, closeIndex));
  state.buffer = state.buffer.slice(closeIndex + (state.activeCloseTag?.length ?? 0));
  state.mode = "content";
  state.activeCloseTag = null;
  state.trimLeadingContent = true;
  return false;
}

function appendReasoningStreamChunk(
  state: ReasoningStreamState,
  text: string,
): ReasoningTextDelta[] {
  state.buffer += text;
  const deltas: ReasoningTextDelta[] = [];

  while (state.buffer !== "") {
    const stalled =
      state.mode === "content"
        ? consumeContentBuffer(state, deltas)
        : consumeReasoningBuffer(state, deltas);
    if (stalled) {
      break;
    }
  }

  return deltas;
}

function flushReasoningStream(state: ReasoningStreamState): ReasoningTextDelta[] {
  if (state.buffer === "") {
    return [];
  }

  const deltas: ReasoningTextDelta[] = [];
  if (state.mode === "content") {
    pushContentDelta(state, deltas, state.buffer);
  } else {
    pushDelta(deltas, "reasoningContent", state.buffer);
  }
  state.buffer = "";
  return deltas;
}

/** Create a streaming splitter for known model reasoning tags. */
export function createReasoningTagStream(): {
  push(text: string): ReasoningTextDelta[];
  finish(): ReasoningTextDelta[];
} {
  const state: ReasoningStreamState = {
    buffer: "",
    mode: "content",
    activeCloseTag: null,
    trimLeadingContent: false,
  };
  return {
    push(text) {
      return appendReasoningStreamChunk(state, text);
    },
    finish() {
      return flushReasoningStream(state);
    },
  };
}
