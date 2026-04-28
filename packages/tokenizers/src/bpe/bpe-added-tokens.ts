import type { AddedToken } from "./bpe-base";

type InputChunk =
  | {
      kind: "text";
      text: string;
      start: number;
      end: number;
    }
  | {
      kind: "added-token";
      token: AddedToken;
      start: number;
      end: number;
    };

export function sortAddedTokenMatches(addedTokens: readonly AddedToken[]): AddedToken[] {
  return [...addedTokens]
    .filter((token) => token.content !== "")
    .toSorted((left, right) => right.content.length - left.content.length);
}

function matchAddedToken(
  text: string,
  start: number,
  addedTokens: readonly AddedToken[],
): AddedToken | null {
  for (const token of addedTokens) {
    if (text.startsWith(token.content, start)) {
      return token;
    }
  }
  return null;
}

export function splitInputByAddedTokens(
  text: string,
  addedTokens: readonly AddedToken[],
): InputChunk[] {
  if (text === "" || addedTokens.length === 0) {
    return text === "" ? [] : [{ kind: "text", text, start: 0, end: text.length }];
  }

  const chunks: InputChunk[] = [];
  let cursor = 0;
  let plainStart = 0;

  while (cursor < text.length) {
    const matchedToken = matchAddedToken(text, cursor, addedTokens);
    if (matchedToken === null) {
      cursor += 1;
      continue;
    }

    if (plainStart < cursor) {
      chunks.push({
        kind: "text",
        text: text.slice(plainStart, cursor),
        start: plainStart,
        end: cursor,
      });
    }

    const end = cursor + matchedToken.content.length;
    chunks.push({
      kind: "added-token",
      token: matchedToken,
      start: cursor,
      end,
    });
    cursor = end;
    plainStart = end;
  }

  if (plainStart < text.length) {
    chunks.push({
      kind: "text",
      text: text.slice(plainStart),
      start: plainStart,
      end: text.length,
    });
  }

  return chunks;
}
