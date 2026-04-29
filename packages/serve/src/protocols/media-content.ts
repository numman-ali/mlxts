/**
 * Protocol-neutral media-content normalization for serving adapters.
 * @module
 */

import { isRecord, ServeError } from "../errors";
import type { GenerationContentPart, GenerationMediaSource } from "../types";

/** Return a text part without losing its ordered position among media parts. */
export function textContentPart(text: string): GenerationContentPart {
  return { kind: "text", text };
}

function nonEmptyString(value: unknown, context: string, param: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError(`${context}: expected a non-empty string.`, { param });
  }
  return value;
}

function dataUrlSource(value: string, context: string, param: string): GenerationMediaSource {
  const commaIndex = value.indexOf(",");
  if (commaIndex < 0) {
    throw new ServeError(`${context}: data URLs must include a comma separator.`, { param });
  }
  const header = value.slice("data:".length, commaIndex);
  const data = value.slice(commaIndex + 1);
  const headerParts = header.split(";").filter((entry) => entry !== "");
  const mediaType = headerParts[0];
  if (mediaType === undefined || mediaType === "base64" || !headerParts.includes("base64")) {
    throw new ServeError(`${context}: data URLs must include a media type and base64 data.`, {
      param,
    });
  }
  return { kind: "data", mediaType, data };
}

/** Normalize a URL or data URL string into a media source. */
export function mediaSourceFromUrl(
  value: unknown,
  context: string,
  param: string,
): GenerationMediaSource {
  const url = nonEmptyString(value, context, param);
  return url.startsWith("data:") ? dataUrlSource(url, context, param) : { kind: "url", url };
}

/** Normalize a media data URL when the protocol field cannot be a remote URL. */
export function mediaSourceFromDataUrl(
  value: unknown,
  context: string,
  param: string,
): GenerationMediaSource {
  const url = nonEmptyString(value, context, param);
  if (!url.startsWith("data:")) {
    throw new ServeError(`${context}: expected a data URL.`, { param });
  }
  return dataUrlSource(url, context, param);
}

/** Normalize a file id into a media source owned by the protocol layer. */
export function mediaSourceFromFileId(
  value: unknown,
  context: string,
  param: string,
): GenerationMediaSource {
  return { kind: "file", fileId: nonEmptyString(value, context, param) };
}

function imageDetail(value: unknown, context: string, param: string): "auto" | "low" | "high" {
  switch (value) {
    case "auto":
    case "low":
    case "high":
      return value;
    default:
      throw new ServeError(`${context}: image detail must be "auto", "low", or "high".`, {
        param,
      });
  }
}

/** Normalize OpenAI-style `image_url` payloads into an image content part. */
export function openAIImageContentPart(value: unknown, context: string): GenerationContentPart {
  if (typeof value === "string") {
    return { kind: "image", source: mediaSourceFromUrl(value, context, "messages") };
  }
  if (!isRecord(value)) {
    throw new ServeError(`${context}: image_url must be a string or object.`, {
      param: "messages",
    });
  }
  const source = mediaSourceFromUrl(value.url, context, "messages");
  return {
    kind: "image",
    source,
    ...(value.detail === undefined || value.detail === null
      ? {}
      : { detail: imageDetail(value.detail, context, "messages") }),
  };
}

/** Normalize Anthropic Messages image content blocks into an image content part. */
export function anthropicImageContentPart(value: unknown, context: string): GenerationContentPart {
  if (!isRecord(value)) {
    throw new ServeError(`${context}: image content blocks must be objects.`, {
      param: "messages",
    });
  }
  if (!isRecord(value.source)) {
    throw new ServeError(`${context}: image content blocks require a source object.`, {
      param: "messages",
    });
  }

  switch (value.source.type) {
    case "base64":
      return {
        kind: "image",
        source: {
          kind: "data",
          mediaType: nonEmptyString(value.source.media_type, context, "messages"),
          data: nonEmptyString(value.source.data, context, "messages"),
        },
      };
    case "url": {
      const url = nonEmptyString(value.source.url, context, "messages");
      if (url.startsWith("data:")) {
        throw new ServeError(`${context}: data payloads use source.type "base64".`, {
          param: "messages",
        });
      }
      return {
        kind: "image",
        source: mediaSourceFromUrl(url, context, "messages"),
      };
    }
    case "file":
      return {
        kind: "image",
        source: mediaSourceFromFileId(value.source.file_id, context, "messages"),
      };
    default:
      throw new ServeError(`${context}: image source type must be "base64", "url", or "file".`, {
        param: "messages",
      });
  }
}

/** True when any ordered content part carries media instead of text. */
export function hasMediaContent(parts: readonly GenerationContentPart[]): boolean {
  return parts.some((part) => part.kind !== "text");
}
