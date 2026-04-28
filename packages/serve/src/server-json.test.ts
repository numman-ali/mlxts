import { describe, expect, test } from "bun:test";

import { DEFAULT_SERVE_JSON_BODY_MAX_BYTES, readJson } from "./server-json";

const encoder = new TextEncoder();

function jsonRequest(body: BodyInit, headers?: HeadersInit): Request {
  const init: RequestInit = {
    method: "POST",
    body,
  };
  if (headers !== undefined) {
    init.headers = headers;
  }
  return new Request("http://localhost/v1/chat/completions", {
    ...init,
  });
}

describe("bounded JSON request parsing", () => {
  test("parses JSON bodies within the byte limit", async () => {
    await expect(readJson(jsonRequest('{"ok":true}'), { maxBytes: 16 })).resolves.toEqual({
      ok: true,
    });
  });

  test("rejects declared oversized bodies before reading the stream", async () => {
    await expect(
      readJson(
        jsonRequest("{}", {
          "content-length": String(DEFAULT_SERVE_JSON_BODY_MAX_BYTES + 1),
        }),
      ),
    ).rejects.toThrow("Request body exceeds");
  });

  test("rejects streamed bodies that cross the byte limit", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('{"text":"'));
        controller.enqueue(encoder.encode("too long"));
        controller.enqueue(encoder.encode('"}'));
        controller.close();
      },
    });

    await expect(readJson(jsonRequest(body), { maxBytes: 12 })).rejects.toThrow(
      "Request body exceeds",
    );
  });

  test("rejects malformed or missing JSON bodies", async () => {
    await expect(readJson(jsonRequest("{"), { maxBytes: 16 })).rejects.toThrow("valid JSON");
    await expect(
      readJson(new Request("http://localhost/v1/chat/completions"), { maxBytes: 16 }),
    ).rejects.toThrow("valid JSON");
  });
});
