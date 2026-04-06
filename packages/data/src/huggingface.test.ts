import { afterEach, describe, expect, test } from "bun:test";

import { loadHuggingFaceRowsDataset } from "./huggingface";

const servers: Array<ReturnType<typeof Bun.serve>> = [];

function createServer(handler: (request: Request) => Response | Promise<Response>): {
  baseUrl: string;
  requests: string[];
} {
  const requests: string[] = [];
  const server = Bun.serve({
    port: 0,
    fetch(request) {
      requests.push(request.url);
      return handler(request);
    },
  });
  servers.push(server);
  return {
    baseUrl: server.url.toString().replace(/\/$/, ""),
    requests,
  };
}

afterEach(() => {
  while (servers.length > 0) {
    servers.pop()?.stop(true);
  }
});

describe("loadHuggingFaceRowsDataset", () => {
  test("loads a deterministic slice across paginated row requests", async () => {
    const { baseUrl, requests } = createServer((request) => {
      const url = new URL(request.url);
      const offset = Number(url.searchParams.get("offset") ?? "0");
      const length = Number(url.searchParams.get("length") ?? "0");
      return Response.json({
        rows: Array.from({ length }, (_, index) => ({
          row_idx: offset + index,
          row: { value: offset + index },
        })),
      });
    });

    const dataset = await loadHuggingFaceRowsDataset({
      dataset: "demo/proof",
      split: "train",
      baseUrl,
      pageSize: 2,
      length: 5,
      parseRow(row) {
        if (typeof row !== "object" || row === null || !("value" in row)) {
          throw new Error("bad row");
        }
        return row.value;
      },
    });

    expect(dataset.items()).toEqual([0, 1, 2, 3, 4]);
    expect(requests).toHaveLength(3);
    expect(requests[0]).toContain("offset=0");
    expect(requests[1]).toContain("offset=2");
    expect(requests[2]).toContain("offset=4");
  });

  test("surfaces dataset server errors", async () => {
    const { baseUrl } = createServer(() => Response.json({ error: "split missing" }));

    await expect(
      loadHuggingFaceRowsDataset({
        dataset: "demo/proof",
        split: "train",
        baseUrl,
        length: 1,
        parseRow(row) {
          return row;
        },
      }),
    ).rejects.toThrow('returned "split missing"');
  });

  test("rejects invalid length and offset inputs", async () => {
    await expect(
      loadHuggingFaceRowsDataset({
        dataset: "demo/proof",
        split: "train",
        length: 0,
        parseRow(row) {
          return row;
        },
      }),
    ).rejects.toThrow("length must be a positive integer");

    await expect(
      loadHuggingFaceRowsDataset({
        dataset: "demo/proof",
        split: "train",
        length: 1,
        offset: -1,
        parseRow(row) {
          return row;
        },
      }),
    ).rejects.toThrow("offset must be a non-negative integer");
  });

  test("surfaces malformed JSON payloads and missing rows", async () => {
    const malformed = createServer(() => Response.json(["bad"]));
    await expect(
      loadHuggingFaceRowsDataset({
        dataset: "demo/proof",
        split: "train",
        baseUrl: malformed.baseUrl,
        length: 1,
        parseRow(row) {
          return row;
        },
      }),
    ).rejects.toThrow("expected an object response");

    const missingRows = createServer(() => Response.json({ ok: true }));
    await expect(
      loadHuggingFaceRowsDataset({
        dataset: "demo/proof",
        split: "train",
        baseUrl: missingRows.baseUrl,
        length: 1,
        parseRow(row) {
          return row;
        },
      }),
    ).rejects.toThrow("did not include rows");
  });

  test("retries transient upstream failures", async () => {
    let attempts = 0;
    const { baseUrl } = createServer(() => {
      attempts += 1;
      if (attempts < 3) {
        return new Response("bad gateway", { status: 502, statusText: "Bad Gateway" });
      }
      return Response.json({
        rows: [{ row_idx: 0, row: { value: 1 } }],
      });
    });

    const dataset = await loadHuggingFaceRowsDataset({
      dataset: "demo/proof",
      split: "train",
      baseUrl,
      length: 1,
      parseRow(row) {
        return row;
      },
    });

    expect(dataset.length).toBe(1);
    expect(attempts).toBeGreaterThanOrEqual(3);
  });

  test("does not retry non-retryable HTTP failures", async () => {
    let attempts = 0;
    const { baseUrl } = createServer(() => {
      attempts += 1;
      return new Response("not found", { status: 404, statusText: "Not Found" });
    });

    await expect(
      loadHuggingFaceRowsDataset({
        dataset: "demo/proof",
        split: "train",
        baseUrl,
        length: 1,
        parseRow(row) {
          return row;
        },
      }),
    ).rejects.toThrow("404 Not Found");

    expect(attempts).toBe(1);
  });

  test("retries thrown transport errors and preserves row_idx fallback", async () => {
    let attempts = 0;
    const dataset = await loadHuggingFaceRowsDataset({
      dataset: "demo/proof",
      split: "train",
      length: 1,
      fetchImpl: async () => {
        attempts += 1;
        if (attempts < 3) {
          throw new Error("socket closed");
        }
        return Response.json({
          rows: [{ row: { value: 7 } }],
        });
      },
      parseRow(row, rowIndex) {
        if (typeof row !== "object" || row === null || !("value" in row)) {
          throw new Error("bad row");
        }
        return `${rowIndex}:${row.value}`;
      },
    });

    expect(attempts).toBe(3);
    expect(dataset.items()).toEqual(["0:7"]);
  });

  test("rejects short reads for pinned subsets", async () => {
    const { baseUrl } = createServer(() =>
      Response.json({
        rows: [{ row_idx: 0, row: { value: 1 } }],
      }),
    );

    await expect(
      loadHuggingFaceRowsDataset({
        dataset: "demo/proof",
        split: "train",
        baseUrl,
        length: 2,
        parseRow(row) {
          return row;
        },
      }),
    ).rejects.toThrow("returned only 1 row");
  });
});
