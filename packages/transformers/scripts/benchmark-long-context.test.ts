import { describe, expect, test } from "bun:test";
import type { Tokenizer } from "@mlxts/tokenizers";
import { mkdtempSync, rmSync } from "fs";
import { join } from "path";
import type { LongContextOptions, LongContextResult } from "./benchmark-long-context";
import {
  assertLongContextResult,
  buildNeedlePromptTokenIds,
  defaultContextTargets,
  findNeedleTokenOffset,
  findNeedleTokenSpan,
  inferMaxContextTokens,
  normalizeExactResponse,
  parseLongContextArgs,
  parseNeedlePositions,
  splitNeedleFillerRepetitions,
  writeLongContextReport,
} from "./benchmark-long-context";

function mockTokenizer(): Tokenizer {
  return {
    vocabSize: 256,
    bosTokenId: 2,
    eosTokenIds: [3],
    padTokenId: 0,
    encode(text: string, options?: { addSpecialTokens?: boolean }) {
      const ids = [...text].map((character) => character.charCodeAt(0));
      return options?.addSpecialTokens === false ? ids : [2, ...ids];
    },
    encodeWithOffsets(text: string, options?: { addSpecialTokens?: boolean }) {
      const ids = [...text].map((character) => character.charCodeAt(0));
      const offsets = [...text].map((_, index) => ({ start: index, end: index + 1 }));
      if (options?.addSpecialTokens === false) {
        return { ids, offsets };
      }
      return {
        ids: [2, ...ids],
        offsets: [{ start: 0, end: 0 }, ...offsets],
        specialTokensMask: [1, ...ids.map(() => 0)],
      };
    },
    encodeBatch() {
      throw new Error("not needed");
    },
    decode(ids: readonly number[]) {
      return ids
        .filter((token) => token !== 2 && token !== 3)
        .map((token) => String.fromCharCode(token))
        .join("");
    },
    decodeBatch() {
      throw new Error("not needed");
    },
  };
}

function options(overrides: Partial<LongContextOptions> = {}): LongContextOptions {
  return {
    model: "tiny",
    rungs: [128],
    generationTokens: 8,
    prefillStepSize: 32,
    metalTrace: false,
    needlePositions: ["late"],
    reportJson: null,
    failOnMismatch: false,
    maxActiveSlopeMbPerToken: null,
    ...overrides,
  };
}

function result(overrides: Partial<LongContextResult> = {}): LongContextResult {
  return {
    needlePosition: "late",
    rungTokens: 128,
    promptTokens: 128,
    expectedMarker: "MKR-TINY-128-LATE",
    needleTokenOffset: 100,
    needleTokenStart: 100,
    needleTokenEnd: 110,
    needleTokenCenter: 105,
    needleTokenFraction: 100 / 128,
    needleTokenCenterFraction: 105 / 128,
    prefillSeconds: 1,
    prefillTps: 128,
    firstTokenSeconds: 0.1,
    decodeTokens: 8,
    decodeTps: 80,
    prefillPeakMemoryGb: 1,
    activeMemoryAfterPrefillGb: 1,
    cacheMemoryAfterPrefillGb: 0.1,
    activeMemoryAfterFirstTokenGb: 1,
    cacheMemoryAfterFirstTokenGb: 0.1,
    activeMemoryAfterDecodeGb: 1,
    cacheMemoryAfterDecodeGb: 0.1,
    activeMemoryDecodeDeltaGb: 0,
    activeMemoryDecodeSlopeMbPerToken: 0,
    peakMemoryAfterDecodeGb: 1,
    exactMatch: true,
    containsSecret: true,
    responseText: "MKR-TINY-128-LATE",
    ...overrides,
  };
}

describe("benchmark-long-context", () => {
  test("defaultContextTargets follows the 32k/64k/128k/256k ladder", () => {
    expect(defaultContextTargets(16_384)).toEqual([]);
    expect(defaultContextTargets(65_536)).toEqual([32_768, 65_536]);
    expect(defaultContextTargets(200_000)).toEqual([32_768, 65_536, 131_072]);
    expect(defaultContextTargets(300_000)).toEqual([32_768, 65_536, 131_072, 262_144]);
  });

  test("inferMaxContextTokens reads the common transformer config fields", () => {
    expect(inferMaxContextTokens({ max_position_embeddings: 131_072 })).toBe(131_072);
    expect(inferMaxContextTokens({ max_sequence_length: 65_536 })).toBe(65_536);
    expect(
      inferMaxContextTokens({
        model_type: "qwen3_5",
        text_config: { max_position_embeddings: 262_144 },
      }),
    ).toBe(262_144);
    expect(() => inferMaxContextTokens({})).toThrow("max context field");
  });

  test("parseNeedlePositions accepts defaults, all, and deduped lists", () => {
    expect(parseNeedlePositions(undefined)).toEqual(["late"]);
    expect(parseNeedlePositions("all")).toEqual(["early", "middle", "late"]);
    expect(parseNeedlePositions("middle,early,middle")).toEqual(["middle", "early"]);
    expect(() => parseNeedlePositions("near")).toThrow("--needle-placements entries");
  });

  test("parseLongContextArgs rejects missing needle placement values", () => {
    const parsed = parseLongContextArgs([
      "mlx-community/Qwen3.6-27B-4bit",
      "--needle-placements",
      "all",
      "--fail-on-mismatch",
      "--max-active-slope-mb-per-token",
      "0.5",
    ]);
    expect(parsed.needlePositions).toEqual(["early", "middle", "late"]);
    expect(parsed.failOnMismatch).toBe(true);
    expect(parsed.maxActiveSlopeMbPerToken).toBe(0.5);
    expect(() =>
      parseLongContextArgs(["mlx-community/Qwen3.6-27B-4bit", "--needle-placements"]),
    ).toThrow("--needle-placements expects a value");
    expect(() =>
      parseLongContextArgs([
        "mlx-community/Qwen3.6-27B-4bit",
        "--needle-placements",
        "--generation-tokens",
        "8",
      ]),
    ).toThrow("--needle-placements expects a value");
    expect(() =>
      parseLongContextArgs([
        "mlx-community/Qwen3.6-27B-4bit",
        "--max-active-slope-mb-per-token",
        "-1",
      ]),
    ).toThrow("--max-active-slope-mb-per-token expects a non-negative number");
  });

  test("splitNeedleFillerRepetitions places markers across the context", () => {
    expect(splitNeedleFillerRepetitions(10, "early")).toEqual({ before: 1, after: 9 });
    expect(splitNeedleFillerRepetitions(10, "middle")).toEqual({ before: 5, after: 5 });
    expect(splitNeedleFillerRepetitions(10, "late")).toEqual({ before: 10, after: 0 });
  });

  test("buildNeedlePromptTokenIds fills the exact token budget and keeps the retrieval tail", () => {
    const tokenizer = mockTokenizer();
    const promptTokenIds = buildNeedlePromptTokenIds(tokenizer, 256, "ALBATROSS");

    expect(promptTokenIds).toHaveLength(256);
    expect(tokenizer.decode(promptTokenIds.slice(-80))).toContain("ALBATROSS");
  });

  test("findNeedleTokenSpan reports the marker location when offsets are available", () => {
    const tokenizer = mockTokenizer();
    const promptText = "aa ALBATROSS zz";
    const promptTokenIds = tokenizer.encode(promptText, { addSpecialTokens: true });

    expect(findNeedleTokenOffset(tokenizer, promptText, promptTokenIds, "ALBATROSS")).toBe(4);
    expect(findNeedleTokenSpan(tokenizer, promptText, promptTokenIds, "ALBATROSS")).toEqual({
      start: 4,
      end: 13,
      center: 8.5,
      centerFraction: 8.5 / promptTokenIds.length,
    });
    expect(findNeedleTokenOffset(tokenizer, promptText, promptTokenIds, "MISSING")).toBeNull();
  });

  test("writeLongContextReport creates parent directories and preserves evidence", async () => {
    const tempDir = mkdtempSync(join(import.meta.dir, "long-context-report-"));
    const reportPath = join(tempDir, "nested", "report.json");

    try {
      await writeLongContextReport(reportPath, {
        createdAt: "2026-04-24T00:00:00.000Z",
        model: "tiny",
        resolvedModelSource: "/tmp/tiny",
        maxContextTokens: 1024,
        rungTargets: [128],
        generationTokens: 8,
        prefillStepSize: 32,
        needlePositions: ["early", "late"],
        results: [
          {
            needlePosition: "early",
            rungTokens: 128,
            promptTokens: 130,
            expectedMarker: "MKR-TINY-128-EARLY",
            needleTokenOffset: 10,
            needleTokenStart: 10,
            needleTokenEnd: 15,
            needleTokenCenter: 12.5,
            needleTokenFraction: 10 / 130,
            needleTokenCenterFraction: 12.5 / 130,
            prefillSeconds: 1,
            prefillTps: 130,
            firstTokenSeconds: 0.1,
            decodeTokens: 8,
            decodeTps: 80,
            prefillPeakMemoryGb: 1,
            activeMemoryAfterPrefillGb: 1,
            cacheMemoryAfterPrefillGb: 0.1,
            activeMemoryAfterFirstTokenGb: 1,
            cacheMemoryAfterFirstTokenGb: 0.1,
            activeMemoryAfterDecodeGb: 1,
            cacheMemoryAfterDecodeGb: 0.1,
            activeMemoryDecodeDeltaGb: 0,
            activeMemoryDecodeSlopeMbPerToken: 0,
            peakMemoryAfterDecodeGb: 1,
            exactMatch: true,
            containsSecret: true,
            responseText: "MKR-TINY-128-EARLY",
          },
        ],
      });

      const report = (await Bun.file(reportPath).json()) as {
        needlePositions?: string[];
        results?: Array<{ expectedMarker?: string }>;
      };
      expect(report.needlePositions).toEqual(["early", "late"]);
      expect(report.results?.[0]?.expectedMarker).toBe("MKR-TINY-128-EARLY");
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("normalizeExactResponse grades the first non-empty generated answer line", () => {
    expect(normalizeExactResponse("\n`MKR-123`,\nassistant\nextra")).toBe("MKR-123");
  });

  test("assertLongContextResult can fail retrieval and decode memory regressions", () => {
    expect(() =>
      assertLongContextResult(result(), options({ failOnMismatch: true })),
    ).not.toThrow();
    expect(() =>
      assertLongContextResult(
        result({ exactMatch: false, responseText: "wrong" }),
        options({ failOnMismatch: true }),
      ),
    ).toThrow("expected exact marker");
    expect(() =>
      assertLongContextResult(
        result({ activeMemoryDecodeSlopeMbPerToken: 2 }),
        options({ maxActiveSlopeMbPerToken: 1 }),
      ),
    ).toThrow("active_decode_slope");
  });
});
