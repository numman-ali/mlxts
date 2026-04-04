/**
 * Data loading and batching for GPT training.
 *
 * Supports loading text from a local file or downloading Shakespeare
 * as a convenience default. Batching creates random windows of token
 * sequences with input/target pairs (target = input shifted by 1).
 *
 * @module
 */

import { mkdirSync } from "fs";
import { array, type MxArray, reshape } from "mlx-ts";
import { join } from "path";

const SHAKESPEARE_URL =
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const CACHE_FILENAME = "shakespeare.txt";

/** Deterministic pseudo-random source for repeatable batching. */
export function createRandomSource(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Load training text from a local file or download Shakespeare.
 *
 * Tries in order: (1) explicit path, (2) cached local file, (3) download and cache.
 */
export async function loadText(options: { path?: string; cacheDir?: string }): Promise<string> {
  // 1. Explicit path
  if (options.path !== undefined) {
    return await Bun.file(options.path).text();
  }

  // 2. Cached
  const cacheDir = options.cacheDir ?? join(process.cwd(), ".nanogpt-cache");
  const cachePath = join(cacheDir, CACHE_FILENAME);
  const cacheFile = Bun.file(cachePath);
  if (await cacheFile.exists()) {
    return await cacheFile.text();
  }

  // 3. Download and cache
  const response = await fetch(SHAKESPEARE_URL);
  if (!response.ok) {
    throw new Error(`Failed to download Shakespeare: ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  mkdirSync(cacheDir, { recursive: true });
  await Bun.write(cachePath, `${text}`);
  return text;
}

/** Tokenize text and split into train/val arrays. */
export function prepareData(
  tokens: number[],
  trainSplit: number,
): { trainTokens: Int32Array; valTokens: Int32Array } {
  if (trainSplit <= 0 || trainSplit >= 1) {
    throw new Error(`prepareData: trainSplit must be in (0, 1), got ${trainSplit}`);
  }

  const splitIndex = Math.floor(tokens.length * trainSplit);
  return {
    trainTokens: new Int32Array(tokens.slice(0, splitIndex)),
    valTokens: new Int32Array(tokens.slice(splitIndex)),
  };
}

/** Get a random batch of input/target pairs from a token array. */
export function getBatch(
  tokens: Int32Array,
  batchSize: number,
  blockSize: number,
  nextRandom: () => number = Math.random,
): { input: MxArray; target: MxArray } {
  if (batchSize <= 0) {
    throw new Error(`getBatch: batchSize must be > 0, got ${batchSize}`);
  }
  if (blockSize <= 0) {
    throw new Error(`getBatch: blockSize must be > 0, got ${blockSize}`);
  }

  const maxStart = tokens.length - blockSize - 1;
  if (maxStart < 0) {
    throw new Error(
      `getBatch: token array length ${tokens.length} is too short for blockSize ${blockSize}`,
    );
  }

  const inputData = new Int32Array(batchSize * blockSize);
  const targetData = new Int32Array(batchSize * blockSize);

  for (let b = 0; b < batchSize; b++) {
    const start = Math.floor(nextRandom() * (maxStart + 1));
    const offset = b * blockSize;
    for (let t = 0; t < blockSize; t++) {
      const srcToken = tokens[start + t];
      const tgtToken = tokens[start + t + 1];
      if (srcToken === undefined || tgtToken === undefined) {
        throw new Error(`getBatch: unexpected undefined token at position ${start + t}`);
      }
      inputData[offset + t] = srcToken;
      targetData[offset + t] = tgtToken;
    }
  }

  const flatInput = array(inputData, "int32");
  const flatTarget = array(targetData, "int32");
  const input = reshape(flatInput, [batchSize, blockSize]);
  const target = reshape(flatTarget, [batchSize, blockSize]);
  flatInput.free();
  flatTarget.free();
  return { input, target };
}
