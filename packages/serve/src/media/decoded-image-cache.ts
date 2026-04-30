/**
 * Host-side decoded image cache for serving media adapters.
 * @module
 */

import type { DecodedRgbImage } from "./image";

export const DEFAULT_DECODED_IMAGE_CACHE_MAX_BYTES = 64 * 1024 * 1024;

type DecodedImageCacheEntry = {
  image: DecodedRgbImage;
  byteSize: number;
};

function decodedImageByteSize(image: DecodedRgbImage): number {
  return image.data.byteLength;
}

function cloneDecodedImage(image: DecodedRgbImage): DecodedRgbImage {
  return {
    width: image.width,
    height: image.height,
    channels: 3,
    data: new Uint8Array(image.data),
  };
}

function requireValidCacheBudget(maxBytes: number): number {
  if (!Number.isInteger(maxBytes) || maxBytes < 0) {
    throw new Error(`DecodedImageCache: maxBytes must be a non-negative integer, got ${maxBytes}.`);
  }
  return maxBytes;
}

/** LRU cache for decoded host RGB image bytes. */
export class DecodedImageCache {
  #byteSize = 0;
  #entries = new Map<string, DecodedImageCacheEntry>();
  #hitCount = 0;
  #maxBytes: number;
  #missCount = 0;

  constructor(maxBytes = DEFAULT_DECODED_IMAGE_CACHE_MAX_BYTES) {
    this.#maxBytes = requireValidCacheBudget(maxBytes);
  }

  get byteSize(): number {
    return this.#byteSize;
  }

  get entryCount(): number {
    return this.#entries.size;
  }

  get hitCount(): number {
    return this.#hitCount;
  }

  get missCount(): number {
    return this.#missCount;
  }

  get(key: string): DecodedRgbImage | undefined {
    const entry = this.#entries.get(key);
    if (entry === undefined) {
      this.#missCount += 1;
      return undefined;
    }
    this.#hitCount += 1;
    this.#entries.delete(key);
    this.#entries.set(key, entry);
    return cloneDecodedImage(entry.image);
  }

  set(key: string, image: DecodedRgbImage): void {
    this.delete(key);
    const byteSize = decodedImageByteSize(image);
    if (byteSize > this.#maxBytes) {
      return;
    }
    this.#entries.set(key, { image: cloneDecodedImage(image), byteSize });
    this.#byteSize += byteSize;
    this.#evictOverBudget();
  }

  delete(key: string): void {
    const entry = this.#entries.get(key);
    if (entry === undefined) {
      return;
    }
    this.#entries.delete(key);
    this.#byteSize -= entry.byteSize;
  }

  clear(): void {
    this.#entries.clear();
    this.#byteSize = 0;
  }

  #evictOverBudget(): void {
    while (this.#byteSize > this.#maxBytes) {
      const oldest = this.#entries.keys().next().value;
      if (oldest === undefined) {
        return;
      }
      this.delete(oldest);
    }
  }
}
