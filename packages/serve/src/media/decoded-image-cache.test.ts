import { describe, expect, test } from "bun:test";

import { DecodedImageCache } from "./decoded-image-cache";
import type { DecodedRgbImage } from "./image";

function image(bytes: readonly number[]): DecodedRgbImage {
  return {
    width: bytes.length / 3,
    height: 1,
    channels: 3,
    data: new Uint8Array(bytes),
  };
}

describe("DecodedImageCache", () => {
  test("returns cloned decoded images and tracks cache hits", () => {
    const cache = new DecodedImageCache(12);
    cache.set("red", image([255, 0, 0]));

    const first = cache.get("red");
    expect(first?.data).toEqual(new Uint8Array([255, 0, 0]));
    expect(cache.hitCount).toBe(1);
    if (first === undefined) {
      throw new Error("expected a decoded image cache hit.");
    }
    first.data[0] = 0;

    expect(cache.get("red")?.data).toEqual(new Uint8Array([255, 0, 0]));
    expect(cache.hitCount).toBe(2);
  });

  test("evicts least-recently-used images by byte budget", () => {
    const cache = new DecodedImageCache(6);
    cache.set("red", image([255, 0, 0]));
    cache.set("green", image([0, 255, 0]));
    expect(cache.byteSize).toBe(6);

    expect(cache.get("red")).toBeDefined();
    cache.set("blue", image([0, 0, 255]));

    expect(cache.get("green")).toBeUndefined();
    expect(cache.get("red")).toBeDefined();
    expect(cache.get("blue")).toBeDefined();
    expect(cache.byteSize).toBe(6);
  });

  test("skips images larger than the whole budget", () => {
    const cache = new DecodedImageCache(2);
    cache.set("red", image([255, 0, 0]));

    expect(cache.entryCount).toBe(0);
    expect(cache.get("red")).toBeUndefined();
    expect(cache.missCount).toBe(1);
  });

  test("zero budget disables storage while preserving reads as misses", () => {
    const cache = new DecodedImageCache(0);
    cache.set("red", image([255, 0, 0]));

    expect(cache.byteSize).toBe(0);
    expect(cache.entryCount).toBe(0);
    expect(cache.get("red")).toBeUndefined();
  });

  test("rejects invalid budgets", () => {
    expect(() => new DecodedImageCache(-1)).toThrow("non-negative integer");
    expect(() => new DecodedImageCache(1.5)).toThrow("non-negative integer");
  });
});
