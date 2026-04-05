import { describe, expect, test } from "bun:test";

import { datasetFromArray } from "./dataset";

describe("datasetFromArray", () => {
  test("wraps immutable item access", () => {
    const dataset = datasetFromArray([1, 2, 3]);

    expect(dataset.length).toBe(3);
    expect(dataset.at(1)).toBe(2);
    expect(dataset.items()).toEqual([1, 2, 3]);
  });

  test("rejects out-of-bounds reads", () => {
    const dataset = datasetFromArray([1]);
    expect(() => dataset.at(2)).toThrow("out of bounds");
  });
});
