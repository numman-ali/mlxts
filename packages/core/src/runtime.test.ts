import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { array } from "./array";
import {
  createStream,
  deviceCount,
  getRecommendedWorkingSetBytes,
  isDeviceAvailable,
  synchronize,
  withDefaultStream,
} from "./device";
import {
  clearMemoryCache,
  getActiveMemoryBytes,
  getCacheMemoryBytes,
  getMemoryLimitBytes,
  getMemoryStats,
  getPeakMemoryBytes,
  resetPeakMemory,
  setCacheLimitBytes,
  setMemoryLimitBytes,
  setWiredLimitBytes,
} from "./memory";
import { isMetalAvailable, startMetalCapture, stopMetalCapture } from "./metal";
import { mxAsyncEval } from "./transforms";

describe("runtime controls", () => {
  test("memory stats are readable", () => {
    const stats = getMemoryStats();
    expect(stats.activeBytes).toBeGreaterThanOrEqual(0);
    expect(stats.cacheBytes).toBeGreaterThanOrEqual(0);
    expect(stats.peakBytes).toBeGreaterThanOrEqual(0);
    expect(stats.limitBytes).toBeGreaterThanOrEqual(0);
    expect(getActiveMemoryBytes()).toBeGreaterThanOrEqual(0);
    expect(getCacheMemoryBytes()).toBeGreaterThanOrEqual(0);
    expect(getPeakMemoryBytes()).toBeGreaterThanOrEqual(0);
    expect(getMemoryLimitBytes()).toBeGreaterThanOrEqual(0);
  });

  test("cache controls round-trip safely", () => {
    const currentLimit = getMemoryLimitBytes();
    const previous = setCacheLimitBytes(currentLimit);
    expect(previous).toBeGreaterThanOrEqual(0);
    clearMemoryCache();
    resetPeakMemory();
  });

  test("memory and wired limit controls round-trip safely", () => {
    const currentLimit = getMemoryLimitBytes();
    const previousMemoryLimit = setMemoryLimitBytes(currentLimit);

    expect(previousMemoryLimit).toBeGreaterThanOrEqual(0);
    try {
      const previousWiredLimit = setWiredLimitBytes(currentLimit);
      expect(previousWiredLimit).toBeGreaterThanOrEqual(0);
    } catch (error) {
      expect(String(error)).toContain("wired");
    }
  });

  test("device availability helpers reflect the current runtime", () => {
    expect(deviceCount("cpu")).toBeGreaterThanOrEqual(1);
    expect(isDeviceAvailable("cpu")).toBe(true);
    expect(deviceCount("gpu")).toBeGreaterThanOrEqual(0);
  });

  test("recommended working-set size is readable on the active GPU", () => {
    if (!isDeviceAvailable("gpu")) {
      return;
    }

    expect(getRecommendedWorkingSetBytes("gpu")).toBeGreaterThan(0);
  });

  test("metal availability is readable", () => {
    expect(typeof isMetalAvailable()).toBe("boolean");
  });

  test("metal capture rejects an empty path", () => {
    expect(() => startMetalCapture("")).toThrow("path must not be empty");
  });

  test("metal capture can start and stop with a valid path", () => {
    if (!isMetalAvailable()) {
      return;
    }

    const capturePath = join(mkdtempSync(join(tmpdir(), "mlx-metal-capture-")), "trace.gputrace");
    try {
      startMetalCapture(capturePath);
      stopMetalCapture();
    } catch (error) {
      expect(String(error)).toContain("capture");
    }
  });

  test("async eval can be synchronized explicitly", () => {
    using values = array([1, 2, 3], "float32");
    mxAsyncEval(values);
    synchronize();
    expect(values.toList()).toEqual([1, 2, 3]);
  });

  test("custom streams can be scoped as the default stream", () => {
    if (!isDeviceAvailable("gpu")) {
      return;
    }

    using stream = createStream("gpu");
    using values = withDefaultStream(stream, () => {
      const numbers = array([1, 2, 3], "float32");
      mxAsyncEval(numbers);
      synchronize(stream);
      return numbers;
    });

    expect(values.toList()).toEqual([1, 2, 3]);
  });
});
