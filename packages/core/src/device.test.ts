import { describe, expect, test } from "bun:test";

import { cpuStream, defaultStream, getDefaultDevice, gpuStream, setDefaultDevice } from "./device";

describe("device", () => {
  test("gpuStream and cpuStream reuse singleton handles", () => {
    expect(gpuStream()).toBe(gpuStream());
    expect(cpuStream()).toBe(cpuStream());
  });

  test("defaultStream tracks the configured default device", () => {
    const original = getDefaultDevice();

    try {
      setDefaultDevice("cpu");
      expect(getDefaultDevice()).toBe("cpu");
      expect(defaultStream()).toBe(cpuStream());

      setDefaultDevice("gpu");
      expect(getDefaultDevice()).toBe("gpu");
      expect(defaultStream()).toBe(gpuStream());
    } finally {
      setDefaultDevice(original);
    }
  });
});
