import { afterEach, describe, expect, test } from "bun:test";
import { MxArray } from "@mlxts/core";
import { existsSync, readFileSync, rmSync } from "fs";

import { writeLtx2AudioWav } from "./audio-output";

const OUTPUT = ".tmp/ltx-video/audio-output-test.wav";

afterEach(() => {
  if (existsSync(OUTPUT)) {
    rmSync(OUTPUT);
  }
});

describe("LTX-2 audio output", () => {
  test("writes finite BCS waveforms as PCM WAV proof artifacts", () => {
    using waveform = MxArray.fromData([0, 0.5, -0.5, 1], [1, 1, 4], "float32");
    const report = writeLtx2AudioWav(waveform, OUTPUT, 16000);
    const bytes = readFileSync(OUTPUT);

    expect(report.status).toBe("passed");
    expect(report.format).toBe("wav");
    expect(report.channels).toBe(1);
    expect(report.samples).toBe(4);
    expect(report.bytes).toBe(52);
    expect(report.sha256).toHaveLength(64);
    expect(String.fromCharCode(bytes[0] ?? 0, bytes[1] ?? 0, bytes[2] ?? 0, bytes[3] ?? 0)).toBe(
      "RIFF",
    );
    expect(String.fromCharCode(bytes[8] ?? 0, bytes[9] ?? 0, bytes[10] ?? 0, bytes[11] ?? 0)).toBe(
      "WAVE",
    );
  });

  test("rejects non-WAV paths and malformed tensors", () => {
    using waveform = MxArray.fromData([0, 1], [1, 2], "float32");

    expect(() => writeLtx2AudioWav(waveform, ".tmp/out.mp3", 16000)).toThrow(".wav");
    expect(() => writeLtx2AudioWav(waveform, OUTPUT, 16000)).toThrow("BCS waveform");
  });
});
