import { describe, expect, test } from "bun:test";
import { mkdirSync, rmSync, writeFileSync } from "fs";
import { join } from "path";

import { loadWhisperWavAudio } from "./wav";

function tempDir(name: string): string {
  return mkdtempSync(join(import.meta.dir, `.tmp-${name}-`));
}

function mkdtempSync(prefix: string): string {
  const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const directory = `${prefix}${id}`;
  mkdirSync(directory, { recursive: true });
  return directory;
}

function wavBytes(samples: readonly number[], sampleRate = 16000, channels = 1): Uint8Array {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);
  for (let index = 0; index < samples.length; index += 1) {
    view.setInt16(44 + index * 2, samples[index] ?? 0, true);
  }
  return new Uint8Array(buffer);
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

describe("loadWhisperWavAudio", () => {
  test("loads 16 kHz PCM WAV as mono float32 samples", () => {
    const directory = tempDir("whisper-wav");
    try {
      const path = join(directory, "sample.wav");
      writeFileSync(path, wavBytes([0, 16384, -16384]));

      const wav = loadWhisperWavAudio(path);
      try {
        expect(wav.sampleRate).toBe(16000);
        expect(wav.frames).toBe(3);
        expect(wav.format).toBe("pcm_s16le");
        expect(wav.audio.toList()).toEqual([0, 0.5, -0.5]);
      } finally {
        wav.audio.free();
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  test("rejects sample rates that would require resampling", () => {
    const directory = tempDir("whisper-wav-rate");
    try {
      const path = join(directory, "sample.wav");
      writeFileSync(path, wavBytes([0], 8000));

      expect(() => loadWhisperWavAudio(path)).toThrow("expected 16000 Hz");
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
