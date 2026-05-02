import type { MxArray } from "@mlxts/core";
import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname } from "path";

export type Ltx2AudioWavChecks = {
  riffHeaderValid: boolean;
  byteLengthMatches: boolean;
  sampleRateMatches: boolean;
  sha256Present: boolean;
  finiteTensor: boolean;
  waveformHasSamples: boolean;
};

export type Ltx2AudioWavWriteResult = {
  path: string;
  format: "wav";
  sampleRate: number;
  channels: number;
  samples: number;
  durationSeconds: number;
  bitsPerSample: 16;
  bytes: number;
  sha256: string;
  peakAbs: number;
  meanAbs: number;
  checks: Ltx2AudioWavChecks;
  status: "passed" | "failed";
};

const WAV_HEADER_BYTES = 44;
const PCM_BYTES_PER_SAMPLE = 2;

function roundMetric(value: number): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  return Math.round(value * 1_000_000) / 1_000_000;
}

function sha256Hex(bytes: Uint8Array): string {
  return new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function clampPcm16(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const clamped = Math.max(-1, Math.min(1, value));
  return clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
}

function expectWaveform(waveform: MxArray): {
  channels: number;
  samples: number;
} {
  const [batch, channels, samples] = waveform.shape;
  if (
    waveform.shape.length !== 3 ||
    batch !== 1 ||
    channels === undefined ||
    samples === undefined ||
    channels <= 0 ||
    samples <= 0
  ) {
    throw new Error(
      `LTX-2 audio proof expects one BCS waveform, got [${waveform.shape.join(",")}].`,
    );
  }
  return { channels, samples };
}

function wavBytes(
  waveform: MxArray,
  sampleRate: number,
): {
  bytes: Uint8Array;
  channels: number;
  samples: number;
  peakAbs: number;
  meanAbs: number;
  finiteTensor: boolean;
} {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new Error("LTX-2 audio sample rate must be a positive integer.");
  }
  const { channels, samples } = expectWaveform(waveform);
  waveform.eval();
  const values = waveform.toTypedArray();
  const payloadBytes = channels * samples * PCM_BYTES_PER_SAMPLE;
  const bytes = new Uint8Array(WAV_HEADER_BYTES + payloadBytes);
  const view = new DataView(bytes.buffer);
  let peakAbs = 0;
  let totalAbs = 0;
  let finiteTensor = true;

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, bytes.byteLength - 8, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * PCM_BYTES_PER_SAMPLE, true);
  view.setUint16(32, channels * PCM_BYTES_PER_SAMPLE, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, payloadBytes, true);

  for (let sample = 0; sample < samples; sample += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const value = Number(values[channel * samples + sample] ?? Number.NaN);
      finiteTensor = finiteTensor && Number.isFinite(value);
      const absValue = Number.isFinite(value) ? Math.abs(value) : 0;
      peakAbs = Math.max(peakAbs, absValue);
      totalAbs += absValue;
      view.setInt16(
        WAV_HEADER_BYTES + (sample * channels + channel) * PCM_BYTES_PER_SAMPLE,
        clampPcm16(value),
        true,
      );
    }
  }

  return {
    bytes,
    channels,
    samples,
    peakAbs: roundMetric(peakAbs),
    meanAbs: roundMetric(totalAbs / Math.max(1, channels * samples)),
    finiteTensor,
  };
}

function checksFor(report: Omit<Ltx2AudioWavWriteResult, "checks" | "status">): Ltx2AudioWavChecks {
  const bytes = readFileSync(report.path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  const wave = String.fromCharCode(
    view.getUint8(8),
    view.getUint8(9),
    view.getUint8(10),
    view.getUint8(11),
  );
  return {
    riffHeaderValid: magic === "RIFF" && wave === "WAVE" && view.getUint16(34, true) === 16,
    byteLengthMatches: bytes.byteLength === report.bytes,
    sampleRateMatches: view.getUint32(24, true) === report.sampleRate,
    sha256Present: report.sha256.length === 64,
    finiteTensor: Number.isFinite(report.peakAbs) && Number.isFinite(report.meanAbs),
    waveformHasSamples: report.samples > 0 && report.channels > 0,
  };
}

function statusFor(checks: Ltx2AudioWavChecks): "passed" | "failed" {
  return Object.values(checks).every(Boolean) ? "passed" : "failed";
}

function failedCheckNames(checks: Ltx2AudioWavChecks): string[] {
  return Object.entries(checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
}

/** Write a BCS LTX-2 waveform as little-endian 16-bit PCM WAV. */
export function writeLtx2AudioWav(
  waveform: MxArray,
  outputPath: string,
  sampleRate: number,
): Ltx2AudioWavWriteResult {
  if (!outputPath.toLowerCase().endsWith(".wav")) {
    throw new Error("LTX-2 audio output path must end with .wav.");
  }
  const proof = wavBytes(waveform, sampleRate);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, proof.bytes);
  const baseReport: Omit<Ltx2AudioWavWriteResult, "checks" | "status"> = {
    path: outputPath,
    format: "wav",
    sampleRate,
    channels: proof.channels,
    samples: proof.samples,
    durationSeconds: roundMetric(proof.samples / sampleRate),
    bitsPerSample: 16,
    bytes: proof.bytes.byteLength,
    sha256: sha256Hex(proof.bytes),
    peakAbs: proof.peakAbs,
    meanAbs: proof.meanAbs,
  };
  const checks = checksFor(baseReport);
  const finalChecks = {
    ...checks,
    finiteTensor: checks.finiteTensor && proof.finiteTensor,
  };
  const status = statusFor(finalChecks);
  if (status !== "passed") {
    throw new Error(
      `LTX-2 audio proof artifact failed checks: ${failedCheckNames(finalChecks).join(", ")}`,
    );
  }
  return {
    ...baseReport,
    checks: finalChecks,
    status,
  };
}
