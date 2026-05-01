/**
 * Local WAV loading for the Whisper example.
 * @module
 */

import { MxArray } from "@mlxts/core";
import { readFileSync } from "fs";

export type WhisperWavFormat = "pcm_s16le" | "float32le";

export type WhisperWavAudio = {
  path: string;
  sampleRate: number;
  channels: number;
  frames: number;
  durationSeconds: number;
  format: WhisperWavFormat;
  audio: MxArray;
};

export type WhisperWavLoadOptions = {
  expectedSampleRate?: number;
};

type WavFormatChunk = {
  audioFormat: number;
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
};

type WavDataChunk = {
  offset: number;
  bytes: number;
};

function fourCc(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

function expectRiffWave(view: DataView, path: string): void {
  if (view.byteLength < 12 || fourCc(view, 0) !== "RIFF" || fourCc(view, 8) !== "WAVE") {
    throw new Error(`loadWhisperWavAudio: ${path} is not a RIFF/WAVE file.`);
  }
}

function readFormatChunk(view: DataView, offset: number, bytes: number): WavFormatChunk {
  if (bytes < 16) {
    throw new Error("loadWhisperWavAudio: fmt chunk is truncated.");
  }
  return {
    audioFormat: view.getUint16(offset, true),
    channels: view.getUint16(offset + 2, true),
    sampleRate: view.getUint32(offset + 4, true),
    bitsPerSample: view.getUint16(offset + 14, true),
  };
}

function scanChunks(view: DataView): { format: WavFormatChunk; data: WavDataChunk } {
  let format: WavFormatChunk | null = null;
  let data: WavDataChunk | null = null;
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = fourCc(view, offset);
    const bytes = view.getUint32(offset + 4, true);
    const payloadOffset = offset + 8;
    const nextOffset = payloadOffset + bytes + (bytes % 2);
    if (payloadOffset + bytes > view.byteLength) {
      throw new Error(`loadWhisperWavAudio: ${id} chunk exceeds file size.`);
    }
    if (id === "fmt ") {
      format = readFormatChunk(view, payloadOffset, bytes);
    } else if (id === "data") {
      data = { offset: payloadOffset, bytes };
    }
    offset = nextOffset;
  }
  if (format === null) {
    throw new Error("loadWhisperWavAudio: missing fmt chunk.");
  }
  if (data === null) {
    throw new Error("loadWhisperWavAudio: missing data chunk.");
  }
  return { format, data };
}

function wavFormat(format: WavFormatChunk): WhisperWavFormat {
  if (format.audioFormat === 1 && format.bitsPerSample === 16) {
    return "pcm_s16le";
  }
  if (format.audioFormat === 3 && format.bitsPerSample === 32) {
    return "float32le";
  }
  throw new Error(
    `loadWhisperWavAudio: unsupported WAV format ${format.audioFormat} with ${format.bitsPerSample} bits per sample.`,
  );
}

function sampleValue(view: DataView, offset: number, format: WhisperWavFormat): number {
  return format === "pcm_s16le"
    ? view.getInt16(offset, true) / 32768
    : view.getFloat32(offset, true);
}

function decodeMonoSamples(
  view: DataView,
  data: WavDataChunk,
  format: WavFormatChunk,
  sampleFormat: WhisperWavFormat,
): Float32Array {
  const bytesPerSample = format.bitsPerSample / 8;
  const frameBytes = bytesPerSample * format.channels;
  if (format.channels <= 0 || !Number.isInteger(data.bytes / frameBytes)) {
    throw new Error("loadWhisperWavAudio: data chunk size is not frame-aligned.");
  }
  const frames = data.bytes / frameBytes;
  const samples = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let mixed = 0;
    const frameOffset = data.offset + frame * frameBytes;
    for (let channel = 0; channel < format.channels; channel += 1) {
      mixed += sampleValue(view, frameOffset + channel * bytesPerSample, sampleFormat);
    }
    samples[frame] = mixed / format.channels;
  }
  return samples;
}

/** Load a local 16 kHz WAV file into mono float32 audio for Whisper features. */
export function loadWhisperWavAudio(
  path: string,
  options: WhisperWavLoadOptions = {},
): WhisperWavAudio {
  const bytes = readFileSync(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  expectRiffWave(view, path);
  const chunks = scanChunks(view);
  const sampleFormat = wavFormat(chunks.format);
  const expectedSampleRate = options.expectedSampleRate ?? 16000;
  if (chunks.format.sampleRate !== expectedSampleRate) {
    throw new Error(
      `loadWhisperWavAudio: expected ${expectedSampleRate} Hz audio, got ${chunks.format.sampleRate} Hz.`,
    );
  }
  const samples = decodeMonoSamples(view, chunks.data, chunks.format, sampleFormat);
  const audio = MxArray.fromData(samples, [samples.length], "float32");
  return {
    path,
    sampleRate: chunks.format.sampleRate,
    channels: chunks.format.channels,
    frames: samples.length,
    durationSeconds: samples.length / chunks.format.sampleRate,
    format: sampleFormat,
    audio,
  };
}
