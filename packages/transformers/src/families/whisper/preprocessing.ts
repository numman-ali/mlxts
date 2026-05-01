/**
 * Whisper audio feature preparation.
 * @module
 */

import {
  abs,
  add,
  array,
  asStrided,
  asType,
  concatenate,
  divide,
  expandDims,
  formatShape,
  hanning,
  log10,
  type MxArray,
  matmul,
  max,
  maximum,
  multiply,
  pad,
  retainArray,
  rfft,
  slice,
  square,
  subtract,
  takeAxis,
  transpose,
} from "@mlxts/core";

import type { WhisperAudioFeatures, WhisperFeatureExtractorConfig } from "./types";

const DEFAULT_WHISPER_FEATURE_EXTRACTOR_CONFIG: WhisperFeatureExtractorConfig = {
  featureSize: 80,
  samplingRate: 16000,
  hopLength: 160,
  chunkLength: 30,
  nFft: 400,
  paddingValue: 0,
  nSamples: 480000,
  nFrames: 3000,
};

function hzToMel(frequency: number): number {
  const linearScale = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / linearScale;
  const logStep = Math.log(6.4) / 27;
  if (frequency < minLogHz) {
    return frequency / linearScale;
  }
  return minLogMel + Math.log(frequency / minLogHz) / logStep;
}

function melToHz(mel: number): number {
  const linearScale = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / linearScale;
  const logStep = Math.log(6.4) / 27;
  if (mel < minLogMel) {
    return mel * linearScale;
  }
  return minLogHz * Math.exp(logStep * (mel - minLogMel));
}

function linspace(start: number, stop: number, count: number): number[] {
  if (count === 1) {
    return [start];
  }
  const step = (stop - start) / (count - 1);
  return Array.from({ length: count }, (_, index) => start + step * index);
}

function validateFeatureConfig(config: WhisperFeatureExtractorConfig): void {
  if (config.nSamples !== config.chunkLength * config.samplingRate) {
    throw new Error("Whisper feature config nSamples must equal chunkLength * samplingRate.");
  }
  if (config.nFrames !== Math.floor(config.nSamples / config.hopLength)) {
    throw new Error("Whisper feature config nFrames must equal floor(nSamples / hopLength).");
  }
}

function normalizeWhisperAudio(audio: MxArray): MxArray {
  if (audio.shape.length !== 1 || audio.shape[0] === undefined) {
    throw new Error(
      `prepareWhisperAudioFeatures: expected rank-1 mono audio, got ${formatShape(audio.shape)}.`,
    );
  }
  if (audio.dtype === "float32") {
    return retainArray(audio);
  }
  return asType(audio, "float32");
}

function padOrTrimAudio(audio: MxArray, targetLength: number, paddingValue: number): MxArray {
  const sourceLength = audio.shape[0];
  if (sourceLength === undefined) {
    throw new Error("prepareWhisperAudioFeatures: audio length is unknown.");
  }
  if (sourceLength > targetLength) {
    return slice(audio, [0], [targetLength]);
  }
  if (sourceLength < targetLength) {
    return pad(audio, [[0, targetLength - sourceLength]], paddingValue);
  }
  return retainArray(audio);
}

function descendingIndices(start: number, count: number): number[] {
  return Array.from({ length: count }, (_, index) => start - index);
}

function reflectPadAudio(audio: MxArray, padding: number): MxArray {
  const sourceLength = audio.shape[0];
  if (sourceLength === undefined) {
    throw new Error("prepareWhisperAudioFeatures: audio length is unknown.");
  }
  if (sourceLength <= padding) {
    throw new Error(
      `prepareWhisperAudioFeatures: audio length ${sourceLength} must exceed reflection padding ${padding}.`,
    );
  }

  using prefixIndices = array(descendingIndices(padding, padding), "int32");
  using suffixIndices = array(descendingIndices(sourceLength - 2, padding), "int32");
  using prefix = takeAxis(audio, prefixIndices, 0);
  using suffix = takeAxis(audio, suffixIndices, 0);
  return concatenate([prefix, audio, suffix], 0);
}

function framedAudio(audio: MxArray, config: WhisperFeatureExtractorConfig): MxArray {
  const padding = Math.floor(config.nFft / 2);
  using paddedAudio = reflectPadAudio(audio, padding);
  const frameCount = config.nFrames + 1;
  return asStrided(paddedAudio, [frameCount, config.nFft], [config.hopLength, 1]);
}

/** Create a Slaney-normalized Whisper mel filter bank. */
export function createWhisperMelFilterBank(config: WhisperFeatureExtractorConfig): MxArray {
  validateFeatureConfig(config);
  const frequencyBins = Math.floor(config.nFft / 2) + 1;
  const fftFrequencies = linspace(0, config.samplingRate / 2, frequencyBins);
  const minMel = hzToMel(0);
  const maxMel = hzToMel(config.samplingRate / 2);
  const melFrequencies = linspace(minMel, maxMel, config.featureSize + 2).map(melToHz);
  const filters = Array.from({ length: config.featureSize }, (_, melIndex) => {
    const lower = melFrequencies[melIndex] ?? 0;
    const center = melFrequencies[melIndex + 1] ?? 0;
    const upper = melFrequencies[melIndex + 2] ?? 0;
    const enorm = 2 / (upper - lower);
    return fftFrequencies.map((frequency) => {
      const lowerSlope = (frequency - lower) / (center - lower);
      const upperSlope = (upper - frequency) / (upper - center);
      return Math.max(0, Math.min(lowerSlope, upperSlope)) * enorm;
    });
  });
  return array(filters, "float32");
}

/** Prepare channel-last Whisper log-mel input features from 16 kHz mono audio. */
export function prepareWhisperAudioFeatures(
  audio: MxArray,
  config: WhisperFeatureExtractorConfig = DEFAULT_WHISPER_FEATURE_EXTRACTOR_CONFIG,
): WhisperAudioFeatures {
  validateFeatureConfig(config);
  using monoAudio = normalizeWhisperAudio(audio);
  using fixedAudio = padOrTrimAudio(monoAudio, config.nSamples, config.paddingValue);
  using frames = framedAudio(fixedAudio, config);
  using periodicWindowSource = hanning(config.nFft + 1);
  using window = slice(periodicWindowSource, [0], [config.nFft]);
  using windowedFrames = multiply(frames, window);
  using spectrum = rfft(windowedFrames, config.nFft, 1);
  using trimmedSpectrum = slice(
    spectrum,
    [0, 0],
    [config.nFrames, Math.floor(config.nFft / 2) + 1],
  );
  using spectrumMagnitude = abs(trimmedSpectrum);
  using magnitudes = square(spectrumMagnitude);
  using melFilters = createWhisperMelFilterBank(config);
  using transposedMelFilters = transpose(melFilters);
  using melSpectrogram = matmul(magnitudes, transposedMelFilters);
  using flooredSpectrogram = maximum(melSpectrogram, 1e-10);
  using logSpectrogram = log10(flooredSpectrogram);
  using maxLogSpectrogram = max(logSpectrogram);
  using logFloor = subtract(maxLogSpectrogram, 8);
  using clippedSpectrogram = maximum(logSpectrogram, logFloor);
  using shiftedSpectrogram = add(clippedSpectrogram, 4);
  using scaledSpectrogram = divide(shiftedSpectrogram, 4);
  const inputFeatures = expandDims(scaledSpectrogram, 0);
  return {
    inputFeatures,
  };
}
