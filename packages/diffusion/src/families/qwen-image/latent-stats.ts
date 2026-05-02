import { array, type MxArray, reshape } from "@mlxts/core";

export type QwenImageLatentStats = {
  readonly latentChannels: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
};

/** Validate Qwen-Image latent statistics before encode/decode scaling. */
export function validateQwenImageLatentStats(owner: string, stats: QwenImageLatentStats): void {
  if (
    stats.latentsMean.length !== stats.latentChannels ||
    stats.latentsStd.length !== stats.latentChannels
  ) {
    throw new Error(`${owner}: VAE latent mean/std lengths must match latent channels.`);
  }
}

/** Create an NCFHW broadcast tensor for Qwen-Image latent statistics. */
export function qwenImageLatentStatsTensor(
  values: readonly number[],
  channels: number,
  dtype: MxArray["dtype"],
): MxArray {
  using vector = array([...values], dtype);
  return reshape(vector, [1, channels, 1, 1, 1]);
}
