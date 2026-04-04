import { AdamW } from "mlx-ts";

export const DEFAULT_ADAMW_CONFIG = Object.freeze({
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
});

export function createDefaultAdamW(learningRate: number, weightDecay: number): AdamW {
  return new AdamW({
    learningRate,
    beta1: DEFAULT_ADAMW_CONFIG.beta1,
    beta2: DEFAULT_ADAMW_CONFIG.beta2,
    eps: DEFAULT_ADAMW_CONFIG.eps,
    weightDecay,
  });
}
