import type { StableDiffusion3Conditioning } from "@mlxts/diffusion";

import type { StableDiffusion3PromptConditioning } from "./conditioning-types";

export function disposeStableDiffusion3Conditioning(
  conditioning: StableDiffusion3Conditioning,
): void {
  conditioning.encoderHiddenStates.free();
  conditioning.pooledProjections.free();
}

export class StableDiffusion3PromptConditioningResult
  implements StableDiffusion3PromptConditioning
{
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: StableDiffusion3Conditioning,
    readonly negativeConditioning: StableDiffusion3Conditioning | undefined,
    readonly promptTruncated: boolean,
    readonly prompt2Truncated: boolean,
    readonly prompt3Truncated: boolean,
    readonly negativePromptTruncated: boolean,
    readonly negativePrompt2Truncated: boolean,
    readonly negativePrompt3Truncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeStableDiffusion3Conditioning(this.conditioning);
    if (this.negativeConditioning !== undefined) {
      disposeStableDiffusion3Conditioning(this.negativeConditioning);
    }
  }
}
