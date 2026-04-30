import type { StableDiffusionConditioning } from "@mlxts/diffusion";

import type { StableDiffusionPromptConditioning } from "./conditioning-types";

export function disposeConditioning(conditioning: StableDiffusionConditioning): void {
  conditioning.encoderHiddenStates.free();
  conditioning.textTime?.textEmbeds.free();
  conditioning.textTime?.timeIds.free();
}

export class PromptConditioningResult implements StableDiffusionPromptConditioning {
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: StableDiffusionConditioning,
    readonly negativeConditioning: StableDiffusionConditioning | undefined,
    readonly promptTruncated: boolean,
    readonly negativePromptTruncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeConditioning(this.conditioning);
    if (this.negativeConditioning !== undefined) {
      disposeConditioning(this.negativeConditioning);
    }
  }
}
