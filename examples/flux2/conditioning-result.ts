import type { Flux2KleinConditioning } from "@mlxts/diffusion";

import type { Flux2KleinPromptConditioning } from "./conditioning-types";

export function disposeFlux2KleinConditioning(conditioning: Flux2KleinConditioning): void {
  conditioning.promptEmbeds.free();
  conditioning.textIds?.free();
  conditioning.negativePromptEmbeds?.free();
  conditioning.negativeTextIds?.free();
}

export class Flux2KleinPromptConditioningResult implements Flux2KleinPromptConditioning {
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: Flux2KleinConditioning,
    readonly promptTruncated: boolean,
    readonly negativePromptTruncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeFlux2KleinConditioning(this.conditioning);
  }
}
