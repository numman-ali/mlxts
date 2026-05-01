import type { ZImageConditioning } from "@mlxts/diffusion";

import type { ZImagePromptConditioning } from "./conditioning-types";

export function disposeZImageConditioning(conditioning: ZImageConditioning): void {
  for (const captionFeature of conditioning.captionFeatures) {
    captionFeature.free();
  }
}

export class ZImagePromptConditioningResult implements ZImagePromptConditioning {
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: ZImageConditioning,
    readonly promptTruncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeZImageConditioning(this.conditioning);
  }
}
