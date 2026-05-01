import type { QwenImageConditioning } from "@mlxts/diffusion";

import type { QwenImagePromptConditioning } from "./conditioning-types";

export function disposeQwenImageConditioning(conditioning: QwenImageConditioning): void {
  conditioning.promptEmbeds.free();
  conditioning.promptEmbedsMask?.free();
  conditioning.negativePromptEmbeds?.free();
  conditioning.negativePromptEmbedsMask?.free();
}

export class QwenImagePromptConditioningResult implements QwenImagePromptConditioning {
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: QwenImageConditioning,
    readonly promptTruncated: boolean,
    readonly negativePromptTruncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeQwenImageConditioning(this.conditioning);
  }
}
