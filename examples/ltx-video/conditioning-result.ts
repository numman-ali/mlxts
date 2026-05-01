import type { LtxVideoConditioning } from "@mlxts/diffusion";

import type { LtxVideoPromptConditioning } from "./conditioning-types";

export function disposeLtxVideoConditioning(conditioning: LtxVideoConditioning): void {
  conditioning.promptEmbeds.free();
  conditioning.promptAttentionMask.free();
  conditioning.negativePromptEmbeds?.free();
  conditioning.negativePromptAttentionMask?.free();
}

export class LtxVideoPromptConditioningResult implements LtxVideoPromptConditioning {
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: LtxVideoConditioning,
    readonly promptTruncated: boolean,
    readonly negativePromptTruncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeLtxVideoConditioning(this.conditioning);
  }
}
