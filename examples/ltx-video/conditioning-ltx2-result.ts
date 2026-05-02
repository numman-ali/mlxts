import type { Ltx2Conditioning } from "@mlxts/diffusion";

import type { Ltx2PromptConditioning } from "./conditioning-ltx2-types";

export function disposeLtx2Conditioning(conditioning: Ltx2Conditioning): void {
  conditioning.promptEmbeds.free();
  conditioning.audioPromptEmbeds.free();
  conditioning.promptAttentionMask.free();
  conditioning.negativePromptEmbeds?.free();
  conditioning.negativeAudioPromptEmbeds?.free();
  conditioning.negativePromptAttentionMask?.free();
}

export class Ltx2PromptConditioningResult implements Ltx2PromptConditioning {
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: Ltx2Conditioning,
    readonly promptTruncated: boolean,
    readonly negativePromptTruncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeLtx2Conditioning(this.conditioning);
  }
}
