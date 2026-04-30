import type { FluxConditioning } from "@mlxts/diffusion";

import type { FluxPromptConditioning } from "./conditioning-types";

export function disposeFluxConditioning(conditioning: FluxConditioning): void {
  conditioning.encoderHiddenStates.free();
  conditioning.pooledProjections.free();
  conditioning.textIds?.free();
  conditioning.guidance?.free();
}

export class FluxPromptConditioningResult implements FluxPromptConditioning {
  #disposed = false;

  constructor(
    readonly batchSize: number,
    readonly conditioning: FluxConditioning,
    readonly promptTruncated: boolean,
    readonly prompt2Truncated: boolean,
  ) {}

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    disposeFluxConditioning(this.conditioning);
  }
}
