import { StableDiffusionAutoencoderKL } from "../stable-diffusion/autoencoder";
import type { StableDiffusion3AutoencoderConfig } from "./config";

/** Stable Diffusion 3 AutoencoderKL with SD3 latent shift metadata. */
export class StableDiffusion3AutoencoderKL extends StableDiffusionAutoencoderKL {
  #shiftFactor: number;

  constructor(config: StableDiffusion3AutoencoderConfig) {
    super(config);
    this.#shiftFactor = config.shiftFactor;
  }

  get shiftFactor(): number {
    return this.#shiftFactor;
  }
}
