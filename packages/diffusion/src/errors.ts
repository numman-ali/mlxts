/** Error raised when diffusion checkpoint metadata cannot be parsed safely. */
export class DiffusionConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DiffusionConfigError";
  }
}
