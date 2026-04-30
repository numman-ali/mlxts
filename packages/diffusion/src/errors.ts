/** Error raised when diffusion checkpoint metadata cannot be parsed safely. */
export class DiffusionConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DiffusionConfigError";
  }
}

/** Error raised when a diffusion checkpoint tensor does not match its parameter slot. */
export class DiffusionWeightMismatchError extends Error {
  constructor(path: string, expectedShape: readonly number[], actualShape: readonly number[]) {
    super(
      `loadDiffusionWeights: checkpoint tensor for "${path}" has shape [${actualShape.join(
        ", ",
      )}], expected [${expectedShape.join(", ")}].`,
    );
    this.name = "DiffusionWeightMismatchError";
  }
}

/** Error raised when a diffusion checkpoint leaves required parameters unassigned. */
export class DiffusionMissingWeightsError extends Error {
  constructor(paths: readonly string[]) {
    const sortedPaths = [...paths].toSorted((left, right) => left.localeCompare(right));
    super(
      `loadDiffusionWeights: checkpoint is missing required parameters: ${sortedPaths.join(", ")}.`,
    );
    this.name = "DiffusionMissingWeightsError";
  }
}
