import type { ContinuousBatchAdmissionController } from "./continuous-batch-types";

export class AdmissionReleaseWakeup {
  readonly #admissionController: ContinuousBatchAdmissionController | undefined;
  #unsubscribe: (() => void) | undefined;
  #waiting = false;

  constructor(admissionController: ContinuousBatchAdmissionController | undefined) {
    this.#admissionController = admissionController;
  }

  get waiting(): boolean {
    return this.#waiting;
  }

  cancel(): void {
    this.#unsubscribe?.();
    this.#unsubscribe = undefined;
    this.#waiting = false;
  }

  pause(schedule: () => void): void {
    if (this.#admissionController === undefined || this.#waiting) {
      return;
    }
    this.#waiting = true;
    this.#unsubscribe = this.#admissionController.onRelease(() => {
      this.cancel();
      schedule();
    });
  }
}
