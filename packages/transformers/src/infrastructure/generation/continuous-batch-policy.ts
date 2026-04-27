import { integerAtLeast, remainingPrefillTokens } from "./continuous-batch-helpers";
import type { PrefillingRequest } from "./continuous-batch-types";
import { validatePrefillStepSize } from "./helpers";

export function resolveActivePrefillStepSize(
  prefillStepSize: number,
  activePrefillStepSize: number | undefined,
): number {
  const resolved = activePrefillStepSize ?? Math.min(prefillStepSize, 512);
  validatePrefillStepSize(resolved, "ContinuousBatchTokenScheduler");
  return resolved;
}

export function resolveActiveDecodeStepsPerPrefillChunk(
  activeDecodeStepsPerPrefillChunk: number | undefined,
): number {
  return integerAtLeast(
    activeDecodeStepsPerPrefillChunk ?? 8,
    "activeDecodeStepsPerPrefillChunk",
    1,
  );
}

export function nextDecodeStepsSincePrefill(
  current: number,
  hadPendingPrefillWork: boolean,
): number {
  return hadPendingPrefillWork ? current + 1 : current;
}

export function shouldPrioritizeActiveDecode(options: {
  activeCount: number;
  hasCurrentToken: boolean;
  waitingCapacity: number;
  waiting: readonly { promptTokenIds: readonly number[] }[];
  prefilling: readonly PrefillingRequest[];
  activePrefillStepSize: number;
  decodeStepsSincePrefill: number;
  activeDecodeStepsPerPrefillChunk: number;
}): boolean {
  return (
    options.activeCount > 0 &&
    options.hasCurrentToken &&
    hasPendingPrefillWork(options.waiting, options.prefilling) &&
    !hasShortPendingPrefillWork(
      options.waiting,
      options.prefilling,
      options.waitingCapacity,
      options.activePrefillStepSize,
    ) &&
    options.decodeStepsSincePrefill < options.activeDecodeStepsPerPrefillChunk
  );
}

function hasPendingPrefillWork(
  waiting: readonly { promptTokenIds: readonly number[] }[],
  prefilling: readonly PrefillingRequest[],
): boolean {
  return waiting.length > 0 || prefilling.length > 0;
}

function hasShortPendingPrefillWork(
  waiting: readonly { promptTokenIds: readonly number[] }[],
  prefilling: readonly PrefillingRequest[],
  waitingCapacity: number,
  activePrefillStepSize: number,
): boolean {
  return (
    (waitingCapacity > 0 &&
      waiting[0] !== undefined &&
      waiting[0].promptTokenIds.length - 1 <= activePrefillStepSize) ||
    prefilling.some((prefilling) => remainingPrefillTokens(prefilling) <= activePrefillStepSize)
  );
}
