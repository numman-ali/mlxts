import { describe, expect, test } from "bun:test";

import type {
  ContinuousBatchAdmissionController,
  ContinuousBatchAdmissionRequest,
  ContinuousBatchAdmissionReservation,
} from "@mlxts/transformers";
import { createContinuousSchedulerTokenBudget } from "./continuous-scheduler-budget";

function request(
  totalTokens: number,
  id = `request-${totalTokens}`,
): ContinuousBatchAdmissionRequest {
  return {
    id,
    promptTokens: totalTokens,
    maxTokens: 0,
    totalTokens,
  };
}

function requireBudget(
  maxTotalTokens: number,
  maxBatchSize: number,
): ContinuousBatchAdmissionController {
  const budget = createContinuousSchedulerTokenBudget({ maxTotalTokens, maxBatchSize });
  if (budget === undefined) {
    throw new Error("Expected a continuous scheduler budget.");
  }
  return budget;
}

function reserve(
  budget: ContinuousBatchAdmissionController,
  reservationRequest: ContinuousBatchAdmissionRequest,
): ContinuousBatchAdmissionReservation {
  const decision = budget.tryReserve(reservationRequest);
  if (decision.type !== "reserved") {
    throw new Error(`Expected ${reservationRequest.id} to reserve budget.`);
  }
  return decision.reservation;
}

describe("continuous scheduler token budget", () => {
  test("defers, rejects, and releases aggregate scheduled-token reservations", () => {
    const budget = requireBudget(4, 2);
    const first = reserve(budget, request(5, "first"));

    expect(budget.snapshot()).toEqual({
      scheduledTotalTokens: 5,
      maxScheduledTotalTokens: 8,
    });
    expect(budget.tryReserve(request(4, "second"))).toEqual({
      type: "deferred",
      scheduledTotalTokens: 5,
      maxScheduledTotalTokens: 8,
    });
    expect(budget.tryReserve(request(9, "too-large"))).toEqual({
      type: "rejected",
      message:
        "Continuous scheduler request too-large requires 9 total tokens, exceeding the model-level scheduled token budget of 8.",
    });

    first[Symbol.dispose]();

    expect(budget.tryReserve(request(4, "second"))).toMatchObject({ type: "reserved" });
  });

  test("wakes waiters only on a real reservation release", async () => {
    const budget = requireBudget(50, 2);
    const first = reserve(budget, request(50, "first"));
    let wakeups = 0;

    expect(budget.tryReserve(request(60, "second"))).toEqual({
      type: "deferred",
      scheduledTotalTokens: 50,
      maxScheduledTotalTokens: 100,
    });
    const unsubscribe = budget.onRelease(() => {
      wakeups += 1;
    });

    await Promise.resolve();

    expect(wakeups).toBe(0);

    first[Symbol.dispose]();

    expect(wakeups).toBe(1);

    unsubscribe();
  });

  test("omits the budget when continuous scheduling cannot oversubscribe a model", () => {
    expect(createContinuousSchedulerTokenBudget({ maxBatchSize: 2 })).toBeUndefined();
    expect(
      createContinuousSchedulerTokenBudget({
        maxTotalTokens: 128,
        maxBatchSize: 1,
      }),
    ).toBeUndefined();
  });
});
