import { describe, expect, test } from "bun:test";

import type {
  ContinuousBatchAdmissionController,
  ContinuousBatchAdmissionRequest,
  ContinuousBatchAdmissionReservation,
} from "@mlxts/transformers";
import { createContinuousSchedulerTokenBudget } from "./continuous-budget";

function request(
  promptTokens: number,
  maxTokens: number,
  id = `request-${promptTokens}-${maxTokens}`,
): ContinuousBatchAdmissionRequest {
  return {
    id,
    promptTokens,
    maxTokens,
    totalTokens: promptTokens + maxTokens,
  };
}

function requireBudget(
  options: Omit<Parameters<typeof createContinuousSchedulerTokenBudget>[0], "maxBatchSize">,
  maxBatchSize: number,
): ContinuousBatchAdmissionController {
  const budget = createContinuousSchedulerTokenBudget({ ...options, maxBatchSize });
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
  test("defers, rejects, and releases split scheduled-token reservations", () => {
    const budget = requireBudget({ maxPromptTokens: 4, maxGeneratedTokens: 2 }, 2);
    const first = reserve(budget, request(4, 1, "first"));

    expect(budget.snapshot()).toEqual({
      scheduledPromptTokens: 4,
      maxScheduledPromptTokens: 8,
      scheduledCompletionTokens: 1,
      maxScheduledCompletionTokens: 4,
      scheduledTotalTokens: 5,
      maxScheduledTotalTokens: 12,
      scheduledMemoryBytes: 0,
      maxScheduledMemoryBytes: null,
    });
    expect(budget.tryReserve(request(1, 4, "completion-bound"))).toEqual({
      type: "deferred",
      reason: "scheduled_completion_budget",
      scheduledPromptTokens: 4,
      maxScheduledPromptTokens: 8,
      scheduledCompletionTokens: 1,
      maxScheduledCompletionTokens: 4,
      scheduledTotalTokens: 5,
      maxScheduledTotalTokens: 12,
      scheduledMemoryBytes: 0,
      maxScheduledMemoryBytes: null,
    });
    expect(budget.tryReserve(request(9, 1, "too-large-prompt"))).toEqual({
      type: "rejected",
      message:
        "Continuous scheduler request too-large-prompt requires 9 prompt tokens, exceeding the model-level scheduled prompt token budget of 8.",
    });
    expect(budget.tryReserve(request(1, 5, "too-large-completion"))).toEqual({
      type: "rejected",
      message:
        "Continuous scheduler request too-large-completion requires 5 completion tokens, exceeding the model-level scheduled completion token budget of 4.",
    });
    expect(
      requireBudget(
        { maxPromptTokens: 8, maxGeneratedTokens: 8, maxTotalTokens: 10 },
        2,
      ).tryReserve(request(12, 9, "too-large-total")),
    ).toEqual({
      type: "rejected",
      message:
        "Continuous scheduler request too-large-total requires 21 total tokens, exceeding the model-level scheduled total token budget of 20.",
    });

    first[Symbol.dispose]();

    expect(budget.tryReserve(request(1, 4, "completion-bound"))).toMatchObject({
      type: "reserved",
    });
  });

  test("wakes waiters only on a real reservation release", async () => {
    const budget = requireBudget({ maxTotalTokens: 50 }, 2);
    const first = reserve(budget, request(40, 10, "first"));
    let wakeups = 0;

    expect(budget.tryReserve(request(60, 0, "second"))).toEqual({
      type: "deferred",
      reason: "scheduled_token_budget",
      scheduledPromptTokens: 40,
      maxScheduledPromptTokens: 100,
      scheduledCompletionTokens: 10,
      maxScheduledCompletionTokens: 100,
      scheduledTotalTokens: 50,
      maxScheduledTotalTokens: 100,
      scheduledMemoryBytes: 0,
      maxScheduledMemoryBytes: null,
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

  test("defers, rejects, and releases scheduled-memory reservations", () => {
    const budget = requireBudget(
      {
        maxScheduledMemoryBytes: 100,
        estimateMemoryBytes(request) {
          return request.totalTokens * 10;
        },
      },
      2,
    );
    const first = reserve(budget, request(4, 1, "first"));

    expect(budget.snapshot()).toEqual({
      scheduledPromptTokens: 4,
      maxScheduledPromptTokens: null,
      scheduledCompletionTokens: 1,
      maxScheduledCompletionTokens: null,
      scheduledTotalTokens: 5,
      maxScheduledTotalTokens: null,
      scheduledMemoryBytes: 50,
      maxScheduledMemoryBytes: 100,
    });
    expect(budget.tryReserve(request(4, 2, "memory-bound"))).toEqual({
      type: "deferred",
      reason: "scheduled_memory_budget",
      scheduledPromptTokens: 4,
      maxScheduledPromptTokens: null,
      scheduledCompletionTokens: 1,
      maxScheduledCompletionTokens: null,
      scheduledTotalTokens: 5,
      maxScheduledTotalTokens: null,
      scheduledMemoryBytes: 50,
      maxScheduledMemoryBytes: 100,
    });
    expect(budget.tryReserve(request(10, 1, "too-large-memory"))).toEqual({
      type: "rejected",
      message:
        "Continuous scheduler request too-large-memory requires estimated memory 110 B, exceeding the model-level scheduled memory budget of 100 B.",
    });

    first[Symbol.dispose]();

    expect(budget.tryReserve(request(4, 2, "memory-bound"))).toMatchObject({
      type: "reserved",
    });
  });
});
