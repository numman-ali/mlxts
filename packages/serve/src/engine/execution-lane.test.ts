import { describe, expect, test } from "bun:test";

import { GenerationAbortError } from "@mlxts/transformers";
import { ModelExecutionLane } from "./execution-lane";

describe("model execution lane", () => {
  test("serializes work through the default single lane", async () => {
    const lane = new ModelExecutionLane();
    let inFlight = 0;
    let maxInFlight = 0;

    const run = (label: string) =>
      lane.run(async () => {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await Bun.sleep(5);
        inFlight -= 1;
        return label;
      });

    await expect(Promise.all([run("a"), run("b")])).resolves.toEqual(["a", "b"]);
    expect(maxInFlight).toBe(1);
  });

  test("rejects queued work when its abort signal fires", async () => {
    const lane = new ModelExecutionLane();
    const controller = new AbortController();
    let releaseFirst: (() => void) | undefined;
    const first = lane.run(
      () =>
        new Promise<void>((resolve) => {
          releaseFirst = resolve;
        }),
    );
    while (releaseFirst === undefined) {
      await Bun.sleep(0);
    }

    const queued = lane.run(async () => "queued", controller.signal);
    expect(lane.stats()).toEqual({ inFlight: 1, queued: 1, maxConcurrentJobs: 1 });
    controller.abort();
    expect(lane.stats()).toEqual({ inFlight: 1, queued: 0, maxConcurrentJobs: 1 });
    releaseFirst?.();

    await expect(queued).rejects.toBeInstanceOf(GenerationAbortError);
    await expect(first).resolves.toBeUndefined();
  });

  test("honors explicit multi-job capacity", async () => {
    const lane = new ModelExecutionLane(2);
    let inFlight = 0;
    let maxInFlight = 0;

    const run = () =>
      lane.run(async () => {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await Bun.sleep(5);
        inFlight -= 1;
      });

    await Promise.all([run(), run(), run()]);
    expect(maxInFlight).toBe(2);
  });
});
