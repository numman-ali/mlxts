import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";
import { Linear } from "./linear";
import { LoRALinear } from "./lora-linear";
import { QuantizedLinear } from "./quantized-linear";

describe("LoRALinear", () => {
  test("matches the base layer before adapters change", () => {
    const base = new Linear(4, 3, true);
    using input = array([[1, 2, 3, 4]], "float32");
    using expected = base.forward(input);
    const lora = LoRALinear.fromBase(base, {
      rank: 2,
      alpha: 4,
      dropout: 0,
    });
    using actual = lora.forward(input);

    mxEval(expected, actual);
    expect(actual.toList()).toEqual(expected.toList());

    lora[Symbol.dispose]();
    base[Symbol.dispose]();
  });

  test("merge returns dense linear for dense bases", () => {
    const base = new Linear(4, 3, true);
    const lora = LoRALinear.fromBase(base, {
      rank: 2,
      alpha: 4,
      dropout: 0,
    });
    lora.loraB.free();
    lora.loraB = array(
      [
        [0.25, -0.5, 0.75],
        [1, -1.25, 1.5],
      ],
      "float32",
    );

    const merged = lora.merge();

    expect(merged).toBeInstanceOf(Linear);

    merged[Symbol.dispose]();
    lora[Symbol.dispose]();
    base[Symbol.dispose]();
  });

  test("merge re-quantizes quantized bases by default", () => {
    const base = new Linear(64, 3, true);
    const quantized = QuantizedLinear.fromLinear(base, {
      bits: 4,
      groupSize: 32,
      mode: "affine",
    });
    const lora = LoRALinear.fromBase(quantized, {
      rank: 2,
      alpha: 4,
      dropout: 0,
    });

    const merged = lora.merge();

    expect(merged).toBeInstanceOf(QuantizedLinear);

    merged[Symbol.dispose]();
    lora[Symbol.dispose]();
    quantized[Symbol.dispose]();
    base[Symbol.dispose]();
  });

  test("takeBase detaches the owned base layer", () => {
    const base = new Linear(4, 3, true);
    const lora = LoRALinear.fromBase(base);

    const detached = lora.takeBase();

    expect(detached).toBe(base);
    expect(lora.linear).toBeNull();

    detached[Symbol.dispose]();
    lora[Symbol.dispose]();
  });
});
