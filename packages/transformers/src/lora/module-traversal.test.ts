import { describe, expect, test } from "bun:test";
import { applyLoRAToModule } from "@mlxts/lora";
import { Module } from "@mlxts/nn";

import { createModel, rawConfigForFamily } from "./__test__/fixtures.test";
import {
  collectLinearModulePaths,
  collectLoRAWrapperStates,
  expectCausalLMModule,
  expectTrainableModule,
} from "./module-traversal";

describe("LoRA module traversal", () => {
  test("finds linear paths and active LoRA wrappers on CausalLM modules", () => {
    using model = createModel(rawConfigForFamily("llama"));

    const module = expectTrainableModule(model);
    expect(module).toBeInstanceOf(Module);
    expect(expectCausalLMModule(model)).toBe(module);

    const linearPaths = collectLinearModulePaths(model);
    expect(linearPaths).toContain("model.layers.0.selfAttention.qProjection");
    expect(linearPaths).toContain("lmHead");
    expect(linearPaths.every((path) => !path.includes("embedTokens"))).toBe(true);

    applyLoRAToModule(module, {
      paths: ["model.layers.0.selfAttention.qProjection"],
      rank: 4,
      alpha: 8,
      dropout: 0,
    });

    const wrappers = collectLoRAWrapperStates(model);
    expect(wrappers).toHaveLength(1);
    expect(wrappers[0]?.path).toBe("model.layers.0.selfAttention.qProjection");
    expect(collectLinearModulePaths(model)).not.toContain(
      "model.layers.0.selfAttention.qProjection",
    );
  });
});
