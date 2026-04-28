import { type Module, QuantizedLinear } from "@mlxts/nn";

import { collectLoRATargetSlots } from "./traversal";

/** Assert that a merged QLoRA target still owns its quantized base module. */
export function assertQuantizedBasePreserved(module: Module, path: string): QuantizedLinear {
  const slot = collectLoRATargetSlots(module).find((target) => target.path === path);
  if (slot === undefined) {
    throw new Error(
      `lora: expected a quantized base at "${path}", but no linear target was found.`,
    );
  }
  if (!(slot.child instanceof QuantizedLinear)) {
    throw new Error(
      `lora: expected a quantized base at "${path}", got ${slot.child.constructor.name}.`,
    );
  }
  return slot.child;
}
