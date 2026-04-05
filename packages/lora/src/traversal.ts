import { Linear, LoRALinear, Module, QuantizedLinear } from "@mlxts/nn";

import type { LoRATargetSlot, LoRAWrapperSlot, ModuleChildSlot } from "./types";

function childPath(prefix: string, key: string): string {
  return prefix === "" ? key : `${prefix}.${key}`;
}

function moduleArray(value: unknown, path: string): Module[] | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  const hasModule = value.some((entry) => entry instanceof Module);
  if (!hasModule) {
    return null;
  }
  if (!value.every((entry) => entry instanceof Module)) {
    throw new Error(`lora: "${path}" mixes Module and non-Module values in one array.`);
  }
  return value;
}

function visitModuleSlot(
  parent: Module,
  key: string,
  path: string,
  child: Module,
  visitor: (slot: ModuleChildSlot) => void,
): void {
  visitor({
    path,
    parent,
    key,
    child,
  });

  if (!(child instanceof LoRALinear)) {
    visitChildModules(child, visitor, path);
  }
}

function visitModuleArray(
  parent: Module,
  key: string,
  path: string,
  children: Module[],
  visitor: (slot: ModuleChildSlot) => void,
): void {
  for (let index = 0; index < children.length; index += 1) {
    const child = children[index];
    if (child === undefined) {
      continue;
    }

    visitModuleSlot(parent, key, childPath(path, String(index)), child, visitor);
  }
}

function visitChildModules(
  module: Module,
  visitor: (slot: ModuleChildSlot) => void,
  prefix = "",
): void {
  for (const key of Object.keys(module)) {
    const value = Reflect.get(module, key);
    const path = childPath(prefix, key);
    if (value instanceof Module) {
      visitModuleSlot(module, key, path, value, visitor);
      continue;
    }

    const entries = moduleArray(value, path);
    if (entries !== null) {
      visitModuleArray(module, key, path, entries, visitor);
    }
  }
}

/** Collect targetable linear slots, excluding already wrapped adapters. */
export function collectLoRATargetSlots(module: Module): LoRATargetSlot[] {
  const slots: LoRATargetSlot[] = [];
  visitChildModules(module, (slot) => {
    if (slot.child instanceof Linear || slot.child instanceof QuantizedLinear) {
      slots.push({
        path: slot.path,
        parent: slot.parent,
        key: slot.key,
        child: slot.child,
      });
    }
  });
  return slots;
}

/** Collect existing LoRA wrapper slots. */
export function collectLoRAWrapperSlots(module: Module): LoRAWrapperSlot[] {
  const slots: LoRAWrapperSlot[] = [];
  visitChildModules(module, (slot) => {
    if (slot.child instanceof LoRALinear) {
      slots.push({
        path: slot.path,
        parent: slot.parent,
        key: slot.key,
        child: slot.child,
      });
    }
  });
  return slots;
}
