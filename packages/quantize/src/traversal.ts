import { Linear, Module, QuantizedLinear } from "@mlxts/nn";

import type { LinearChildSlot, ModuleChildSlot } from "./types";

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
    throw new Error(`quantize: "${path}" mixes Module and non-Module values in one array.`);
  }
  return value;
}

/** Visit direct child module slots in stable enumerable-key order. */
export function visitChildModules(
  module: Module,
  visitor: (slot: ModuleChildSlot) => void,
  prefix = "",
): void {
  for (const key of Object.keys(module)) {
    const value = Reflect.get(module, key);
    const path = childPath(prefix, key);
    if (value instanceof Module) {
      visitor({
        path,
        parent: module,
        key,
        child: value,
      });
      visitChildModules(value, visitor, path);
      continue;
    }

    const entries = moduleArray(value, path);
    if (entries === null) {
      continue;
    }

    for (let index = 0; index < entries.length; index += 1) {
      const child = entries[index];
      if (child === undefined) {
        continue;
      }
      const indexedPath = childPath(path, String(index));
      visitor({
        path: indexedPath,
        parent: module,
        key,
        child,
      });
      visitChildModules(child, visitor, indexedPath);
    }
  }
}

/** Visit direct child linear slots in stable enumerable-key order. */
export function visitLinearChildren(
  module: Module,
  visitor: (slot: LinearChildSlot) => void,
  prefix = "",
): void {
  visitChildModules(
    module,
    (slot) => {
      if (!(slot.child instanceof Linear || slot.child instanceof QuantizedLinear)) {
        return;
      }

      visitor({
        path: slot.path,
        parent: slot.parent,
        key: slot.key,
        child: slot.child,
      });
    },
    prefix,
  );
}
