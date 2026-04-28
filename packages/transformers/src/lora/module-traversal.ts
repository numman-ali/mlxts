import { Linear, LoRALinear, Module, QuantizedLinear } from "@mlxts/nn";

import type { CausalLM } from "../types";

type ModuleSlot = {
  path: string;
  child: Module;
};

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
    throw new Error(`transformers: "${path}" mixes Module and non-Module values in one array.`);
  }

  return value;
}

function visitModuleSlot(path: string, child: Module, visitor: (slot: ModuleSlot) => void): void {
  visitor({ path, child });

  if (!(child instanceof LoRALinear)) {
    visitChildModules(child, visitor, path);
  }
}

function visitModuleArray(
  path: string,
  children: Module[],
  visitor: (slot: ModuleSlot) => void,
): void {
  for (let index = 0; index < children.length; index += 1) {
    const child = children[index];
    if (child === undefined) {
      continue;
    }

    visitModuleSlot(childPath(path, String(index)), child, visitor);
  }
}

function visitChildModules(module: Module, visitor: (slot: ModuleSlot) => void, prefix = ""): void {
  for (const key of Object.keys(module)) {
    const value = Reflect.get(module, key);
    const path = childPath(prefix, key);
    if (value instanceof Module) {
      visitModuleSlot(path, value, visitor);
      continue;
    }

    const entries = moduleArray(value, path);
    if (entries !== null) {
      visitModuleArray(path, entries, visitor);
    }
  }
}

/** Narrow a loaded CausalLM to its trainable nn.Module implementation. */
export function expectTrainableModule(model: CausalLM): Module {
  if (!(model instanceof Module)) {
    throw new Error("transformers: expected a loaded CausalLM backed by nn.Module.");
  }
  return model;
}

export function expectCausalLMModule(model: CausalLM): Module {
  return expectTrainableModule(model);
}

export function collectLinearModulePaths(model: CausalLM): string[] {
  const module = expectCausalLMModule(model);
  const paths: string[] = [];
  visitChildModules(module, (slot) => {
    if (slot.child instanceof Linear || slot.child instanceof QuantizedLinear) {
      paths.push(slot.path);
    }
  });
  return paths;
}

export type LoRAWrapperState = {
  path: string;
  child: LoRALinear;
};

export function collectLoRAWrapperStates(model: CausalLM): LoRAWrapperState[] {
  const module = expectCausalLMModule(model);
  const states: LoRAWrapperState[] = [];
  visitChildModules(module, (slot) => {
    if (slot.child instanceof LoRALinear) {
      states.push({
        path: slot.path,
        child: slot.child,
      });
    }
  });
  return states;
}
