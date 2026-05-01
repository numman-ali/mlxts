import {
  add,
  type MxArray,
  multiply,
  random,
  reshape,
  retainArray,
  split,
  squeeze,
} from "@mlxts/core";

import { freeArrays } from "./tensor-utils";

export type Ltx2BlockModulation = {
  shiftMsa: MxArray;
  scaleMsa: MxArray;
  gateMsa: MxArray;
  shiftMlp: MxArray;
  scaleMlp: MxArray;
  gateMlp: MxArray;
};

export type Ltx2CrossModulation = {
  a2vScale: MxArray;
  a2vShift: MxArray;
  v2aScale: MxArray;
  v2aShift: MxArray;
  gate: MxArray;
};

export function scaledNormal(shape: number[], scale: number): MxArray {
  using values = random.normal(shape);
  return multiply(values, scale);
}

export function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: split failed.`);
  }
  return part;
}

export function disposeBlockModulation(modulation: Ltx2BlockModulation): void {
  modulation.shiftMsa.free();
  modulation.scaleMsa.free();
  modulation.gateMsa.free();
  modulation.shiftMlp.free();
  modulation.scaleMlp.free();
  modulation.gateMlp.free();
}

export function disposeCrossModulation(modulation: Ltx2CrossModulation): void {
  modulation.a2vScale.free();
  modulation.a2vShift.free();
  modulation.v2aScale.free();
  modulation.v2aShift.free();
  modulation.gate.free();
}

export function disposeBlockOutput(output: {
  hiddenStates: MxArray;
  audioHiddenStates: MxArray;
}): void {
  output.hiddenStates.free();
  output.audioHiddenStates.free();
}

export function modParams(
  table: MxArray,
  temb: MxArray,
  batch: number,
  hiddenSize: number,
  count: number,
  owner: string,
): MxArray[] {
  if (
    table.shape.length !== 2 ||
    table.shape[0] !== count ||
    table.shape[1] !== hiddenSize ||
    temb.shape.length !== 3 ||
    temb.shape[0] !== batch ||
    temb.shape[2] !== count * hiddenSize
  ) {
    throw new Error(`${owner}: modulation shape mismatch.`);
  }
  const sequenceLength = temb.shape[1];
  if (sequenceLength === undefined) {
    throw new Error(`${owner}: modulation sequence length is undefined.`);
  }
  using tableView = reshape(table, [1, 1, count, hiddenSize]);
  using tembView = reshape(temb, [batch, sequenceLength, count, hiddenSize]);
  using values = add(tableView, tembView);
  const parts = split(values, count, 2);
  const retained: MxArray[] = [];
  try {
    for (let index = 0; index < count; index += 1) {
      using squeezed = squeeze(partAt(parts, index, owner), 2);
      retained.push(retainArray(squeezed));
    }
    return retained;
  } catch (error) {
    freeArrays(retained);
    throw error;
  } finally {
    freeArrays(parts);
  }
}
