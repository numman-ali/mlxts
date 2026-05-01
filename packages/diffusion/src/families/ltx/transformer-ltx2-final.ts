import { add, fastLayerNorm, type MxArray, reshape, split } from "@mlxts/core";
import type { Linear } from "@mlxts/nn";

import { applyScaleShift, freeArrays, reshapeModulation } from "./tensor-utils";
import { partAt } from "./transformer-ltx2-state";

type Ltx2FinalModulation = {
  shift: MxArray;
  scale: MxArray;
};

function disposeFinalModulation(modulation: Ltx2FinalModulation): void {
  modulation.shift.free();
  modulation.scale.free();
}

function finalModulation(
  embeddedTimestep: MxArray,
  scaleShiftTable: MxArray,
  hiddenSize: number,
  name: string,
): Ltx2FinalModulation {
  if (
    embeddedTimestep.shape.length !== 3 ||
    embeddedTimestep.shape[1] !== 1 ||
    embeddedTimestep.shape[2] !== hiddenSize
  ) {
    throw new Error(`Ltx2VideoTransformer3DModel.${name}FinalModulation: timestep mismatch.`);
  }
  using table = reshape(scaleShiftTable, [1, 2, hiddenSize]);
  using values = add(table, embeddedTimestep);
  const parts = split(values, 2, 1);
  try {
    const shift = reshapeModulation(
      partAt(parts, 0, `Ltx2VideoTransformer3DModel.${name}FinalModulation`),
      embeddedTimestep.shape[0] ?? 0,
      hiddenSize,
    );
    try {
      const scale = reshapeModulation(
        partAt(parts, 1, `Ltx2VideoTransformer3DModel.${name}FinalModulation`),
        embeddedTimestep.shape[0] ?? 0,
        hiddenSize,
      );
      return { shift, scale };
    } catch (error) {
      shift.free();
      throw error;
    }
  } finally {
    freeArrays(parts);
  }
}

export function projectLtx2FinalOutput(options: {
  hiddenStates: MxArray;
  embeddedTimestep: MxArray;
  scaleShiftTable: MxArray;
  projection: Linear;
  hiddenSize: number;
  name: string;
}): MxArray {
  const modulation = finalModulation(
    options.embeddedTimestep,
    options.scaleShiftTable,
    options.hiddenSize,
    options.name,
  );
  try {
    using normalized = fastLayerNorm(options.hiddenStates, undefined, undefined, { eps: 1e-6 });
    using modulated = applyScaleShift(normalized, modulation.shift, modulation.scale);
    return options.projection.forward(modulated);
  } finally {
    disposeFinalModulation(modulation);
  }
}
