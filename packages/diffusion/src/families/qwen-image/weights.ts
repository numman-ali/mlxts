import { contiguous, type MxArray, reshape, transpose } from "@mlxts/core";

function hasOnlySingletonTrailingDimensions(shape: readonly number[]): boolean {
  return shape.length > 1 && shape.slice(1).every((dimension) => dimension === 1);
}

function camelCaseAutoencoderWeightPath(checkpointName: string): string {
  return checkpointName
    .replaceAll("post_quant_conv", "postQuantConv")
    .replaceAll("quant_conv", "quantConv")
    .replaceAll("down_blocks", "downBlocks")
    .replaceAll("up_blocks", "upBlocks")
    .replaceAll("mid_block", "midBlock")
    .replaceAll("conv_in", "convIn")
    .replaceAll("conv_out", "convOut")
    .replaceAll("conv_shortcut", "convShortcut")
    .replaceAll("norm_out", "normOut")
    .replaceAll("time_conv", "timeConv")
    .replaceAll("to_qkv", "toQkv")
    .replaceAll("upsamplers.0", "upsampler")
    .replaceAll("resample.1", "resample")
    .replaceAll(".gamma", ".weight");
}

/** Map a Diffusers Qwen-Image VAE tensor name onto the package parameter tree. */
export function qwenImageAutoencoderWeightPath(checkpointName: string): string | null {
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }
  return camelCaseAutoencoderWeightPath(checkpointName);
}

/** Transform a Diffusers Qwen-Image VAE tensor into the package-owned layout. */
export function transformQwenImageAutoencoderWeight(
  checkpointName: string,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  if (
    checkpointName.endsWith(".gamma") &&
    weightPath.endsWith(".weight") &&
    hasOnlySingletonTrailingDimensions(tensor.shape)
  ) {
    const channels = tensor.shape[0];
    if (channels === undefined) {
      throw new Error("transformQwenImageAutoencoderWeight: gamma tensor is missing channels.");
    }
    return reshape(tensor, [channels]);
  }

  if (weightPath.endsWith(".weight") && tensor.shape.length === 5) {
    using transposed = transpose(tensor, [0, 2, 3, 4, 1]);
    return contiguous(transposed);
  }

  if (weightPath.endsWith(".weight") && tensor.shape.length === 4) {
    using transposed = transpose(tensor, [0, 2, 3, 1]);
    return contiguous(transposed);
  }

  return tensor;
}
