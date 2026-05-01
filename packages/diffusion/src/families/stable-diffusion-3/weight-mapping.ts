function camelCaseTransformerWeightPath(checkpointName: string): string {
  return checkpointName
    .replaceAll("pos_embed.proj", "posEmbed.projection")
    .replaceAll("time_text_embed.timestep_embedder.linear_1", "timeTextEmbed.timestepLinear1")
    .replaceAll("time_text_embed.timestep_embedder.linear_2", "timeTextEmbed.timestepLinear2")
    .replaceAll("time_text_embed.text_embedder.linear_1", "timeTextEmbed.textEmbedder.linear1")
    .replaceAll("time_text_embed.text_embedder.linear_2", "timeTextEmbed.textEmbedder.linear2")
    .replaceAll("context_embedder", "contextEmbedder")
    .replaceAll("transformer_blocks", "transformerBlocks")
    .replaceAll("norm1_context", "norm1Context")
    .replaceAll("attn2.", "attention2.")
    .replaceAll("attn.", "attention.")
    .replaceAll("add_q_proj", "addQProj")
    .replaceAll("add_k_proj", "addKProj")
    .replaceAll("add_v_proj", "addVProj")
    .replaceAll("to_q", "toQ")
    .replaceAll("to_k", "toK")
    .replaceAll("to_v", "toV")
    .replaceAll("to_out.0", "toOut")
    .replaceAll("to_add_out", "toAddOut")
    .replaceAll("norm_added_q", "normAddedQ")
    .replaceAll("norm_added_k", "normAddedK")
    .replaceAll("norm_q", "normQ")
    .replaceAll("norm_k", "normK")
    .replaceAll("ff_context.net.0.proj", "ffContext.linear1")
    .replaceAll("ff_context.net.2", "ffContext.linear2")
    .replaceAll("ff.net.0.proj", "ff.linear1")
    .replaceAll("ff.net.2", "ff.linear2")
    .replaceAll("norm_out.linear", "normOut.linear")
    .replaceAll("proj_out", "projOut");
}

function camelCaseAutoencoderWeightPath(checkpointName: string): string {
  return checkpointName
    .replaceAll("down_blocks", "downBlocks")
    .replaceAll("up_blocks", "upBlocks")
    .replaceAll("conv_in", "convIn")
    .replaceAll("conv_norm_out", "convNormOut")
    .replaceAll("conv_out", "convOut")
    .replaceAll("conv_shortcut", "convShortcut")
    .replaceAll("post_quant_conv", "postQuantConv")
    .replaceAll("quant_conv", "quantConv")
    .replaceAll("downsamplers.0.conv", "downsample.conv")
    .replaceAll("upsamplers.0.conv", "upsample.conv")
    .replaceAll("mid_block.resnets.0", "midBlock.resnetIn")
    .replaceAll("mid_block.attentions.0", "midBlock.attention")
    .replaceAll("mid_block.resnets.1", "midBlock.resnetOut")
    .replaceAll("group_norm", "groupNorm")
    .replaceAll("to_q", "queryProjection")
    .replaceAll("to_k", "keyProjection")
    .replaceAll("to_v", "valueProjection")
    .replaceAll("to_out.0", "outputProjection");
}

/** Return whether a Diffusers SD3 transformer tensor is reproduced from config. */
export function isIgnoredStableDiffusion3TransformerWeight(checkpointName: string): boolean {
  return checkpointName === "pos_embed.pos_embed";
}

/** Map a Diffusers SD3 transformer tensor name onto the package parameter tree. */
export function stableDiffusion3TransformerWeightPath(checkpointName: string): string | null {
  if (
    checkpointName.trim() === "" ||
    checkpointName.includes("num_batches_tracked") ||
    isIgnoredStableDiffusion3TransformerWeight(checkpointName)
  ) {
    return null;
  }
  return camelCaseTransformerWeightPath(checkpointName);
}

/** Map a Diffusers SD3 VAE tensor name onto the package parameter tree. */
export function stableDiffusion3AutoencoderWeightPath(checkpointName: string): string | null {
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }
  return camelCaseAutoencoderWeightPath(checkpointName);
}
