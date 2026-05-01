function camelCaseTransformerWeightPath(checkpointName: string): string {
  return checkpointName
    .replaceAll("time_text_embed", "timeTextEmbed")
    .replaceAll("timestep_embedder", "timestepEmbedder")
    .replaceAll("linear_1", "linear1")
    .replaceAll("linear_2", "linear2")
    .replaceAll("txt_norm", "txtNorm")
    .replaceAll("img_in", "imgIn")
    .replaceAll("txt_in", "txtIn")
    .replaceAll("transformer_blocks", "transformerBlocks")
    .replaceAll("img_mod.1", "imgMod.linear")
    .replaceAll("txt_mod.1", "txtMod.linear")
    .replaceAll("attn.to_q", "attn.toQ")
    .replaceAll("attn.to_k", "attn.toK")
    .replaceAll("attn.to_v", "attn.toV")
    .replaceAll("attn.add_q_proj", "attn.addQProj")
    .replaceAll("attn.add_k_proj", "attn.addKProj")
    .replaceAll("attn.add_v_proj", "attn.addVProj")
    .replaceAll("attn.norm_q", "attn.norm.queryNorm")
    .replaceAll("attn.norm_k", "attn.norm.keyNorm")
    .replaceAll("attn.norm_added_q", "attn.addedNorm.queryNorm")
    .replaceAll("attn.norm_added_k", "attn.addedNorm.keyNorm")
    .replaceAll("attn.to_out.0", "attn.toOut")
    .replaceAll("attn.to_add_out", "attn.toAddOut")
    .replaceAll("img_mlp.net.0.proj", "imgMlp.linear1")
    .replaceAll("img_mlp.net.2", "imgMlp.linear2")
    .replaceAll("txt_mlp.net.0.proj", "txtMlp.linear1")
    .replaceAll("txt_mlp.net.2", "txtMlp.linear2")
    .replaceAll("norm_out.linear", "normOut.linear")
    .replaceAll("proj_out", "projOut");
}

/** Map a Diffusers Qwen-Image transformer tensor name onto the package parameter tree. */
export function qwenImageTransformerWeightPath(checkpointName: string): string | null {
  if (checkpointName.trim() === "" || checkpointName.includes("num_batches_tracked")) {
    return null;
  }
  return camelCaseTransformerWeightPath(checkpointName);
}
