import type { InteractionProfile, loadPretrainedTokenizer } from "@mlxts/transformers";

export type LoadedAssets = {
  tokenizer: Awaited<ReturnType<typeof loadPretrainedTokenizer>>;
  profile: InteractionProfile;
};
