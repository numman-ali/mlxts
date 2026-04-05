/** One token-level supervision example for next-token training. */
export type TokenSupervisionExample = {
  inputIds: readonly number[];
  targetIds: readonly number[];
  lossMask?: readonly number[];
};

/** One prompt/completion preference example for alignment training. */
export type PreferenceExample = {
  promptIds: readonly number[];
  chosenIds: readonly number[];
  rejectedIds: readonly number[];
};
