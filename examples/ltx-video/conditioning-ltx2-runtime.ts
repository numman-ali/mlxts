import { array, type MxArray, repeat, reshape, stack, transpose } from "@mlxts/core";
import type { Ltx2TextConnectorsConfig } from "@mlxts/diffusion";
import { disposeGemma3TextModelOutput } from "@mlxts/transformers";

import { Ltx2PromptConditioningResult } from "./conditioning-ltx2-result";
import type {
  Ltx2Gemma3TextEncoder,
  Ltx2Prompt,
  Ltx2PromptConditionerComponents,
  Ltx2PromptConditioning,
  Ltx2PromptConditioningOptions,
  Ltx2PromptConnector,
} from "./conditioning-ltx2-types";

type EncodedPrompts = {
  inputIds: MxArray;
  attentionMask: MxArray;
  truncated: boolean;
};

type Ltx2Branch = {
  videoPromptEmbeds: MxArray;
  audioPromptEmbeds: MxArray;
  attentionMask: MxArray;
  truncated: boolean;
};

function promptBatch(value: Ltx2Prompt, name: string): readonly string[] {
  if (typeof value === "string") {
    return [value];
  }
  if (value.length === 0) {
    throw new Error(`${name} must contain at least one prompt.`);
  }
  return value;
}

function matchingPromptBatch(
  value: Ltx2Prompt | undefined,
  fallback: readonly string[],
  name: string,
): readonly string[] {
  if (value === undefined) {
    return Array.from({ length: fallback.length }, () => "");
  }
  if (typeof value === "string") {
    return Array.from({ length: fallback.length }, () => value);
  }
  if (value.length !== fallback.length) {
    throw new Error(`${name} batch size must match prompt batch size.`);
  }
  return value;
}

function resolveNumVideosPerPrompt(value: number | undefined): number {
  const resolved = value ?? 1;
  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new Error("numVideosPerPrompt must be a positive integer.");
  }
  return resolved;
}

function resolveMaxSequenceLength(value: number | undefined): number {
  const resolved = value ?? 1024;
  if (!Number.isInteger(resolved) || resolved <= 0 || resolved > 1024) {
    throw new Error("maxSequenceLength must be a positive integer no greater than 1024.");
  }
  return resolved;
}

function repeatPromptBatch(value: MxArray, numVideosPerPrompt: number): MxArray {
  if (numVideosPerPrompt === 1) {
    return value;
  }
  try {
    return repeat(value, numVideosPerPrompt, 0);
  } finally {
    value.free();
  }
}

function repeatConnectorOutput(
  output: {
    videoPromptEmbeds: MxArray;
    audioPromptEmbeds: MxArray;
    attentionMask: MxArray;
  },
  numVideosPerPrompt: number,
): Omit<Ltx2Branch, "truncated"> {
  const videoPromptEmbeds = repeatPromptBatch(output.videoPromptEmbeds, numVideosPerPrompt);
  try {
    const audioPromptEmbeds = repeatPromptBatch(output.audioPromptEmbeds, numVideosPerPrompt);
    try {
      const attentionMask = repeatPromptBatch(output.attentionMask, numVideosPerPrompt);
      return { videoPromptEmbeds, audioPromptEmbeds, attentionMask };
    } catch (error) {
      audioPromptEmbeds.free();
      throw error;
    }
  } catch (error) {
    videoPromptEmbeds.free();
    throw error;
  }
}

function tokenizerPadId(
  tokenizer: Ltx2PromptConditionerComponents["tokenizer"],
  context: string,
): number {
  const padId = tokenizer.padTokenId ?? tokenizer.eosTokenIds[0];
  if (padId === undefined) {
    throw new Error(`${context}: tokenizer must expose padTokenId or eosTokenIds[0].`);
  }
  return padId;
}

function encodeGemmaPromptIds(
  tokenizer: Ltx2PromptConditionerComponents["tokenizer"],
  prompts: readonly string[],
  maxSequenceLength: number,
): EncodedPrompts {
  const padId = tokenizerPadId(tokenizer, "encodeGemmaPromptIds");
  const inputRows: number[][] = [];
  const maskRows: number[][] = [];
  let truncated = false;

  for (const prompt of prompts) {
    const rawIds = tokenizer.encode(prompt, { addSpecialTokens: true });
    const tokenIds =
      rawIds.length > maxSequenceLength ? rawIds.slice(0, maxSequenceLength) : rawIds;
    truncated = truncated || rawIds.length > maxSequenceLength;
    const padLength = maxSequenceLength - tokenIds.length;
    inputRows.push([...Array.from({ length: padLength }, () => padId), ...tokenIds]);
    maskRows.push([
      ...Array.from({ length: padLength }, () => 0),
      ...Array.from({ length: tokenIds.length }, () => 1),
    ]);
  }

  return {
    inputIds: array(inputRows, "int32"),
    attentionMask: array(maskRows, "int32"),
    truncated,
  };
}

function expectedTextHiddenSize(config: Ltx2TextConnectorsConfig): number {
  if (config.textEncoderDim % config.textProjInFactor !== 0) {
    throw new Error("LTX-2 connector textEncoderDim must be divisible by textProjInFactor.");
  }
  return config.textEncoderDim / config.textProjInFactor;
}

function retainFlattenedGemmaHiddenStack(
  hiddenStates: readonly MxArray[],
  config: Ltx2TextConnectorsConfig,
): MxArray {
  const expectedLayers = config.textProjInFactor;
  if (hiddenStates.length !== expectedLayers) {
    throw new Error(
      `LTX-2 prompt conditioning expected ${expectedLayers} Gemma hidden states, got ${hiddenStates.length}.`,
    );
  }
  const first = hiddenStates[0];
  if (first === undefined) {
    throw new Error("LTX-2 prompt conditioning received no Gemma hidden states.");
  }
  const [batch, sequenceLength, hiddenSize] = first.shape;
  const expectedHiddenSize = expectedTextHiddenSize(config);
  if (
    first.shape.length !== 3 ||
    batch === undefined ||
    sequenceLength === undefined ||
    hiddenSize !== expectedHiddenSize
  ) {
    throw new Error(`LTX-2 Gemma hidden states must be [batch, sequence, ${expectedHiddenSize}].`);
  }

  for (const hiddenState of hiddenStates) {
    if (
      hiddenState.shape.length !== 3 ||
      hiddenState.shape[0] !== batch ||
      hiddenState.shape[1] !== sequenceLength ||
      hiddenState.shape[2] !== expectedHiddenSize
    ) {
      throw new Error("LTX-2 Gemma hidden states must share batch, sequence, and hidden size.");
    }
  }

  using layerMajor = stack([...hiddenStates], 1);
  using tokenMajor = transpose(layerMajor, [0, 2, 3, 1]);
  return reshape(tokenMajor, [batch, sequenceLength, config.textEncoderDim]);
}

function encodeGemmaBranch(
  tokenizer: Ltx2PromptConditionerComponents["tokenizer"],
  textEncoder: Ltx2Gemma3TextEncoder,
  connectors: Ltx2PromptConnector,
  prompts: readonly string[],
  maxSequenceLength: number,
  numVideosPerPrompt: number,
): Ltx2Branch {
  const encoded = encodeGemmaPromptIds(tokenizer, prompts, maxSequenceLength);
  try {
    const output = textEncoder.model.runWithHiddenStates(encoded.inputIds, {
      attentionMask: encoded.attentionMask,
      outputHiddenStates: true,
    });
    try {
      if (output.hiddenStates === undefined) {
        throw new Error("LTX-2 prompt conditioning requires Gemma hidden states.");
      }
      using textEncoderHiddenStates = retainFlattenedGemmaHiddenStack(
        output.hiddenStates,
        connectors.config,
      );
      const connectorOutput = connectors.run(textEncoderHiddenStates, encoded.attentionMask);
      const repeated = repeatConnectorOutput(connectorOutput, numVideosPerPrompt);
      return { ...repeated, truncated: encoded.truncated };
    } finally {
      disposeGemma3TextModelOutput(output);
    }
  } finally {
    encoded.inputIds.free();
    encoded.attentionMask.free();
  }
}

/** Encode positive and optional negative prompt conditioning for LTX-2. */
export function encodeLtx2Prompt(
  components: Ltx2PromptConditionerComponents,
  options: Ltx2PromptConditioningOptions,
): Ltx2PromptConditioning {
  const prompts = promptBatch(options.prompt, "prompt");
  const numVideosPerPrompt = resolveNumVideosPerPrompt(options.numVideosPerPrompt);
  const maxSequenceLength = resolveMaxSequenceLength(options.maxSequenceLength);
  const batchSize = prompts.length * numVideosPerPrompt;
  const prompt = encodeGemmaBranch(
    components.tokenizer,
    components.textEncoder,
    components.connectors,
    prompts,
    maxSequenceLength,
    numVideosPerPrompt,
  );
  let negative: Ltx2Branch | undefined;
  try {
    if (options.includeNegativePrompt === true) {
      negative = encodeGemmaBranch(
        components.tokenizer,
        components.textEncoder,
        components.connectors,
        matchingPromptBatch(options.negativePrompt, prompts, "negativePrompt"),
        maxSequenceLength,
        numVideosPerPrompt,
      );
    }
    return new Ltx2PromptConditioningResult(
      batchSize,
      {
        promptEmbeds: prompt.videoPromptEmbeds,
        audioPromptEmbeds: prompt.audioPromptEmbeds,
        promptAttentionMask: prompt.attentionMask,
        ...(negative === undefined
          ? {}
          : {
              negativePromptEmbeds: negative.videoPromptEmbeds,
              negativeAudioPromptEmbeds: negative.audioPromptEmbeds,
              negativePromptAttentionMask: negative.attentionMask,
            }),
      },
      prompt.truncated,
      negative?.truncated ?? false,
    );
  } catch (error) {
    prompt.videoPromptEmbeds.free();
    prompt.audioPromptEmbeds.free();
    prompt.attentionMask.free();
    negative?.videoPromptEmbeds.free();
    negative?.audioPromptEmbeds.free();
    negative?.attentionMask.free();
    throw error;
  }
}
