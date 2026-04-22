import { type ChatMessage, loadHuggingFaceRowsDataset } from "@mlxts/data";

import type { TrainingProofArgs } from "./args";

/** Proof chat corpus used for the tiny local smoke path. */
export type TrainingProofCorpus = {
  supervisionExamples: readonly ChatMessage[][];
  promptMessages: readonly ChatMessage[];
  chosen: ChatMessage;
  rejected: ChatMessage;
};

export type ParsedPreferenceConversation = {
  promptMessages: readonly ChatMessage[];
  chosen: ChatMessage;
  rejected: ChatMessage;
};

export type TrainingProofRawDatasets = {
  supervisionTrainMessages: readonly ChatMessage[][];
  supervisionEvalMessages: readonly ChatMessage[][];
  preferenceTrainRows: readonly ParsedPreferenceConversation[];
  preferenceEvalRows: readonly ParsedPreferenceConversation[];
  samplePromptMessages: readonly ChatMessage[];
  notes: string[];
};

const ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k";
const ULTRAFEEDBACK_DATASET = "HuggingFaceH4/ultrafeedback_binarized";
const DATASET_OVERSAMPLE_FACTOR = 4;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function expectObject(value: unknown, context: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`${context} must be an object.`);
  }
  return value;
}

function expectString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context} must be a string.`);
  }
  return value;
}

function parseChatRole(value: unknown, context: string): ChatMessage["role"] {
  const role = expectString(value, context);
  if (role === "system" || role === "user" || role === "assistant") {
    return role;
  }
  throw new Error(`${context} must be one of system, user, or assistant.`);
}

function parseChatMessages(value: unknown, context: string): ChatMessage[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context} must be an array of chat messages.`);
  }

  return value.map((entry, index) => {
    const record = expectObject(entry, `${context}[${index}]`);
    return {
      role: parseChatRole(record.role, `${context}[${index}].role`),
      content: expectString(record.content, `${context}[${index}].content`),
    };
  });
}

function expectAssistant(messages: readonly ChatMessage[], context: string): void {
  if (messages.at(-1)?.role !== "assistant") {
    throw new Error(`${context} must end in an assistant message.`);
  }
}

function samePromptPrefix(left: readonly ChatMessage[], right: readonly ChatMessage[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  return left.every(
    (message, index) =>
      message.role === right[index]?.role && message.content === right[index]?.content,
  );
}

function repeatToLength<T>(items: readonly T[], length: number): T[] {
  const repeated: T[] = [];
  for (let index = 0; index < length; index += 1) {
    const item = items[index % items.length];
    if (item === undefined) {
      throw new Error("training proof: expected at least one item to repeat.");
    }
    repeated.push(item);
  }
  return repeated;
}

async function runCommand(args: readonly string[]): Promise<string> {
  const process = Bun.spawn(args, {
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
    process.exited,
  ]);
  if (exitCode !== 0) {
    throw new Error(stderr.trim() === "" ? `${args[0]} failed.` : stderr.trim());
  }
  return stdout.trim();
}

async function fetchDatasetParquetPaths(dataset: string): Promise<string[]> {
  const response = await fetch(`https://huggingface.co/api/datasets/${dataset}`);
  if (!response.ok) {
    throw new Error(
      `training proof: failed to inspect dataset ${dataset}: ${response.status} ${response.statusText}.`,
    );
  }

  const payload: unknown = await response.json();
  if (!isRecord(payload)) {
    throw new Error(`training proof: expected dataset metadata for ${dataset}.`);
  }

  if (!Array.isArray(payload.siblings)) {
    throw new Error(`training proof: dataset ${dataset} metadata did not expose siblings.`);
  }

  const files = payload.siblings
    .map((entry) => (isRecord(entry) ? entry.rfilename : undefined))
    .filter((path): path is string => typeof path === "string" && path.endsWith(".parquet"))
    .sort((left, right) => left.localeCompare(right));
  if (files.length === 0) {
    throw new Error(`training proof: dataset ${dataset} did not expose parquet files.`);
  }
  return files;
}

async function resolveDatasetSplitParquet(dataset: string, split: string): Promise<string> {
  const parquetFiles = await fetchDatasetParquetPaths(dataset);
  const match = parquetFiles.find((path) => path.startsWith(`data/${split}-`));
  if (match === undefined) {
    throw new Error(`training proof: dataset ${dataset} is missing a parquet shard for ${split}.`);
  }
  return match;
}

function escapeSqlString(value: string): string {
  return value.replaceAll("'", "''");
}

async function loadDatasetRowsFromParquet<T>(
  dataset: string,
  split: string,
  selectSql: string,
  parseRow: (row: unknown) => T,
  length: number,
): Promise<T[]> {
  const relativePath = await resolveDatasetSplitParquet(dataset, split);
  const localPath = await runCommand([
    "hf",
    "download",
    "--repo-type",
    "dataset",
    dataset,
    relativePath,
    "--quiet",
  ]);
  const sql = `SELECT ${selectSql} FROM read_parquet('${escapeSqlString(localPath)}') LIMIT ${length}`;
  const payload = await runCommand(["duckdb", "-json", "-c", sql]);
  const rows: unknown = JSON.parse(payload);
  if (!Array.isArray(rows)) {
    throw new Error(`training proof: expected duckdb JSON rows for ${dataset}:${split}.`);
  }
  return rows.map((row) => parseRow(row));
}

/** Canonical small chat corpus for the tiny proof runner fallback. */
export function createTrainingProofCorpus(): TrainingProofCorpus {
  return {
    supervisionExamples: [
      [
        { role: "system", content: "You are concise, exact, and helpful." },
        { role: "user", content: "Explain LoRA in one sentence." },
        {
          role: "assistant",
          content: "LoRA trains a small low-rank adapter instead of updating the entire model.",
        },
      ],
      [
        { role: "system", content: "You are concise, exact, and helpful." },
        { role: "user", content: "Why does quantization help on Apple Silicon?" },
        {
          role: "assistant",
          content: "It lowers memory use and keeps the unified-memory runtime practical.",
        },
      ],
    ],
    promptMessages: [
      { role: "system", content: "You are concise, exact, and helpful." },
      { role: "user", content: "Write a friendly one-sentence greeting." },
    ],
    chosen: {
      role: "assistant",
      content: "You're welcome. Happy to help.",
    },
    rejected: {
      role: "assistant",
      content: "ok",
    },
  };
}

/** Parse one Ultrachat row into a supervision-ready chat transcript. */
export function parseUltrachatMessagesRow(row: unknown): readonly ChatMessage[] {
  const record = expectObject(row, "ultrachat row");
  const messages = parseChatMessages(record.messages, "ultrachat row.messages");
  expectAssistant(messages, "ultrachat row.messages");
  return messages;
}

/** Parse one Ultrafeedback preference row into prompt, chosen, and rejected turns. */
export function parseUltrafeedbackPreferenceRow(row: unknown): ParsedPreferenceConversation {
  const record = expectObject(row, "ultrafeedback row");
  const chosenMessages = parseChatMessages(record.chosen, "ultrafeedback row.chosen");
  const rejectedMessages = parseChatMessages(record.rejected, "ultrafeedback row.rejected");
  expectAssistant(chosenMessages, "ultrafeedback row.chosen");
  expectAssistant(rejectedMessages, "ultrafeedback row.rejected");

  const promptMessages = chosenMessages.slice(0, -1);
  const rejectedPrompt = rejectedMessages.slice(0, -1);
  if (!samePromptPrefix(promptMessages, rejectedPrompt)) {
    throw new Error("ultrafeedback row chosen and rejected prompts diverged.");
  }

  const chosen = chosenMessages.at(-1);
  const rejected = rejectedMessages.at(-1);
  if (chosen === undefined || rejected === undefined) {
    throw new Error("ultrafeedback row must include chosen and rejected replies.");
  }

  return {
    promptMessages,
    chosen,
    rejected,
  };
}

async function loadHuggingFaceRawDatasets(
  args: TrainingProofArgs,
): Promise<TrainingProofRawDatasets> {
  const trainCandidateLength = args.trainLimit * DATASET_OVERSAMPLE_FACTOR;
  const evalCandidateLength = args.evalLimit * DATASET_OVERSAMPLE_FACTOR;

  try {
    const supervisionTrainCandidates = await loadHuggingFaceRowsDataset({
      dataset: ULTRACHAT_DATASET,
      split: "train_sft",
      length: trainCandidateLength,
      parseRow: parseUltrachatMessagesRow,
    });
    const supervisionEvalCandidates = await loadHuggingFaceRowsDataset({
      dataset: ULTRACHAT_DATASET,
      split: "test_sft",
      length: evalCandidateLength,
      parseRow: parseUltrachatMessagesRow,
    });
    const preferenceTrainCandidates = await loadHuggingFaceRowsDataset({
      dataset: ULTRAFEEDBACK_DATASET,
      split: "train_prefs",
      length: trainCandidateLength,
      parseRow: parseUltrafeedbackPreferenceRow,
    });
    const preferenceEvalCandidates = await loadHuggingFaceRowsDataset({
      dataset: ULTRAFEEDBACK_DATASET,
      split: "test_prefs",
      length: evalCandidateLength,
      parseRow: parseUltrafeedbackPreferenceRow,
    });

    const samplePromptMessages =
      supervisionEvalCandidates.items()[0]?.slice(0, -1) ??
      supervisionTrainCandidates.items()[0]?.slice(0, -1) ??
      createTrainingProofCorpus().promptMessages;

    return {
      supervisionTrainMessages: supervisionTrainCandidates.items(),
      supervisionEvalMessages: supervisionEvalCandidates.items(),
      preferenceTrainRows: preferenceTrainCandidates.items(),
      preferenceEvalRows: preferenceEvalCandidates.items(),
      samplePromptMessages,
      notes: [
        `dataset_source=huggingface`,
        `supervision_dataset=${ULTRACHAT_DATASET}`,
        `preference_dataset=${ULTRAFEEDBACK_DATASET}`,
        `oversample_factor=${DATASET_OVERSAMPLE_FACTOR}`,
      ],
    };
  } catch {
    const supervisionTrainCandidates = await loadDatasetRowsFromParquet(
      ULTRACHAT_DATASET,
      "train_sft",
      "to_json(messages) AS messages",
      parseUltrachatMessagesRow,
      trainCandidateLength,
    );
    const supervisionEvalCandidates = await loadDatasetRowsFromParquet(
      ULTRACHAT_DATASET,
      "test_sft",
      "to_json(messages) AS messages",
      parseUltrachatMessagesRow,
      evalCandidateLength,
    );
    const preferenceTrainCandidates = await loadDatasetRowsFromParquet(
      ULTRAFEEDBACK_DATASET,
      "train_prefs",
      "to_json(chosen) AS chosen, to_json(rejected) AS rejected",
      parseUltrafeedbackPreferenceRow,
      trainCandidateLength,
    );
    const preferenceEvalCandidates = await loadDatasetRowsFromParquet(
      ULTRAFEEDBACK_DATASET,
      "test_prefs",
      "to_json(chosen) AS chosen, to_json(rejected) AS rejected",
      parseUltrafeedbackPreferenceRow,
      evalCandidateLength,
    );

    const samplePromptMessages =
      supervisionEvalCandidates[0]?.slice(0, -1) ??
      supervisionTrainCandidates[0]?.slice(0, -1) ??
      createTrainingProofCorpus().promptMessages;

    return {
      supervisionTrainMessages: supervisionTrainCandidates,
      supervisionEvalMessages: supervisionEvalCandidates,
      preferenceTrainRows: preferenceTrainCandidates,
      preferenceEvalRows: preferenceEvalCandidates,
      samplePromptMessages,
      notes: [
        `dataset_source=huggingface`,
        `dataset_transport=parquet_fallback`,
        `supervision_dataset=${ULTRACHAT_DATASET}`,
        `preference_dataset=${ULTRAFEEDBACK_DATASET}`,
        `oversample_factor=${DATASET_OVERSAMPLE_FACTOR}`,
      ],
    };
  }
}

/** Load the raw proof datasets before tokenization and length filtering. */
export async function loadTrainingProofRawDatasets(
  args: TrainingProofArgs,
): Promise<TrainingProofRawDatasets> {
  if (args.datasetSource === "tiny") {
    const corpus = createTrainingProofCorpus();
    return {
      supervisionTrainMessages: repeatToLength(corpus.supervisionExamples, args.trainLimit),
      supervisionEvalMessages: repeatToLength(corpus.supervisionExamples, args.evalLimit),
      preferenceTrainRows: repeatToLength(
        [
          {
            promptMessages: corpus.promptMessages,
            chosen: corpus.chosen,
            rejected: corpus.rejected,
          },
        ],
        args.trainLimit,
      ),
      preferenceEvalRows: repeatToLength(
        [
          {
            promptMessages: corpus.promptMessages,
            chosen: corpus.chosen,
            rejected: corpus.rejected,
          },
        ],
        args.evalLimit,
      ),
      samplePromptMessages: corpus.promptMessages,
      notes: ["dataset_source=tiny"],
    };
  }

  return await loadHuggingFaceRawDatasets(args);
}
