import { type ArrayDataset, datasetFromArray } from "./dataset";

const DEFAULT_BASE_URL = "https://datasets-server.huggingface.co";
const DEFAULT_PAGE_SIZE = 100;
const MAX_TRANSIENT_RETRIES = 3;

type DatasetRowsResponse = {
  rows?: Array<{
    row_idx?: number;
    row?: unknown;
  }>;
  error?: string;
};

type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

/** Options for loading a deterministic row slice from a Hugging Face dataset split. */
export type LoadHuggingFaceRowsDatasetOptions<T> = {
  dataset: string;
  split: string;
  parseRow: (row: unknown, rowIndex: number) => T;
  config?: string;
  offset?: number;
  length: number;
  pageSize?: number;
  baseUrl?: string;
  fetchImpl?: FetchLike;
};

function isRowsResponse(value: unknown): value is DatasetRowsResponse {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

type RowsRequestOptions = Pick<
  LoadHuggingFaceRowsDatasetOptions<unknown>,
  "dataset" | "split" | "config" | "baseUrl" | "fetchImpl"
>;

type RetryPayload = {
  retryable?: boolean;
  message?: string;
};

function isRetryPayload(value: unknown): value is RetryPayload {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readPositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`data.loadHuggingFaceRowsDataset: ${name} must be a positive integer.`);
  }
  return value;
}

function readNonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`data.loadHuggingFaceRowsDataset: ${name} must be a non-negative integer.`);
  }
  return value;
}

function readRowsResponse(
  payload: unknown,
  dataset: string,
  split: string,
): Array<{ row_idx?: number; row?: unknown }> {
  if (!isRowsResponse(payload)) {
    throw new Error(
      `data.loadHuggingFaceRowsDataset: expected an object response for ${dataset}:${split}.`,
    );
  }

  if (typeof payload.error === "string" && payload.error.trim() !== "") {
    throw new Error(
      `data.loadHuggingFaceRowsDataset: ${dataset}:${split} returned "${payload.error}".`,
    );
  }

  if (!Array.isArray(payload.rows)) {
    throw new Error(
      `data.loadHuggingFaceRowsDataset: response for ${dataset}:${split} did not include rows.`,
    );
  }

  return payload.rows;
}

function rowsUrl(options: RowsRequestOptions, offset: number, length: number): string {
  const params = new URLSearchParams({
    dataset: options.dataset,
    config: options.config ?? "default",
    split: options.split,
    offset: String(offset),
    length: String(length),
  });
  return `${options.baseUrl ?? DEFAULT_BASE_URL}/rows?${params.toString()}`;
}

function shouldRetry(status: number): boolean {
  return status === 429 || status >= 500;
}

async function requestRowsOnce(
  options: RowsRequestOptions,
  url: string,
): Promise<Array<{ row_idx?: number; row?: unknown }>> {
  const response = await (options.fetchImpl ?? fetch)(url);
  if (!response.ok) {
    throw new Error(
      JSON.stringify({
        retryable: shouldRetry(response.status),
        message: `data.loadHuggingFaceRowsDataset: ${options.dataset}:${options.split} request failed with ${response.status} ${response.statusText}.`,
      }),
    );
  }
  return readRowsResponse(await response.json(), options.dataset, options.split);
}

function parseRetryPayload(error: unknown): RetryPayload | null {
  if (!(error instanceof Error)) {
    return null;
  }
  try {
    const payload: unknown = JSON.parse(error.message);
    return isRetryPayload(payload) ? payload : null;
  } catch {
    return null;
  }
}

function retryError(error: unknown, options: RowsRequestOptions): Error {
  if (error instanceof Error) {
    const payload = parseRetryPayload(error);
    if (typeof payload?.message === "string") {
      return new Error(payload.message);
    }
    return error;
  }
  return new Error(
    `data.loadHuggingFaceRowsDataset: ${options.dataset}:${options.split} request failed.`,
  );
}

function isRetryableError(error: unknown): boolean {
  return parseRetryPayload(error)?.retryable ?? true;
}

async function fetchRowsPage(
  options: RowsRequestOptions,
  offset: number,
  length: number,
): Promise<Array<{ row_idx?: number; row?: unknown }>> {
  const url = rowsUrl(options, offset, length);

  for (let attempt = 0; attempt < MAX_TRANSIENT_RETRIES; attempt += 1) {
    try {
      return await requestRowsOnce(options, url);
    } catch (error) {
      if (attempt < MAX_TRANSIENT_RETRIES - 1 && isRetryableError(error)) {
        await Bun.sleep(250 * (attempt + 1));
        continue;
      }
      throw retryError(error, options);
    }
  }

  throw new Error(
    `data.loadHuggingFaceRowsDataset: ${options.dataset}:${options.split} request failed.`,
  );
}

/** Load a deterministic row slice from a Hugging Face dataset split through the datasets server. */
export async function loadHuggingFaceRowsDataset<T>(
  options: LoadHuggingFaceRowsDatasetOptions<T>,
): Promise<ArrayDataset<T>> {
  const targetLength = readPositiveInteger(options.length, "length");
  const startOffset = readNonNegativeInteger(options.offset ?? 0, "offset");
  const pageSize = Math.min(
    DEFAULT_PAGE_SIZE,
    readPositiveInteger(options.pageSize ?? DEFAULT_PAGE_SIZE, "pageSize"),
  );

  const records: T[] = [];
  let fetched = 0;
  const requestOptions: RowsRequestOptions = {
    dataset: options.dataset,
    split: options.split,
    ...(options.config === undefined ? {} : { config: options.config }),
    ...(options.baseUrl === undefined ? {} : { baseUrl: options.baseUrl }),
    ...(options.fetchImpl === undefined ? {} : { fetchImpl: options.fetchImpl }),
  };
  while (records.length < targetLength) {
    const remaining = targetLength - records.length;
    const page = await fetchRowsPage(
      requestOptions,
      startOffset + fetched,
      Math.min(pageSize, remaining),
    );
    if (page.length === 0) {
      break;
    }

    for (const entry of page) {
      records.push(options.parseRow(entry.row, entry.row_idx ?? startOffset + fetched));
    }

    fetched += page.length;
    if (page.length < Math.min(pageSize, remaining)) {
      break;
    }
  }

  if (records.length < targetLength) {
    throw new Error(
      `data.loadHuggingFaceRowsDataset: ${options.dataset}:${options.split} returned only ${records.length} row(s); expected ${targetLength}.`,
    );
  }

  return datasetFromArray(records);
}
