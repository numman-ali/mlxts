import { type ArrayDataset, datasetFromArray } from "./dataset";

/** Load JSONL records from disk into an array-backed dataset. */
export async function loadJsonlDataset<T>(
  path: string,
  parseRecord: (value: unknown, lineIndex: number) => T,
): Promise<ArrayDataset<T>> {
  const text = await Bun.file(path).text();
  const records: T[] = [];
  for (const [lineIndex, line] of text.split(/\r?\n/).entries()) {
    const trimmed = line.trim();
    if (trimmed === "") {
      continue;
    }
    records.push(parseRecord(JSON.parse(trimmed), lineIndex + 1));
  }
  return datasetFromArray(records);
}
