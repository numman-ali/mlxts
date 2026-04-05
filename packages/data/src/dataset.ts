/** Shared dataset contracts for in-memory mlxts training recipes. */

export interface Dataset<T> {
  readonly length: number;
  at(index: number): T;
  items(): readonly T[];
}

/** Simple array-backed dataset. */
export class ArrayDataset<T> implements Dataset<T> {
  #items: readonly T[];

  constructor(items: readonly T[]) {
    this.#items = [...items];
  }

  get length(): number {
    return this.#items.length;
  }

  at(index: number): T {
    const item = this.#items[index];
    if (item === undefined) {
      throw new Error(`data.ArrayDataset: index ${index} is out of bounds.`);
    }
    return item;
  }

  items(): readonly T[] {
    return this.#items;
  }
}

/** Construct an array-backed dataset. */
export function datasetFromArray<T>(items: readonly T[]): ArrayDataset<T> {
  return new ArrayDataset(items);
}
