type RankedMerge = {
  left: string;
  right: string;
  rank: number;
};

function pairKey(left: string, right: string): string {
  return `${left}\u0000${right}`;
}

export function createMergeKey(left: string, right: string): string {
  return pairKey(left, right);
}

export function findBestMerge(word: string[], merges: Map<string, number>): RankedMerge | null {
  let best: RankedMerge | null = null;
  for (let index = 0; index < word.length - 1; index += 1) {
    const left = word[index];
    const right = word[index + 1];
    if (left === undefined || right === undefined) {
      continue;
    }

    const rank = merges.get(pairKey(left, right));
    if (rank === undefined) {
      continue;
    }

    if (best === null || rank < best.rank) {
      best = { left, right, rank };
    }
  }
  return best;
}

export function mergeWordPieces(word: string[], merge: RankedMerge): string[] {
  const merged: string[] = [];
  let index = 0;
  while (index < word.length) {
    const left = word[index];
    const right = word[index + 1];
    if (left === undefined) {
      index += 1;
      continue;
    }

    if (left === merge.left && right === merge.right) {
      merged.push(`${left}${right}`);
      index += 2;
      continue;
    }

    merged.push(left);
    index += 1;
  }
  return merged;
}
