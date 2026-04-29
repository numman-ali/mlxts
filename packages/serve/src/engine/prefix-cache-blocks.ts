/**
 * Token-block ownership for prompt-prefix cache metadata.
 * @module
 */

/** Token-block ownership counters for the prompt-prefix cache. */
export type PromptPrefixTokenBlockStats = {
  /** Configured token count per retained block. */
  blockSize: number;
  /** Unique retained token blocks. */
  blockCount: number;
  /** Sum of block reference counts across retained entries. */
  blockReferences: number;
  /** Tokens stored once across unique retained blocks. */
  uniqueTokenCount: number;
  /** Tokens represented across all entry references before block deduplication. */
  referencedTokenCount: number;
};

/** Token-block chain metadata attached to a prompt-prefix cache hit. */
export type PromptPrefixTokenBlockMetadata = {
  /** Configured token count per retained block. */
  blockSize: number;
  /** Number of blocks retained for the source prompt snapshot. */
  blockCount: number;
  /** Prefix length covered by complete blocks. */
  blockAlignedTokenLength: number;
};

/** Hash function for one token block in a parent-linked block chain. */
export type PromptPrefixTokenBlockHasher = (
  parentHash: string,
  tokenIds: readonly number[],
) => string;

/** Token-block store construction options. */
export type PromptPrefixTokenBlockStoreOptions = {
  /** Configured token count per retained block. */
  blockSize?: number;
  /** Custom hash seam used by collision and backend tests. */
  hashBlock?: PromptPrefixTokenBlockHasher;
};

type PromptPrefixTokenBlock = {
  hash: string;
  parentHash: string;
  tokenIds: readonly number[];
  refCount: number;
};

const ROOT_BLOCK_HASH = "root";
const FNV_OFFSET_BASIS = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

function mixByte(hash: bigint, byte: number): bigint {
  return ((hash ^ BigInt(byte & 0xff)) * FNV_PRIME) & FNV_MASK;
}

function mixString(hash: bigint, value: string): bigint {
  let mixed = hash;
  for (let index = 0; index < value.length; index += 1) {
    mixed = mixByte(mixed, value.charCodeAt(index));
  }
  return mixed;
}

function mixToken(hash: bigint, tokenId: number): bigint {
  let mixed = hash;
  const unsigned = tokenId >>> 0;
  mixed = mixByte(mixed, unsigned);
  mixed = mixByte(mixed, unsigned >>> 8);
  mixed = mixByte(mixed, unsigned >>> 16);
  mixed = mixByte(mixed, unsigned >>> 24);
  return mixed;
}

function tokenArraysEqual(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

/** Return the deterministic chain hash for a prompt-prefix token block. */
export function promptPrefixTokenBlockHash(
  parentHash: string,
  tokenIds: readonly number[],
): string {
  let hash = mixString(FNV_OFFSET_BASIS, parentHash);
  hash = mixByte(hash, 0xff);
  for (const tokenId of tokenIds) {
    hash = mixToken(hash, tokenId);
  }
  return hash.toString(16).padStart(16, "0");
}

/** Ref-counted token-block store used by prompt-prefix cache entries. */
export class PromptPrefixTokenBlockStore {
  readonly #blockSize: number;
  readonly #hashBlock: PromptPrefixTokenBlockHasher;
  readonly #blocks = new Map<string, PromptPrefixTokenBlock>();

  constructor(options: number | PromptPrefixTokenBlockStoreOptions = 64) {
    const blockSize = typeof options === "number" ? options : (options.blockSize ?? 64);
    if (!Number.isInteger(blockSize) || blockSize <= 0) {
      throw new Error(
        `PromptPrefixTokenBlockStore: blockSize must be a positive integer, got ${blockSize}.`,
      );
    }
    this.#blockSize = blockSize;
    this.#hashBlock =
      typeof options === "number"
        ? promptPrefixTokenBlockHash
        : (options.hashBlock ?? promptPrefixTokenBlockHash);
  }

  /** Configured token count per retained block. */
  get blockSize(): number {
    return this.#blockSize;
  }

  /** Retain a block-chain handle for the provided reusable prompt tokens. */
  retain(tokenIds: readonly number[]): PromptPrefixTokenBlockHandle {
    const hashes = this.hashesFor(tokenIds);
    const retainedHashes: string[] = [];

    try {
      for (let start = 0; start < tokenIds.length; start += this.#blockSize) {
        const blockTokenIds = tokenIds.slice(start, start + this.#blockSize);
        const hash = hashes[start / this.#blockSize];
        const parentHash =
          start === 0 ? ROOT_BLOCK_HASH : (hashes[start / this.#blockSize - 1] ?? ROOT_BLOCK_HASH);
        if (hash === undefined) {
          throw new Error("PromptPrefixTokenBlockStore: missing block hash.");
        }
        const existing = this.#blocks.get(hash);
        if (existing === undefined) {
          this.#blocks.set(hash, {
            hash,
            parentHash,
            tokenIds: blockTokenIds,
            refCount: 1,
          });
        } else {
          if (
            existing.parentHash !== parentHash ||
            !tokenArraysEqual(existing.tokenIds, blockTokenIds)
          ) {
            throw new Error(`PromptPrefixTokenBlockStore: hash collision for block ${hash}.`);
          }
          existing.refCount += 1;
        }
        retainedHashes.push(hash);
      }
    } catch (error) {
      this.release(retainedHashes);
      throw error;
    }

    return new PromptPrefixTokenBlockHandle(this, hashes, tokenIds.length, this.#blockSize);
  }

  /** Return the parent-linked block hashes for a prompt-token prefix. */
  hashesFor(tokenIds: readonly number[]): readonly string[] {
    const hashes: string[] = [];
    let parentHash = ROOT_BLOCK_HASH;
    for (let start = 0; start < tokenIds.length; start += this.#blockSize) {
      const blockTokenIds = tokenIds.slice(start, start + this.#blockSize);
      const hash = this.#hashBlock(parentHash, blockTokenIds);
      hashes.push(hash);
      parentHash = hash;
    }
    return hashes;
  }

  /** Return current token-block retention counters. */
  stats(): PromptPrefixTokenBlockStats {
    let blockReferences = 0;
    let uniqueTokenCount = 0;
    let referencedTokenCount = 0;
    for (const block of this.#blocks.values()) {
      blockReferences += block.refCount;
      uniqueTokenCount += block.tokenIds.length;
      referencedTokenCount += block.tokenIds.length * block.refCount;
    }
    return {
      blockSize: this.#blockSize,
      blockCount: this.#blocks.size,
      blockReferences,
      uniqueTokenCount,
      referencedTokenCount,
    };
  }

  /** Release a previously retained token-block chain. */
  release(hashes: readonly string[]): void {
    for (let index = hashes.length - 1; index >= 0; index -= 1) {
      const hash = hashes[index];
      if (hash === undefined) {
        continue;
      }
      const block = this.#blocks.get(hash);
      if (block === undefined) {
        throw new Error(`PromptPrefixTokenBlockStore: block ${hash} is not retained.`);
      }
      block.refCount -= 1;
      if (block.refCount === 0) {
        this.#blocks.delete(hash);
      }
    }
  }
}

/** Owned reference to a retained prompt-token block chain. */
export class PromptPrefixTokenBlockHandle implements Disposable {
  readonly #store: PromptPrefixTokenBlockStore;
  readonly #hashes: readonly string[];
  readonly #tokenLength: number;
  readonly #blockSize: number;
  #disposed = false;

  constructor(
    store: PromptPrefixTokenBlockStore,
    hashes: readonly string[],
    tokenLength: number,
    blockSize: number,
  ) {
    this.#store = store;
    this.#hashes = [...hashes];
    this.#tokenLength = tokenLength;
    this.#blockSize = blockSize;
  }

  /** Return metadata for the retained source block chain. */
  metadata(): PromptPrefixTokenBlockMetadata {
    return {
      blockSize: this.#blockSize,
      blockCount: this.#hashes.length,
      blockAlignedTokenLength: Math.floor(this.#tokenLength / this.#blockSize) * this.#blockSize,
    };
  }

  /** Return retained block hashes for internal prompt-cache indexing. */
  hashes(): readonly string[] {
    return [...this.#hashes];
  }

  [Symbol.dispose](): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.#store.release(this.#hashes);
  }
}
