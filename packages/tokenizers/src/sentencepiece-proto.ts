/**
 * Minimal SentencePiece model protobuf reader for inference-time loading.
 * @module
 */

import { UnsupportedTokenizerError } from "./errors";

export type SentencePieceEntry = {
  piece: string;
  score: number;
  type: number;
};

export type SentencePieceModel = {
  pieces: SentencePieceEntry[];
  modelType: number | undefined;
  byteFallback: boolean;
  unkId: number | undefined;
  bosId: number | undefined;
  eosId: number | undefined;
  padId: number | undefined;
  addDummyPrefix: boolean;
  removeExtraWhitespaces: boolean;
  escapeWhitespaces: boolean;
};

type ReadCursor = {
  offset: number;
};

function readVarint(bytes: Uint8Array, cursor: ReadCursor): number {
  let shift = 0;
  let result = 0;

  while (cursor.offset < bytes.length) {
    const byte = bytes[cursor.offset];
    if (byte === undefined) {
      break;
    }

    cursor.offset += 1;
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return result;
    }
    shift += 7;
  }

  throw new UnsupportedTokenizerError("SentencePiece model is truncated");
}

function readBytes(bytes: Uint8Array, cursor: ReadCursor): Uint8Array {
  const length = readVarint(bytes, cursor);
  const start = cursor.offset;
  const end = start + length;
  if (end > bytes.length) {
    throw new UnsupportedTokenizerError("SentencePiece length-delimited field exceeds file size");
  }
  cursor.offset = end;
  return bytes.subarray(start, end);
}

function skipField(bytes: Uint8Array, cursor: ReadCursor, wireType: number): void {
  switch (wireType) {
    case 0:
      readVarint(bytes, cursor);
      return;
    case 1:
      cursor.offset += 8;
      return;
    case 2:
      readBytes(bytes, cursor);
      return;
    case 5:
      cursor.offset += 4;
      return;
    default:
      throw new UnsupportedTokenizerError(`Unsupported protobuf wire type ${wireType}`);
  }
}

function parseSentencePiece(message: Uint8Array): SentencePieceEntry {
  const cursor: ReadCursor = { offset: 0 };
  let piece = "";
  let score = 0;
  let type = 1;

  while (cursor.offset < message.length) {
    const tag = readVarint(message, cursor);
    const fieldNumber = tag >>> 3;
    const wireType = tag & 0x7;

    if (fieldNumber === 1 && wireType === 2) {
      piece = new TextDecoder().decode(readBytes(message, cursor));
      continue;
    }
    if (fieldNumber === 2 && wireType === 5) {
      const view = new DataView(message.buffer, message.byteOffset + cursor.offset, 4);
      score = view.getFloat32(0, true);
      cursor.offset += 4;
      continue;
    }
    if (fieldNumber === 3 && wireType === 0) {
      type = readVarint(message, cursor);
      continue;
    }

    skipField(message, cursor, wireType);
  }

  return { piece, score, type };
}

function applyTrainerSpecField(
  model: SentencePieceModel,
  fieldNumber: number,
  wireType: number,
  message: Uint8Array,
  cursor: ReadCursor,
): boolean {
  if (wireType !== 0) {
    return false;
  }

  switch (fieldNumber) {
    case 3:
      model.modelType = readVarint(message, cursor);
      return true;
    case 35:
      model.byteFallback = readVarint(message, cursor) !== 0;
      return true;
    case 40:
      model.unkId = readVarint(message, cursor);
      return true;
    case 41:
      model.bosId = readVarint(message, cursor);
      return true;
    case 42:
      model.eosId = readVarint(message, cursor);
      return true;
    case 43:
      model.padId = readVarint(message, cursor);
      return true;
    default:
      return false;
  }
}

function parseTrainerSpec(message: Uint8Array, model: SentencePieceModel): void {
  const cursor: ReadCursor = { offset: 0 };
  while (cursor.offset < message.length) {
    const tag = readVarint(message, cursor);
    const fieldNumber = tag >>> 3;
    const wireType = tag & 0x7;

    if (applyTrainerSpecField(model, fieldNumber, wireType, message, cursor)) {
      continue;
    }

    skipField(message, cursor, wireType);
  }
}

function parseNormalizerSpec(message: Uint8Array, model: SentencePieceModel): void {
  const cursor: ReadCursor = { offset: 0 };
  while (cursor.offset < message.length) {
    const tag = readVarint(message, cursor);
    const fieldNumber = tag >>> 3;
    const wireType = tag & 0x7;

    if (fieldNumber === 3 && wireType === 0) {
      model.addDummyPrefix = readVarint(message, cursor) !== 0;
      continue;
    }
    if (fieldNumber === 4 && wireType === 0) {
      model.removeExtraWhitespaces = readVarint(message, cursor) !== 0;
      continue;
    }
    if (fieldNumber === 5 && wireType === 0) {
      model.escapeWhitespaces = readVarint(message, cursor) !== 0;
      continue;
    }

    skipField(message, cursor, wireType);
  }
}

/** Parse the small SentencePiece proto subset needed for inference-time tokenization. */
export function parseSentencePieceModel(bytes: Uint8Array): SentencePieceModel {
  const model: SentencePieceModel = {
    pieces: [],
    modelType: undefined,
    byteFallback: false,
    unkId: undefined,
    bosId: undefined,
    eosId: undefined,
    padId: undefined,
    addDummyPrefix: true,
    removeExtraWhitespaces: true,
    escapeWhitespaces: true,
  };

  const cursor: ReadCursor = { offset: 0 };
  while (cursor.offset < bytes.length) {
    const tag = readVarint(bytes, cursor);
    const fieldNumber = tag >>> 3;
    const wireType = tag & 0x7;

    if (fieldNumber === 1 && wireType === 2) {
      model.pieces.push(parseSentencePiece(readBytes(bytes, cursor)));
      continue;
    }
    if (fieldNumber === 2 && wireType === 2) {
      parseTrainerSpec(readBytes(bytes, cursor), model);
      continue;
    }
    if (fieldNumber === 3 && wireType === 2) {
      parseNormalizerSpec(readBytes(bytes, cursor), model);
      continue;
    }

    skipField(bytes, cursor, wireType);
  }

  if (model.pieces.length === 0) {
    throw new UnsupportedTokenizerError("SentencePiece model does not contain any pieces");
  }

  return model;
}
