/**
 * GGUF load/save helpers backed by MLX's native GGUF support.
 * @module
 */

import { MxArray, readResultPointer } from "./array";
import { defaultStream } from "./device";
import { checkStatus } from "./error";
import { ffi, OutSlot, unwrapPointer } from "./ffi";

export type GgufMetadataValue = null | boolean | number | string | GgufMetadataValue[];

export type LoadedGguf = {
  tensors: Record<string, MxArray>;
  metadata: Record<string, GgufMetadataValue>;
};

const textEncoder = new TextEncoder();

function encodeCString(value: string): Uint8Array {
  return textEncoder.encode(`${value}\0`);
}

function readStringObject(handle: ReturnType<typeof readResultPointer>): string {
  try {
    const value = ffi.mlx_string_data(handle);
    if (value === null) {
      throw new Error("mlx_string_data returned a null pointer");
    }
    return value.toString();
  } finally {
    checkStatus(ffi.mlx_string_free(handle), "mlx_string_free");
  }
}

function readArrayMapKeys(handle: ReturnType<typeof readResultPointer>): string[] {
  const keysHandle = readResultPointer("gguf map keys", (out) => {
    checkStatus(ffi.mlxts_map_string_to_array_keys(out, handle), "mlxts_map_string_to_array_keys");
  });
  const joinedKeys = readStringObject(keysHandle);
  return joinedKeys === "" ? [] : joinedKeys.split("\n");
}

function collectArrayMap(handle: ReturnType<typeof readResultPointer>): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};

  try {
    for (const key of readArrayMapKeys(handle)) {
      const encodedKey = encodeCString(key);
      const tensorHandle = readResultPointer(`gguf tensor ${key}`, (out) => {
        checkStatus(
          ffi.mlx_map_string_to_array_get(out, handle, encodedKey),
          `mlx_map_string_to_array_get:${key}`,
        );
      });
      tensors[key] = MxArray._fromCtx(tensorHandle);
    }
    return tensors;
  } catch (error) {
    for (const tensor of Object.values(tensors)) {
      tensor.free();
    }
    throw error;
  } finally {
    checkStatus(ffi.mlx_map_string_to_array_free(handle), "mlx_map_string_to_array_free");
  }
}

/** Parse the JSON metadata payload produced by the native GGUF bridge. */
export function parseGgufMetadataJson(json: string): Record<string, GgufMetadataValue> {
  if (json === "") {
    return {};
  }

  const parsed = JSON.parse(json);
  if (!isMetadataRecord(parsed)) {
    throw new Error("loadGguf: native GGUF metadata payload must be a JSON object.");
  }
  return parsed;
}

function isMetadataValue(value: unknown): value is GgufMetadataValue {
  if (
    value === null ||
    typeof value === "boolean" ||
    typeof value === "number" ||
    typeof value === "string"
  ) {
    return true;
  }

  if (!Array.isArray(value)) {
    return false;
  }

  return value.every((entry) => isMetadataValue(entry));
}

function isMetadataRecord(value: unknown): value is Record<string, GgufMetadataValue> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }

  return Object.values(value).every((entry) => isMetadataValue(entry));
}

function createArrayMap(tensors: Record<string, MxArray>): ReturnType<typeof readResultPointer> {
  const handle = unwrapPointer(ffi.mlx_map_string_to_array_new(), "mlx_map_string_to_array_new");

  try {
    for (const [key, tensor] of Object.entries(tensors)) {
      const encodedKey = encodeCString(key);
      checkStatus(
        ffi.mlx_map_string_to_array_insert(handle, encodedKey, tensor._ctx),
        `mlx_map_string_to_array_insert:${key}`,
      );
    }
    return handle;
  } catch (error) {
    checkStatus(ffi.mlx_map_string_to_array_free(handle), "mlx_map_string_to_array_free");
    throw error;
  }
}

function createStringMap(values: Record<string, string>): ReturnType<typeof readResultPointer> {
  const handle = unwrapPointer(ffi.mlx_map_string_to_string_new(), "mlx_map_string_to_string_new");

  try {
    for (const [key, value] of Object.entries(values)) {
      const encodedKey = encodeCString(key);
      const encodedValue = encodeCString(value);
      checkStatus(
        ffi.mlx_map_string_to_string_insert(handle, encodedKey, encodedValue),
        `mlx_map_string_to_string_insert:${key}`,
      );
    }
    return handle;
  } catch (error) {
    checkStatus(ffi.mlx_map_string_to_string_free(handle), "mlx_map_string_to_string_free");
    throw error;
  }
}

/** Load a GGUF file into named MLX tensors and parsed metadata. */
export function loadGguf(path: string): LoadedGguf {
  const weightsSlot = new OutSlot();
  const metadataSlot = new OutSlot();
  const encodedPath = encodeCString(path);

  checkStatus(
    ffi.mlxts_load_gguf(
      weightsSlot.prepare(),
      metadataSlot.prepare(),
      encodedPath,
      defaultStream(),
    ),
    "mlxts_load_gguf",
  );

  const weightsHandle = weightsSlot.read("gguf weights");
  const metadataHandle = metadataSlot.read("gguf metadata");

  const metadataJson = readStringObject(metadataHandle);
  const tensors = collectArrayMap(weightsHandle);

  try {
    return { tensors, metadata: parseGgufMetadataJson(metadataJson) };
  } catch (error) {
    for (const tensor of Object.values(tensors)) {
      tensor.free();
    }
    throw error;
  }
}

/** Save named tensors to a GGUF file with optional string metadata. */
export function saveGguf(
  tensors: Record<string, MxArray>,
  path: string,
  metadata: Record<string, string> = {},
): void {
  const arrayMap = createArrayMap(tensors);
  const stringMap = createStringMap(metadata);
  const encodedPath = encodeCString(path);

  try {
    checkStatus(ffi.mlxts_save_gguf(encodedPath, arrayMap, stringMap), "mlxts_save_gguf");
  } finally {
    checkStatus(ffi.mlx_map_string_to_array_free(arrayMap), "mlx_map_string_to_array_free");
    checkStatus(ffi.mlx_map_string_to_string_free(stringMap), "mlx_map_string_to_string_free");
  }
}
