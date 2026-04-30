/**
 * Local file-id image transport policy for serving.
 * @module
 */

import { realpathSync, type Stats, statSync } from "fs";
import { extname, isAbsolute, relative, resolve } from "path";
import { ServeError } from "../errors";

const SUPPORTED_LOCAL_IMAGE_EXTENSIONS = new Set([
  ".bmp",
  ".gif",
  ".heic",
  ".heif",
  ".jpeg",
  ".jpg",
  ".png",
  ".tif",
  ".tiff",
  ".webp",
]);

export type LocalImageReadOptions = {
  signal?: AbortSignal;
  maxBytes: number;
  localImageRoots?: readonly string[];
};

type LocatedLocalImage = {
  path: string;
  stats: Stats;
};

function throwIfLocalImageAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw new DOMException("Local image file: image read was cancelled.", "AbortError");
  }
}

function requireLocalImageRoot(root: string): string {
  if (root.trim() === "") {
    throw new Error("localImageRoots must contain non-empty directory paths.");
  }
  const resolved = realpathSync(resolve(root));
  if (!statSync(resolved).isDirectory()) {
    throw new Error(`localImageRoots entry is not a directory: ${root}`);
  }
  return resolved;
}

/** Resolve local image roots to canonical directories before request handling. */
export function resolveLocalImageRoots(roots: readonly string[] | undefined): readonly string[] {
  return [...new Set((roots ?? []).map(requireLocalImageRoot))];
}

function localImageRootsForRead(roots: readonly string[] | undefined): readonly string[] {
  if (roots === undefined || roots.length === 0) {
    throw new ServeError("File-id image inputs require at least one configured local image root.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
  try {
    return resolveLocalImageRoots(roots);
  } catch {
    throw new ServeError("Local image roots must be existing directories.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
}

function localImageFileIdSegments(fileId: string): readonly string[] {
  if (fileId.trim() === "") {
    throw new ServeError("Local image file_id must be a non-empty relative path.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
  if (
    fileId.includes("\0") ||
    fileId.includes("\\") ||
    isAbsolute(fileId) ||
    fileId.startsWith("~") ||
    /^[A-Za-z][A-Za-z0-9+.-]*:/.test(fileId)
  ) {
    throw new ServeError("Local image file_id must be a relative path under a configured root.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
  const segments = fileId.split("/");
  if (segments.some((segment) => segment === "" || segment === "." || segment === "..")) {
    throw new ServeError("Local image file_id must not contain empty or traversal segments.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
  if (!SUPPORTED_LOCAL_IMAGE_EXTENSIONS.has(extname(fileId).toLowerCase())) {
    throw new ServeError("Local image file_id must point to a supported image extension.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
  return segments;
}

function assertPathInsideRoot(root: string, path: string): void {
  const pathFromRoot = relative(root, path);
  if (pathFromRoot === "" || pathFromRoot.startsWith("..") || isAbsolute(pathFromRoot)) {
    throw new ServeError("Local image file_id resolves outside the configured image root.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
}

function locateLocalImageFile(fileId: string, roots: readonly string[]): LocatedLocalImage {
  const segments = localImageFileIdSegments(fileId);
  for (const root of roots) {
    const candidate = resolve(root, ...segments);
    let realCandidate: string;
    try {
      realCandidate = realpathSync(candidate);
    } catch {
      continue;
    }
    assertPathInsideRoot(root, realCandidate);
    const stats = statSync(realCandidate);
    if (!stats.isFile()) {
      throw new ServeError("Local image file_id must resolve to a file.", {
        code: "unsupported_input",
        param: "messages",
      });
    }
    return { path: realCandidate, stats };
  }
  throw new ServeError(
    "Local image file_id did not match an image under the configured local image roots.",
    {
      code: "unsupported_input",
      param: "messages",
    },
  );
}

function enforceLocalImageByteLimit(bytes: number, maxBytes: number): void {
  if (bytes > maxBytes) {
    throw new ServeError(
      `Local image file: image payload is ${bytes} bytes, exceeding the ${maxBytes} byte limit.`,
      { code: "unsupported_input", param: "messages" },
    );
  }
}

/** Read an image file_id from explicitly configured local image roots. */
export async function readLocalImageFileBytes(
  fileId: string,
  options: LocalImageReadOptions,
): Promise<Uint8Array> {
  throwIfLocalImageAborted(options.signal);
  const roots = localImageRootsForRead(options.localImageRoots);
  const image = locateLocalImageFile(fileId, roots);
  enforceLocalImageByteLimit(image.stats.size, options.maxBytes);
  throwIfLocalImageAborted(options.signal);
  const bytes = new Uint8Array(await Bun.file(image.path).arrayBuffer());
  enforceLocalImageByteLimit(bytes.byteLength, options.maxBytes);
  throwIfLocalImageAborted(options.signal);
  return bytes;
}
