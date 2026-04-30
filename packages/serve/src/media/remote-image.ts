/**
 * Remote image URL transport policy for local serving.
 * @module
 */

import { ServeError } from "../errors";

/** Default remote image request timeout. */
export const DEFAULT_REMOTE_IMAGE_TIMEOUT_MS = 10_000;

/** Default redirect cap for remote image requests. */
export const DEFAULT_REMOTE_IMAGE_MAX_REDIRECTS = 3;

export type RemoteAddress = {
  address: string;
  family: number;
};

export type RemoteImageFetcher = (input: string, init: RequestInit) => Promise<Response>;

export type RemoteImageResolver = (hostname: string) => Promise<readonly RemoteAddress[]>;

/** Safety controls for remote image URL loading. */
export type RemoteImageReadOptions = {
  signal?: AbortSignal;
  maxBytes: number;
  remoteImageHosts?: readonly string[];
  remoteTimeoutMs?: number;
  remoteMaxRedirects?: number;
  remoteFetch?: RemoteImageFetcher;
  remoteResolve?: RemoteImageResolver;
};

type FetchSignal = {
  signal: AbortSignal;
  timedOut(): boolean;
  cleanup: () => void;
};

function normalizeHost(hostname: string): string {
  const lower = hostname.toLowerCase().replace(/\.$/, "");
  return lower.startsWith("[") && lower.endsWith("]") ? lower.slice(1, -1) : lower;
}

function parseIpv4(address: string): readonly number[] | null {
  const parts = address.split(".");
  if (parts.length !== 4) {
    return null;
  }
  const octets = parts.map((part) => Number.parseInt(part, 10));
  if (
    octets.some(
      (octet, index) =>
        !Number.isInteger(octet) || octet < 0 || octet > 255 || String(octet) !== parts[index],
    )
  ) {
    return null;
  }
  return octets;
}

function isBlockedIpv4(address: string): boolean {
  const octets = parseIpv4(address);
  if (octets === null) {
    return false;
  }
  const [a = 0, b = 0, c = 0] = octets;
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 192 && b === 0 && c === 0) ||
    (a === 192 && b === 0 && c === 2) ||
    (a === 198 && (b === 18 || b === 19)) ||
    (a === 198 && b === 51 && c === 100) ||
    (a === 203 && b === 0 && c === 113) ||
    a >= 224
  );
}

function isBlockedIpv6(address: string): boolean {
  const value = normalizeHost(address);
  if (!value.includes(":")) {
    return false;
  }
  if (value === "::" || value === "::1" || value.startsWith("::ffff:")) {
    return true;
  }
  const first = Number.parseInt(value.split(":")[0] ?? "", 16);
  if (!Number.isInteger(first)) {
    return true;
  }
  const second = Number.parseInt(value.split(":")[1] ?? "", 16);
  return (
    (first & 0xfe00) === 0xfc00 ||
    (first & 0xffc0) === 0xfe80 ||
    (first & 0xff00) === 0xff00 ||
    (first === 0x2001 && second === 0x0db8)
  );
}

function isBlockedAddress(address: string): boolean {
  return isBlockedIpv4(address) || isBlockedIpv6(address);
}

function isIpLiteral(hostname: string): boolean {
  const host = normalizeHost(hostname);
  return parseIpv4(host) !== null || host.includes(":");
}

function allowedHost(hostname: string, allowedHosts: readonly string[] | undefined): boolean {
  if (allowedHosts === undefined || allowedHosts.length === 0) {
    return false;
  }
  const normalized = normalizeHost(hostname);
  return allowedHosts.some((host) => normalizeHost(host) === normalized);
}

function validateUrlShape(url: URL, context: string): string {
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new ServeError(`${context}: remote image URLs must use http or https.`, {
      code: "unsupported_input",
      param: "messages",
    });
  }
  if (url.username !== "" || url.password !== "") {
    throw new ServeError(`${context}: remote image URLs must not include credentials.`, {
      code: "unsupported_input",
      param: "messages",
    });
  }
  const expectedPort = url.protocol === "https:" ? "443" : "80";
  if (url.port !== "" && url.port !== expectedPort) {
    throw new ServeError(`${context}: remote image URLs must use the default protocol port.`, {
      code: "unsupported_input",
      param: "messages",
    });
  }
  const hostname = normalizeHost(url.hostname);
  if (hostname === "" || hostname === "localhost" || hostname.endsWith(".localhost")) {
    throw new ServeError(`${context}: remote image host is not allowed.`, {
      code: "unsupported_input",
      param: "messages",
    });
  }
  if (!isIpLiteral(hostname) && (!hostname.includes(".") || hostname.endsWith(".local"))) {
    throw new ServeError(`${context}: remote image host must be a public DNS name.`, {
      code: "unsupported_input",
      param: "messages",
    });
  }
  return hostname;
}

async function defaultRemoteResolve(hostname: string): Promise<readonly RemoteAddress[]> {
  const result = await Bun.dns.lookup(hostname);
  return Array.isArray(result) ? result : [result];
}

function remoteTimeoutError(context: string): ServeError {
  return new ServeError(`${context}: remote image request timed out.`, {
    code: "unsupported_input",
    param: "messages",
  });
}

function throwIfRemoteAborted(signal: AbortSignal | undefined, context: string): void {
  if (signal?.aborted === true) {
    throw new DOMException(`${context}: image read was cancelled.`, "AbortError");
  }
}

async function withRemoteTimeout<T>(
  work: () => Promise<T>,
  options: RemoteImageReadOptions,
  context: string,
): Promise<T> {
  throwIfRemoteAborted(options.signal, context);
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(remoteTimeoutError(context));
    }, options.remoteTimeoutMs ?? DEFAULT_REMOTE_IMAGE_TIMEOUT_MS);
    function cleanup(): void {
      clearTimeout(timeout);
      options.signal?.removeEventListener("abort", abort);
    }
    function abort(): void {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(new DOMException(`${context}: image read was cancelled.`, "AbortError"));
    }
    options.signal?.addEventListener("abort", abort, { once: true });
    let pending: Promise<T>;
    try {
      pending = work();
    } catch (error) {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error);
      return;
    }
    pending.then(
      (value) => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        resolve(value);
      },
      (error: unknown) => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        reject(error);
      },
    );
  });
}

async function validateResolvedHost(
  url: URL,
  options: RemoteImageReadOptions,
  context: string,
): Promise<void> {
  const hostname = validateUrlShape(url, context);
  if (!allowedHost(hostname, options.remoteImageHosts)) {
    throw new ServeError(
      `${context}: remote image host "${hostname}" is not in the configured allowlist.`,
      {
        code: "unsupported_input",
        param: "messages",
      },
    );
  }
  if (isIpLiteral(hostname)) {
    if (isBlockedAddress(hostname)) {
      throw new ServeError(`${context}: remote image host resolves to a local address.`, {
        code: "unsupported_input",
        param: "messages",
      });
    }
    return;
  }
  const resolve = options.remoteResolve ?? defaultRemoteResolve;
  let addresses: readonly RemoteAddress[];
  try {
    addresses = await withRemoteTimeout(() => resolve(hostname), options, context);
  } catch (error) {
    if (error instanceof ServeError || error instanceof DOMException) {
      throw error;
    }
    throw new ServeError(`${context}: remote image host could not be resolved.`, {
      code: "unsupported_input",
      param: "messages",
    });
  }
  if (addresses.length === 0 || addresses.some((address) => isBlockedAddress(address.address))) {
    throw new ServeError(`${context}: remote image host resolves to a local address.`, {
      code: "unsupported_input",
      param: "messages",
    });
  }
}

function createFetchSignal(options: RemoteImageReadOptions, context: string): FetchSignal {
  throwIfRemoteAborted(options.signal, context);
  const controller = new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort(remoteTimeoutError(context));
  }, options.remoteTimeoutMs ?? DEFAULT_REMOTE_IMAGE_TIMEOUT_MS);
  function abort(): void {
    controller.abort(new DOMException(`${context}: image read was cancelled.`, "AbortError"));
  }
  options.signal?.addEventListener("abort", abort, { once: true });
  return {
    signal: controller.signal,
    timedOut: () => timedOut,
    cleanup: () => {
      clearTimeout(timeout);
      options.signal?.removeEventListener("abort", abort);
    },
  };
}

function contentLength(response: Response): number | null {
  const raw = response.headers.get("content-length");
  if (raw === null) {
    return null;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function assertImageResponse(response: Response, options: RemoteImageReadOptions): void {
  const length = contentLength(response);
  if (length !== null && length > options.maxBytes) {
    throw new ServeError(
      `Remote image URL: image payload is ${length} bytes, exceeding the ${options.maxBytes} byte limit.`,
      { code: "unsupported_input", param: "messages" },
    );
  }
  const mediaType = response.headers.get("content-type")?.toLowerCase().split(";")[0]?.trim();
  if (mediaType === undefined || !mediaType.startsWith("image/")) {
    throw new ServeError("Remote image URL: response must use an image/* content type.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
}

async function readBoundedResponseBody(
  response: Response,
  options: RemoteImageReadOptions,
): Promise<Uint8Array> {
  if (response.body === null) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > options.maxBytes) {
      throw new ServeError(
        `Remote image URL: image payload is ${bytes.byteLength} bytes, exceeding the ${options.maxBytes} byte limit.`,
        { code: "unsupported_input", param: "messages" },
      );
    }
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  try {
    while (true) {
      if (options.signal?.aborted === true) {
        throw new DOMException("Remote image URL: image read was cancelled.", "AbortError");
      }
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      totalBytes += value.byteLength;
      if (totalBytes > options.maxBytes) {
        await reader.cancel();
        throw new ServeError(
          `Remote image URL: image payload exceeds the ${options.maxBytes} byte limit.`,
          { code: "unsupported_input", param: "messages" },
        );
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function redirectUrl(response: Response, currentUrl: URL): URL | null {
  if (response.status < 300 || response.status >= 400) {
    return null;
  }
  const location = response.headers.get("location");
  if (location === null) {
    return null;
  }
  try {
    return new URL(location, currentUrl);
  } catch {
    throw new ServeError("Remote image URL: redirect location must be a valid URL.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
}

async function fetchRemoteImageUrl(
  url: URL,
  options: RemoteImageReadOptions,
  redirects: number,
): Promise<Uint8Array> {
  await validateResolvedHost(url, options, "Remote image URL");
  const fetcher = options.remoteFetch ?? fetch;
  const fetchSignal = createFetchSignal(options, "Remote image URL");
  try {
    const response = await fetcher(url.toString(), {
      method: "GET",
      redirect: "manual",
      signal: fetchSignal.signal,
      headers: { accept: "image/*" },
    });
    const nextUrl = redirectUrl(response, url);
    if (nextUrl !== null) {
      const maxRedirects = options.remoteMaxRedirects ?? DEFAULT_REMOTE_IMAGE_MAX_REDIRECTS;
      if (redirects >= maxRedirects) {
        throw new ServeError("Remote image URL: redirect limit exceeded.", {
          code: "unsupported_input",
          param: "messages",
        });
      }
      fetchSignal.cleanup();
      return fetchRemoteImageUrl(nextUrl, options, redirects + 1);
    }
    if (!response.ok) {
      throw new ServeError(`Remote image URL: upstream returned HTTP ${response.status}.`, {
        code: "unsupported_input",
        param: "messages",
      });
    }
    assertImageResponse(response, options);
    return await readBoundedResponseBody(response, { ...options, signal: fetchSignal.signal });
  } catch (error) {
    if (fetchSignal.timedOut()) {
      throw remoteTimeoutError("Remote image URL");
    }
    if (error instanceof ServeError || error instanceof DOMException) {
      throw error;
    }
    throw new ServeError("Remote image URL: image request failed.", {
      code: "unsupported_input",
      param: "messages",
    });
  } finally {
    fetchSignal.cleanup();
  }
}

/** Fetch a remote image URL under the local serving transport policy. */
export async function readRemoteImageUrlBytes(
  value: string,
  options: RemoteImageReadOptions,
): Promise<Uint8Array> {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new ServeError("Remote image URL: expected a valid absolute URL.", {
      code: "unsupported_input",
      param: "messages",
    });
  }
  return fetchRemoteImageUrl(url, options, 0);
}
