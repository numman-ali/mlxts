import { describe, expect, test } from "bun:test";

import { readImageSourceBytes } from "./image";
import type { RemoteAddress, RemoteImageFetcher, RemoteImageResolver } from "./remote-image";

function publicResolver(hosts: string[] = []): RemoteImageResolver {
  return async (hostname: string): Promise<readonly RemoteAddress[]> => {
    hosts.push(hostname);
    return [{ address: "93.184.216.34", family: 4 }];
  };
}

function imageResponse(body: string, headers: HeadersInit = {}): Response {
  return new Response(body, {
    headers: {
      "content-type": "image/png",
      "content-length": String(body.length),
      ...headers,
    },
  });
}

describe("remote image URL loading", () => {
  test("fetches public HTTP(S) image URLs under the remote policy", async () => {
    const calls: string[] = [];
    const fetcher: RemoteImageFetcher = async (input, init) => {
      calls.push(input);
      expect(init.method).toBe("GET");
      expect(init.redirect).toBe("manual");
      expect(new Headers(init.headers).get("accept")).toBe("image/*");
      return imageResponse("abc");
    };

    const bytes = await readImageSourceBytes(
      { kind: "url", url: "https://example.com/image.png" },
      { remoteImageHosts: ["example.com"], remoteFetch: fetcher, remoteResolve: publicResolver() },
    );

    expect(calls).toEqual(["https://example.com/image.png"]);
    expect(Array.from(bytes)).toEqual([97, 98, 99]);

    const ipBytes = await readImageSourceBytes(
      { kind: "url", url: "https://93.184.216.34/image.png" },
      { remoteImageHosts: ["93.184.216.34"], remoteFetch: fetcher },
    );
    expect(Array.from(ipBytes)).toEqual([97, 98, 99]);
  });

  test("revalidates redirects before fetching the redirected image", async () => {
    const hosts: string[] = [];
    const calls: string[] = [];
    const fetcher: RemoteImageFetcher = async (input) => {
      calls.push(input);
      if (input === "https://example.com/image.png") {
        return new Response(null, {
          status: 302,
          headers: { location: "https://cdn.example.com/next.png" },
        });
      }
      return imageResponse("de");
    };

    const bytes = await readImageSourceBytes(
      { kind: "url", url: "https://example.com/image.png" },
      {
        remoteImageHosts: ["example.com", "cdn.example.com"],
        remoteFetch: fetcher,
        remoteResolve: publicResolver(hosts),
        remoteMaxRedirects: 1,
      },
    );

    expect(hosts).toEqual(["example.com", "cdn.example.com"]);
    expect(calls).toEqual(["https://example.com/image.png", "https://cdn.example.com/next.png"]);
    expect(Array.from(bytes)).toEqual([100, 101]);
  });

  test("rejects local hosts and private address resolutions before fetching", async () => {
    let fetchCalls = 0;
    const fetcher: RemoteImageFetcher = async () => {
      fetchCalls += 1;
      return imageResponse("a");
    };
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "http://127.0.0.1/image.png" },
        { remoteImageHosts: ["127.0.0.1"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("local address");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteFetch: fetcher,
          remoteResolve: async () => [{ address: "10.0.0.2", family: 4 }],
        },
      ),
    ).rejects.toThrow("local address");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://[2001:db8::1]/image.png" },
        { remoteImageHosts: ["2001:db8::1"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("local address");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://[::1]/image.png" },
        { remoteImageHosts: ["::1"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("local address");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteFetch: fetcher,
          remoteResolve: async () => [{ address: ":", family: 6 }],
        },
      ),
    ).rejects.toThrow("local address");
    expect(fetchCalls).toBe(0);
  });

  test("rejects unsupported URL shapes", async () => {
    const fetcher: RemoteImageFetcher = async () => imageResponse("a");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "not-a-url" },
        { remoteImageHosts: ["example.com"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("valid absolute URL");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "ftp://example.com/image.png" },
        { remoteFetch: fetcher },
      ),
    ).rejects.toThrow("http or https");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://user@example.com/image.png" },
        { remoteImageHosts: ["example.com"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("credentials");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com:444/image.png" },
        { remoteImageHosts: ["example.com"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("default protocol port");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://localhost/image.png" },
        { remoteFetch: fetcher },
      ),
    ).rejects.toThrow("not allowed");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example/image.png" },
        { remoteImageHosts: ["example"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("public DNS name");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.local/image.png" },
        { remoteImageHosts: ["example.local"], remoteFetch: fetcher },
      ),
    ).rejects.toThrow("public DNS name");
  });

  test("rejects remote image hosts outside the operator allowlist", async () => {
    let fetchCalls = 0;
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteFetch: async () => {
            fetchCalls += 1;
            return imageResponse("a");
          },
          remoteResolve: publicResolver(),
        },
      ),
    ).rejects.toThrow("not in the configured allowlist");
    expect(fetchCalls).toBe(0);
  });

  test("rejects non-image or oversized remote responses", async () => {
    const empty = await readImageSourceBytes(
      { kind: "url", url: "https://example.com/empty.png" },
      {
        remoteImageHosts: ["example.com"],
        remoteResolve: publicResolver(),
        remoteFetch: async () => new Response(null, { headers: { "content-type": "image/png" } }),
      },
    );
    expect(empty.byteLength).toBe(0);
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/readme.txt" },
        {
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () =>
            new Response("hello", { headers: { "content-type": "text/plain" } }),
        },
      ),
    ).rejects.toThrow("image/* content type");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/missing.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () =>
            new Response("missing", { status: 404, headers: { "content-type": "image/png" } }),
        },
      ),
    ).rejects.toThrow("HTTP 404");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/large.png" },
        {
          maxBytes: 2,
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () => imageResponse("abc", { "content-length": "3" }),
        },
      ),
    ).rejects.toThrow("exceeding the 2 byte limit");
  });

  test("enforces redirect and streamed-byte limits", async () => {
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteMaxRedirects: 0,
          remoteResolve: publicResolver(),
          remoteFetch: async () =>
            new Response(null, {
              status: 302,
              headers: { location: "https://cdn.example.com/a.png" },
            }),
        },
      ),
    ).rejects.toThrow("redirect limit exceeded");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          maxBytes: 2,
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () => imageResponse("abc", { "content-length": "" }),
        },
      ),
    ).rejects.toThrow("exceeds the 2 byte limit");
  });

  test("maps resolver, fetch, and malformed redirect failures to remote image errors", async () => {
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteTimeoutMs: 1,
          remoteResolve: () => new Promise(() => {}),
          remoteFetch: async () => imageResponse("a"),
        },
      ),
    ).rejects.toThrow("timed out");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteResolve: async () => {
            throw new Error("resolver exploded");
          },
          remoteFetch: async () => imageResponse("a"),
        },
      ),
    ).rejects.toThrow("could not be resolved");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteResolve: () => {
            throw new Error("resolver exploded");
          },
          remoteFetch: async () => imageResponse("a"),
        },
      ),
    ).rejects.toThrow("could not be resolved");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () => {
            throw new Error("fetch exploded");
          },
        },
      ),
    ).rejects.toThrow("image request failed");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () =>
            new Response(null, { status: 302, headers: { "content-type": "image/png" } }),
        },
      ),
    ).rejects.toThrow("HTTP 302");
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () =>
            new Response(null, { status: 302, headers: { location: "http://%" } }),
        },
      ),
    ).rejects.toThrow("redirect location");
  });

  test("maps cancellation and transport timeouts to remote image errors", async () => {
    const preAborted = new AbortController();
    preAborted.abort();
    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          signal: preAborted.signal,
          remoteImageHosts: ["example.com"],
          remoteResolve: publicResolver(),
          remoteFetch: async () => imageResponse("a"),
        },
      ),
    ).rejects.toThrow("cancelled");

    const duringResolve = new AbortController();
    const pendingResolve = readImageSourceBytes(
      { kind: "url", url: "https://example.com/image.png" },
      {
        signal: duringResolve.signal,
        remoteImageHosts: ["example.com"],
        remoteResolve: () => new Promise<readonly RemoteAddress[]>(() => {}),
        remoteFetch: async () => imageResponse("a"),
      },
    );
    duringResolve.abort();
    await expect(pendingResolve).rejects.toThrow("cancelled");

    const duringFetch = new AbortController();
    const pendingFetch = readImageSourceBytes(
      { kind: "url", url: "https://example.com/image.png" },
      {
        signal: duringFetch.signal,
        remoteImageHosts: ["example.com"],
        remoteResolve: publicResolver(),
        remoteFetch: (_input, init) =>
          new Promise<Response>((_resolve, reject) => {
            init.signal?.addEventListener(
              "abort",
              () => reject(init.signal?.reason ?? new DOMException("aborted", "AbortError")),
              { once: true },
            );
          }),
      },
    );
    duringFetch.abort();
    await expect(pendingFetch).rejects.toThrow("cancelled");

    await expect(
      readImageSourceBytes(
        { kind: "url", url: "https://example.com/image.png" },
        {
          remoteImageHosts: ["example.com"],
          remoteTimeoutMs: 1,
          remoteResolve: publicResolver(),
          remoteFetch: (_input, init) =>
            new Promise<Response>((_resolve, reject) => {
              init.signal?.addEventListener(
                "abort",
                () => reject(init.signal?.reason ?? new DOMException("aborted", "AbortError")),
                { once: true },
              );
            }),
        },
      ),
    ).rejects.toThrow("timed out");
  });
});
