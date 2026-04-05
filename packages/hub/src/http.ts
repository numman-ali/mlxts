import type { HubRepoType } from "./types";

type FetchMethod = "GET" | "HEAD";

function apiPrefix(repoType: HubRepoType): string {
  return repoType === "dataset" ? "datasets" : "models";
}

function repoPrefix(repoType: HubRepoType): string {
  return repoType === "dataset" ? "datasets/" : "";
}

function buildHeaders(token: string | undefined): HeadersInit {
  if (token === undefined || token === "") {
    return {};
  }
  return { Authorization: `Bearer ${token}` };
}

/** Fetch a Hugging Face revision payload as a validated JSON object. */
export async function fetchJson(
  repoId: string,
  repoType: HubRepoType,
  revision: string,
  token: string | undefined,
): Promise<Record<string, unknown>> {
  const url = `https://huggingface.co/api/${apiPrefix(repoType)}/${repoId}/revision/${encodeURIComponent(revision)}`;
  const response = await fetch(url, { headers: buildHeaders(token) });
  if (!response.ok) {
    throw new Error(
      `resolveSnapshot: HuggingFace API request failed (${response.status} ${response.statusText})`,
    );
  }
  const payload: unknown = await response.json();
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new Error("resolveSnapshot: HuggingFace API returned a non-object payload");
  }
  return Object.fromEntries(Object.entries(payload));
}

/** Resolve a repo-relative file path to its Hugging Face download URL. */
export function resolveFileUrl(
  repoId: string,
  repoType: HubRepoType,
  revision: string,
  relativePath: string,
): string {
  return `https://huggingface.co/${repoPrefix(repoType)}${repoId}/resolve/${encodeURIComponent(revision)}/${relativePath}`;
}

/** Fetch a snapshot file or metadata URL from Hugging Face. */
export async function fetchFile(
  url: string,
  token: string | undefined,
  method: FetchMethod = "GET",
): Promise<Response> {
  const response = await fetch(url, {
    method,
    headers: buildHeaders(token),
    redirect: "follow",
  });
  if (!response.ok) {
    throw new Error(
      `downloadSnapshot: ${method} ${url} failed (${response.status} ${response.statusText})`,
    );
  }
  return response;
}
