function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function patternToRegExp(pattern: string): RegExp {
  const parts = Array.from(pattern).map((char) => {
    if (char === "*") {
      return ".*";
    }
    if (char === "?") {
      return ".";
    }
    return escapeRegExp(char);
  });
  return new RegExp(`^${parts.join("")}$`);
}

function matchesPattern(path: string, pattern: string): boolean {
  return patternToRegExp(pattern).test(path);
}

/** Decide whether a snapshot file should be included after file and glob filters. */
export function shouldIncludePath(
  path: string,
  files: string[] | undefined,
  include: string[],
  exclude: string[],
): boolean {
  if (files !== undefined && files.length > 0) {
    return files.includes(path);
  }

  const included = include.length === 0 || include.some((pattern) => matchesPattern(path, pattern));
  if (!included) {
    return false;
  }

  return !exclude.some((pattern) => matchesPattern(path, pattern));
}
