/**
 * Read-only local tools for CLI and app composition.
 * @module
 */

import { relative, resolve } from "path";

import type { AgentTool } from "./types";

export type ReadOnlyFileToolsOptions = {
  root: string;
  maxFiles?: number;
  maxBytes?: number;
};

function stringArg(args: Record<string, unknown>, key: string): string {
  const value = args[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`"${key}" must be a non-empty string.`);
  }
  return value;
}

function safePath(root: string, requested: string): string {
  const target = resolve(root, requested);
  const rel = relative(root, target);
  if (rel.startsWith("..") || rel === "") {
    throw new Error(`Path must stay inside ${root}.`);
  }
  return target;
}

function relativePath(root: string, target: string): string {
  return relative(root, target).replaceAll("\\", "/");
}

/** Create read-only file tools rooted inside one directory. */
export function createReadOnlyFileTools(options: ReadOnlyFileToolsOptions): AgentTool[] {
  const root = resolve(options.root);
  const maxFiles = options.maxFiles ?? 200;
  const maxBytes = options.maxBytes ?? 20_000;

  return [
    {
      name: "list_files",
      description: "List files under the allowed working directory.",
      parameters: {
        type: "object",
        properties: {
          pattern: {
            type: "string",
            description: "Optional Bun glob pattern, for example README.md or src/**/*.ts.",
          },
        },
      },
      async execute(args) {
        const pattern =
          typeof args.pattern === "string" && args.pattern.trim() !== "" ? args.pattern : "**/*";
        const files: string[] = [];
        for await (const file of new Bun.Glob(pattern).scan({
          cwd: root,
          absolute: true,
          onlyFiles: true,
        })) {
          const rel = relative(root, file);
          if (rel.startsWith("..") || rel === "") {
            continue;
          }
          files.push(relativePath(root, file));
          if (files.length >= maxFiles) {
            break;
          }
        }
        files.sort((left, right) => left.localeCompare(right));
        return files.length === 0 ? "(no files)" : files.join("\n");
      },
    },
    {
      name: "read_file",
      description: "Read a UTF-8 text file under the allowed working directory.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Relative path to read." },
        },
        required: ["path"],
      },
      async execute(args) {
        const target = safePath(root, stringArg(args, "path"));
        const file = Bun.file(target);
        if (!(await file.exists())) {
          return { content: `File not found: ${relativePath(root, target)}`, isError: true };
        }
        const text = await file.text();
        if (text.length <= maxBytes) {
          return text;
        }
        return `${text.slice(0, maxBytes)}\n...[truncated ${text.length - maxBytes} chars]`;
      },
    },
  ];
}
