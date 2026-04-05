/**
 * Builds workspace packages in dependency order so declaration emit can resolve
 * already-built sibling packages through their published package manifests.
 */

import { resolve } from "path";

const ROOT = resolve(import.meta.dirname, "..");
const WORKSPACE_ORDER = [
  "core",
  "tokenizers",
  "data",
  "nn",
  "quantize",
  "optimizers",
  "train",
  "transformers",
  "lora",
  "align",
  "nanogpt",
];

async function run(command: string[], cwd: string): Promise<void> {
  const proc = Bun.spawn(command, {
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    throw new Error(`Command failed (${exitCode}): ${command.join(" ")}`);
  }
}

async function main(): Promise<void> {
  for (const packageName of WORKSPACE_ORDER) {
    const packageRoot = resolve(ROOT, "packages", packageName);
    console.log(`\nBuilding ${packageName}...`);
    await run([process.execPath, "run", "build"], packageRoot);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
