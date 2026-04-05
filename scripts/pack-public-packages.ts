/**
 * Dry-run pack every public workspace package.
 */

import { resolve } from "path";

const ROOT = resolve(import.meta.dirname, "..");
const PUBLIC_PACKAGES = [
  "core",
  "nn",
  "optimizers",
  "train",
  "data",
  "quantize",
  "lora",
  "align",
  "tokenizers",
  "transformers",
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
  for (const packageName of PUBLIC_PACKAGES) {
    const packageRoot = resolve(ROOT, "packages", packageName);
    console.log(`\nPacking ${packageName}...`);
    await run([process.execPath, "pm", "pack", "--dry-run"], packageRoot);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
