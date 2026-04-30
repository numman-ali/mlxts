/**
 * Builds workspace packages in dependency order so declaration emit can resolve
 * already-built sibling packages through their published package manifests.
 */

import { resolve } from "path";

const ROOT = resolve(import.meta.dirname, "..");
const WORKSPACE_ORDER = [
  { label: "core", path: ["packages", "core"] },
  { label: "tokenizers", path: ["packages", "tokenizers"] },
  { label: "data", path: ["packages", "data"] },
  { label: "nn", path: ["packages", "nn"] },
  { label: "diffusion", path: ["packages", "diffusion"] },
  { label: "quantize", path: ["packages", "quantize"] },
  { label: "optimizers", path: ["packages", "optimizers"] },
  { label: "train", path: ["packages", "train"] },
  { label: "transformers", path: ["packages", "transformers"] },
  { label: "lora", path: ["packages", "lora"] },
  { label: "align", path: ["packages", "align"] },
  { label: "protocols", path: ["packages", "protocols"] },
  { label: "serve", path: ["packages", "serve"] },
  { label: "agent", path: ["packages", "agent"] },
  { label: "nanogpt example", path: ["examples", "nanogpt"] },
] as const;

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
  for (const workspace of WORKSPACE_ORDER) {
    const workspaceRoot = resolve(ROOT, ...workspace.path);
    console.log(`\nBuilding ${workspace.label}...`);
    await run([process.execPath, "run", "build"], workspaceRoot);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
