/**
 * Shared package build script for dist JS and declaration output.
 *
 * Usage from a package directory:
 *   bun run ../../scripts/build-package.ts --entry src/index.ts
 *   bun run ../../scripts/build-package.ts --entry src/index.ts --entry src/cli.ts
 */

import { chmodSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs";
import { relative, resolve } from "path";

const packageRoot = process.cwd();
const distDir = resolve(packageRoot, "dist");

type BuildOptions = {
  entries: string[];
};

function parseArgs(argv: string[]): BuildOptions {
  const entries: string[] = [];

  for (let index = 2; index < argv.length; index++) {
    const argument = argv[index];
    if (argument !== "--entry") {
      throw new Error(`Unknown argument: ${argument ?? "<missing>"}`);
    }

    const entry = argv[index + 1];
    if (entry === undefined) {
      throw new Error("Missing value for --entry");
    }

    entries.push(entry);
    index += 1;
  }

  if (entries.length === 0) {
    throw new Error("At least one --entry path is required");
  }

  return { entries };
}

async function run(command: string[], cwd = packageRoot): Promise<void> {
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

function outputPathFor(entry: string): string {
  const relativeEntry = relative(resolve(packageRoot, "src"), resolve(packageRoot, entry))
    .replaceAll("\\", "/")
    .replace(/\.ts$/, ".js");
  return resolve(distDir, relativeEntry);
}

function sourceHasShebang(entry: string): boolean {
  return readFileSync(resolve(packageRoot, entry), "utf8").startsWith("#!");
}

function ensureShebang(entry: string): void {
  if (!sourceHasShebang(entry)) {
    return;
  }

  const outputPath = outputPathFor(entry);
  const current = readFileSync(outputPath, "utf8");
  if (!current.startsWith("#!/usr/bin/env bun\n")) {
    writeFileSync(outputPath, `#!/usr/bin/env bun\n${current}`);
  }
  chmodSync(outputPath, 0o755);
}

async function buildJavaScript(entries: readonly string[]): Promise<void> {
  const args = [
    process.execPath,
    "build",
    "--target=bun",
    "--format=esm",
    "--packages=external",
    "--root=src",
    "--outdir=dist",
    "--entry-naming=[dir]/[name].js",
    "--sourcemap=external",
    "--external=@mlxts/*",
    ...entries,
  ];
  await run(args);
}

async function buildDeclarations(): Promise<void> {
  await run(["bunx", "tsc", "--project", "tsconfig.build.json"]);
}

async function main(): Promise<void> {
  const { entries } = parseArgs(process.argv);

  rmSync(distDir, { force: true, recursive: true });
  mkdirSync(distDir, { recursive: true });

  await buildJavaScript(entries);
  await buildDeclarations();

  for (const entry of entries) {
    ensureShebang(entry);
  }

  const packageName = packageRoot.split("/").at(-1) ?? packageRoot;
  console.log(`Built ${packageName} -> ${relative(packageRoot, distDir)}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
