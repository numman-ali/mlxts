/**
 * Build script for mlx-ts native dependencies.
 *
 * Runs CMake to build mlx-c (which auto-fetches MLX via FetchContent),
 * then copies the resulting dylibs to native/lib/ and fixes rpaths
 * so they can find each other at runtime.
 */
import { existsSync, mkdirSync, readdirSync } from "fs";
import { resolve } from "path";

const ROOT = resolve(import.meta.dirname, "..");
const NATIVE_DIR = resolve(ROOT, "native");
const BUILD_DIR = resolve(NATIVE_DIR, "build");
const LIB_DIR = resolve(NATIVE_DIR, "lib");

/** Run a shell command, streaming output. Throws on non-zero exit. */
async function run(command: string[], options: { cwd: string }): Promise<void> {
  const label = command.join(" ");
  console.log(`\n> ${label}`);
  console.log(`  cwd: ${options.cwd}\n`);

  const proc = Bun.spawn(command, {
    cwd: options.cwd,
    stdout: "inherit",
    stderr: "inherit",
  });

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    throw new Error(`Command failed (exit ${exitCode}): ${label}`);
  }
}

/** Find dylibs matching a pattern recursively in a directory. */
function findDylibs(dir: string, pattern: RegExp): string[] {
  const results: string[] = [];
  function walk(current: string) {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const full = resolve(current, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (pattern.test(entry.name)) {
        results.push(full);
      }
    }
  }
  walk(dir);
  return results;
}

/** Get the dylib install name using otool. */
async function getInstallName(dylib: string): Promise<string> {
  const proc = Bun.spawn(["otool", "-D", dylib], { stdout: "pipe" });
  const text = await new Response(proc.stdout).text();
  await proc.exited;
  // otool -D output: first line is the file path, second line is the install name
  const lines = text.trim().split("\n");
  return lines[1]?.trim() ?? "";
}

/** Get all dylib dependencies using otool -L. */
async function getDeps(dylib: string): Promise<string[]> {
  const proc = Bun.spawn(["otool", "-L", dylib], { stdout: "pipe" });
  const text = await new Response(proc.stdout).text();
  await proc.exited;
  return text
    .trim()
    .split("\n")
    .slice(1) // skip the first line (the file itself)
    .map((line) => line.trim().split(" ")[0] ?? "")
    .filter((dep) => dep.length > 0);
}

function basenameOf(path: string): string {
  const basename = path.split("/").pop();
  if (!basename) {
    throw new Error(`Could not determine basename for path: ${path}`);
  }
  return basename;
}

function firstPath(paths: readonly string[], description: string): string {
  const match = paths[0];
  if (!match) {
    throw new Error(`${description} not found in build output`);
  }
  return match;
}

async function setLoaderPathInstallName(dylib: string): Promise<string> {
  const basename = basenameOf(dylib);
  await run(["install_name_tool", "-id", `@loader_path/${basename}`, dylib], {
    cwd: LIB_DIR,
  });
  return basename;
}

function shouldRewriteDependency(
  dependency: string,
  candidatePath: string,
  installNames: ReadonlyMap<string, string>,
): boolean {
  const candidateName = basenameOf(candidatePath);
  const currentInstallName = installNames.get(candidatePath);
  return dependency === currentInstallName || dependency.endsWith(`/${candidateName}`);
}

async function rewriteLoaderPathDependencies(
  dylib: string,
  dylibs: readonly string[],
  installNames: ReadonlyMap<string, string>,
): Promise<void> {
  const deps = await getDeps(dylib);

  for (const dep of deps) {
    for (const otherDylib of dylibs) {
      const otherBasename = basenameOf(otherDylib);
      if (!shouldRewriteDependency(dep, otherDylib, installNames)) {
        continue;
      }

      const loaderPath = `@loader_path/${otherBasename}`;
      if (dep === loaderPath) {
        continue;
      }

      await run(["install_name_tool", "-change", dep, loaderPath, dylib], {
        cwd: LIB_DIR,
      });
    }
  }
}

/** Fix rpaths so dylibs can find each other in native/lib/. */
async function fixRpaths(): Promise<void> {
  console.log("\nFixing rpaths...");

  const dylibs = readdirSync(LIB_DIR)
    .filter((f) => f.endsWith(".dylib"))
    .map((f) => resolve(LIB_DIR, f));

  // Build a map of library base names to their current install names
  const installNames = new Map<string, string>();
  for (const dylib of dylibs) {
    const name = await getInstallName(dylib);
    if (name) {
      installNames.set(dylib, name);
    }
  }

  // For each dylib, fix its install name and rewrite deps to use @loader_path
  for (const dylib of dylibs) {
    const basename = await setLoaderPathInstallName(dylib);
    await rewriteLoaderPathDependencies(dylib, dylibs, installNames);

    console.log(`  Fixed: ${basename}`);
  }
}

async function main(): Promise<void> {
  console.log("mlx-ts native build");
  console.log("=".repeat(50));

  // Check if dylibs already exist (skip rebuild)
  const mlxcExists = existsSync(resolve(LIB_DIR, "libmlxc.dylib"));
  const mlxExists = existsSync(resolve(LIB_DIR, "libmlx.dylib"));
  if (mlxcExists && mlxExists) {
    console.log("\nDylibs already exist in native/lib/. Skipping build.");
    console.log("To force rebuild, delete native/lib/ and native/build/.");
    return;
  }

  // Create directories
  mkdirSync(BUILD_DIR, { recursive: true });
  mkdirSync(LIB_DIR, { recursive: true });

  // Resolve the macOS SDK path — Xcode's SDK includes the Metal compiler,
  // whereas the Command Line Tools SDK does not.
  const sdkProc = Bun.spawn(["xcrun", "--sdk", "macosx", "--show-sdk-path"], {
    stdout: "pipe",
  });
  const sdkPath = (await new Response(sdkProc.stdout).text()).trim();
  await sdkProc.exited;
  if (!sdkPath) {
    throw new Error("Could not resolve macOS SDK path via xcrun");
  }
  console.log(`Using macOS SDK: ${sdkPath}`);

  // Configure CMake
  const cpuCount = navigator.hardwareConcurrency ?? 8;
  await run(
    [
      "cmake",
      NATIVE_DIR,
      "-DBUILD_SHARED_LIBS=ON",
      "-DCMAKE_BUILD_TYPE=Release",
      `-DCMAKE_OSX_SYSROOT=${sdkPath}`,
    ],
    { cwd: BUILD_DIR },
  );

  // Build
  await run(["cmake", "--build", ".", "--config", "Release", "-j", String(cpuCount)], {
    cwd: BUILD_DIR,
  });

  // Find and copy dylibs
  console.log("\nSearching for built dylibs...");
  const mlxcLibs = findDylibs(BUILD_DIR, /^libmlxc\.dylib$/);
  const mlxLibs = findDylibs(BUILD_DIR, /^libmlx\.dylib$/);

  if (mlxcLibs.length === 0) {
    throw new Error("libmlxc.dylib not found in build output");
  }
  if (mlxLibs.length === 0) {
    throw new Error("libmlx.dylib not found in build output");
  }

  // Copy the first match of each
  console.log(`\nCopying dylibs to ${LIB_DIR}/`);
  for (const [src, name] of [
    [firstPath(mlxcLibs, "libmlxc.dylib"), "libmlxc.dylib"],
    [firstPath(mlxLibs, "libmlx.dylib"), "libmlx.dylib"],
  ] as const) {
    const dst = resolve(LIB_DIR, name);
    await Bun.write(dst, Bun.file(src));
    console.log(`  ${src} → ${dst}`);
  }

  // Fix rpaths so the dylibs can find each other
  await fixRpaths();

  console.log("\nBuild complete!");
  console.log(`  libmlxc.dylib: ${resolve(LIB_DIR, "libmlxc.dylib")}`);
  console.log(`  libmlx.dylib:  ${resolve(LIB_DIR, "libmlx.dylib")}`);
}

main().catch((error) => {
  console.error("\nBuild failed:", error.message);
  process.exit(1);
});
