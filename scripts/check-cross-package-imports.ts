/**
 * Enforces the package dependency graph declared in docs/ecosystem-structure.md.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { join } from "path";

const PROJECT_ROOT = join(import.meta.dirname, "..");
const PACKAGES_ROOT = join(PROJECT_ROOT, "packages");
const ECOSYSTEM_DOC = "docs/ecosystem-structure.md";
const DEPENDENCY_GRAPH_HEADING = "## Dependency Graph";
const WORKSPACE_SCOPE = "@mlxts/";

type PackageInfo = {
  dir: string;
  name: string;
  shortName: string;
  root: string;
};

type DependencyEdge = {
  from: string;
  to: string;
  source: string;
};

type DependencyGraphBlock = {
  content: string;
  startLine: number;
};

type ParsedGraphLine = {
  packageName: string;
  dependencies: Set<string>;
};

function packageNameFromSpecifier(specifier: string): string | null {
  if (!specifier.startsWith(WORKSPACE_SCOPE)) {
    return null;
  }
  const shortName = specifier.slice(WORKSPACE_SCOPE.length).split("/")[0];
  return shortName === undefined || shortName === "" ? null : shortName;
}

function readJson(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
}

function workspacePackages(): PackageInfo[] {
  return readdirSync(PACKAGES_ROOT)
    .filter((entry) => {
      const packageRoot = join(PACKAGES_ROOT, entry);
      return statSync(packageRoot).isDirectory() && existsSync(join(packageRoot, "package.json"));
    })
    .map((dir) => {
      const root = join(PACKAGES_ROOT, dir);
      const manifest = readJson(join(root, "package.json"));
      const name = manifest.name;
      if (typeof name !== "string" || !name.startsWith(WORKSPACE_SCOPE)) {
        throw new Error(`packages/${dir}/package.json must declare an @mlxts/* package name.`);
      }
      const shortName = packageNameFromSpecifier(name);
      if (shortName === null) {
        throw new Error(`packages/${dir}/package.json has an invalid package name ${name}.`);
      }
      return { dir, name, shortName, root };
    })
    .sort((left, right) => left.shortName.localeCompare(right.shortName));
}

function dependencyGraphBlock(): DependencyGraphBlock {
  const content = readFileSync(join(PROJECT_ROOT, ECOSYSTEM_DOC), "utf8");
  const headingIndex = content.indexOf(DEPENDENCY_GRAPH_HEADING);
  if (headingIndex === -1) {
    throw new Error(`${ECOSYSTEM_DOC}: missing ${DEPENDENCY_GRAPH_HEADING}.`);
  }
  const fenceStart = content.indexOf("```", headingIndex);
  const fenceEnd = fenceStart === -1 ? -1 : content.indexOf("```", fenceStart + 3);
  if (fenceStart === -1 || fenceEnd === -1) {
    throw new Error(`${ECOSYSTEM_DOC}: dependency graph must be a fenced code block.`);
  }
  const startLine = content.slice(0, fenceStart).split(/\r?\n/).length + 1;
  return {
    content: content.slice(fenceStart + 3, fenceEnd),
    startLine,
  };
}

function parseDocDependencies(
  rawDependencies: string,
  packageName: string,
  knownPackages: Set<string>,
  lineNumber: number,
  errors: string[],
): Set<string> {
  const normalized = rawDependencies.trim().toLowerCase();
  if (normalized === "none") {
    return new Set();
  }
  if (normalized === "all packages") {
    return new Set([...knownPackages].filter((candidate) => candidate !== packageName));
  }

  const dependencies = new Set<string>();
  for (const token of rawDependencies.split(",")) {
    const dependencyName = token.trim();
    if (dependencyName === "") {
      continue;
    }
    const dependency = packageNameFromSpecifier(dependencyName);
    if (dependency === null) {
      errors.push(
        `${ECOSYSTEM_DOC}:${lineNumber}: dependency ${dependencyName} must use @mlxts/<name>.`,
      );
      continue;
    }
    if (!knownPackages.has(dependency)) {
      errors.push(`${ECOSYSTEM_DOC}:${lineNumber}: unknown package @mlxts/${dependency}.`);
      continue;
    }
    if (dependency !== packageName) {
      dependencies.add(dependency);
    }
  }
  return dependencies;
}

function parseGraphLine(
  rawLine: string,
  lineNumber: number,
  knownPackages: Set<string>,
  errors: string[],
): ParsedGraphLine | null {
  const line = rawLine.trim();
  if (line === "") {
    return null;
  }
  const packageMatch = line.match(/^@mlxts\/([a-z0-9_-]+)\s*->\s*(.+)$/i);
  if (packageMatch === null) {
    errors.push(
      `${ECOSYSTEM_DOC}:${lineNumber}: dependency graph lines must be "@mlxts/<name> -> <deps>".`,
    );
    return null;
  }
  const packageName = packageMatch[1];
  if (packageName === undefined || !knownPackages.has(packageName)) {
    errors.push(`${ECOSYSTEM_DOC}:${lineNumber}: unknown package @mlxts/${packageName}.`);
    return null;
  }
  const rawDependencies = packageMatch[2];
  if (rawDependencies === undefined) {
    errors.push(`${ECOSYSTEM_DOC}:${lineNumber}: missing dependency list.`);
    return null;
  }
  return {
    packageName,
    dependencies: parseDocDependencies(
      rawDependencies,
      packageName,
      knownPackages,
      lineNumber,
      errors,
    ),
  };
}

function dependencyKey(dependencies: Set<string>): string {
  return [...dependencies].sort().join(",");
}

function recordGraphEntry(
  graph: Map<string, Set<string>>,
  entry: ParsedGraphLine,
  errors: string[],
): void {
  const existing = graph.get(entry.packageName);
  if (existing !== undefined && dependencyKey(existing) !== dependencyKey(entry.dependencies)) {
    errors.push(`${ECOSYSTEM_DOC}: duplicate graph entries for @mlxts/${entry.packageName}.`);
  }
  graph.set(entry.packageName, entry.dependencies);
}

function documentedDependencyGraph(packages: readonly PackageInfo[]): Map<string, Set<string>> {
  const knownPackages = new Set(packages.map((entry) => entry.shortName));
  const graph = new Map<string, Set<string>>();
  const errors: string[] = [];
  const graphBlock = dependencyGraphBlock();
  const lines = graphBlock.content.split("\n");

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index] ?? "";
    const lineNumber = graphBlock.startLine + index;
    const entry = parseGraphLine(rawLine, lineNumber, knownPackages, errors);
    if (entry !== null) {
      recordGraphEntry(graph, entry, errors);
    }
  }

  for (const entry of packages) {
    if (!graph.has(entry.shortName)) {
      errors.push(`${ECOSYSTEM_DOC}: missing dependency graph entry for ${entry.name}.`);
    }
  }

  if (errors.length > 0) {
    throw new Error(errors.join("\n"));
  }
  return graph;
}

function workspaceDependenciesFromManifest(info: PackageInfo): DependencyEdge[] {
  const manifest = readJson(join(info.root, "package.json"));
  const dependencyRecords = [
    manifest.dependencies,
    manifest.peerDependencies,
    manifest.optionalDependencies,
  ].filter(
    (value): value is Record<string, unknown> => typeof value === "object" && value !== null,
  );

  const dependencies = new Set<string>();
  for (const record of dependencyRecords) {
    for (const name of Object.keys(record)) {
      const dependency = packageNameFromSpecifier(name);
      if (dependency !== null && dependency !== info.shortName) {
        dependencies.add(dependency);
      }
    }
  }

  return [...dependencies].sort().map((to) => ({
    from: info.shortName,
    to,
    source: `packages/${info.dir}/package.json`,
  }));
}

async function packageSourceFiles(info: PackageInfo): Promise<string[]> {
  const files: string[] = [];
  const patterns = ["src/**/*.ts", "scripts/**/*.ts"];
  for (const pattern of patterns) {
    const glob = new Bun.Glob(pattern);
    for await (const path of glob.scan({ cwd: info.root })) {
      if (!path.endsWith(".test.ts")) {
        files.push(path);
      }
    }
  }
  return files.sort((left, right) => left.localeCompare(right));
}

function importSpecifiers(content: string): string[] {
  const specifiers: string[] = [];
  const fromPattern = /(?:from\s+["']|import\s*\(\s*["'])(@mlxts\/[^"']+)["']/g;
  const barePattern = /import\s+["'](@mlxts\/[^"']+)["']/g;
  for (const pattern of [fromPattern, barePattern]) {
    for (const match of content.matchAll(pattern)) {
      const specifier = match[1];
      if (specifier !== undefined) {
        specifiers.push(specifier);
      }
    }
  }
  return specifiers;
}

async function workspaceImportsFromSource(info: PackageInfo): Promise<DependencyEdge[]> {
  const edges: DependencyEdge[] = [];
  for (const path of await packageSourceFiles(info)) {
    const content = readFileSync(join(info.root, path), "utf8");
    for (const specifier of importSpecifiers(content)) {
      const dependency = packageNameFromSpecifier(specifier);
      if (dependency !== null && dependency !== info.shortName) {
        edges.push({
          from: info.shortName,
          to: dependency,
          source: `packages/${info.dir}/${path}`,
        });
      }
    }
  }
  return edges;
}

function uniqueEdges(edges: readonly DependencyEdge[]): DependencyEdge[] {
  const seen = new Set<string>();
  const unique: DependencyEdge[] = [];
  for (const edge of edges) {
    const key = `${edge.from}\0${edge.to}\0${edge.source}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(edge);
    }
  }
  return unique.sort(
    (left, right) =>
      left.from.localeCompare(right.from) ||
      left.to.localeCompare(right.to) ||
      left.source.localeCompare(right.source),
  );
}

const packages = workspacePackages();
const knownPackages = new Set(packages.map((entry) => entry.shortName));
const graph = documentedDependencyGraph(packages);
const edges = uniqueEdges(
  (
    await Promise.all(
      packages.map(async (info) => [
        ...workspaceDependenciesFromManifest(info),
        ...(await workspaceImportsFromSource(info)),
      ]),
    )
  ).flat(),
);

const unknownEdges = edges.filter((edge) => !knownPackages.has(edge.to));
const violations = edges.filter(
  (edge) => knownPackages.has(edge.to) && !(graph.get(edge.from)?.has(edge.to) ?? false),
);

if (unknownEdges.length > 0 || violations.length > 0) {
  console.error(`Cross-package imports must follow ${ECOSYSTEM_DOC}.\n`);
  for (const edge of unknownEdges) {
    console.error(
      `  ${edge.source}: @mlxts/${edge.from} depends on unknown workspace package @mlxts/${edge.to}`,
    );
  }
  for (const violation of violations) {
    const allowed = [...(graph.get(violation.from) ?? [])].sort();
    const allowedText =
      allowed.length === 0 ? "none" : allowed.map((name) => `@mlxts/${name}`).join(", ");
    console.error(
      `  ${violation.source}: @mlxts/${violation.from} may not depend on @mlxts/${violation.to} (allowed: ${allowedText})`,
    );
  }
  process.exit(1);
}

console.log(
  `Checked ${edges.length} package dependency edges against ${ECOSYSTEM_DOC}. No stack inversions found.`,
);
