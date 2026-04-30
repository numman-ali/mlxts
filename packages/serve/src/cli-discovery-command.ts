import {
  type DiscoveredLocalModelSource,
  discoverLocalModelSources,
} from "./model-loading/discovery";

export type ServeDiscoverCliOptions = {
  modelRoots: readonly string[];
  full: boolean;
};

export type ServeDiscoverCliParseResult =
  | { kind: "discover"; options: ServeDiscoverCliOptions }
  | { kind: "help"; exitCode: number; message?: string };

export type ServeDiscoverRootResult = {
  root: string;
  models: readonly DiscoveredLocalModelSource[];
};

export type ServeDiscoverCliRuntime = {
  discoverLocalModelSources?: (root: string) => readonly DiscoveredLocalModelSource[];
  log?: (message: string) => void;
  exit?: (code: number) => void;
};

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

function parseDiscoverState(argv: readonly string[]): ServeDiscoverCliOptions {
  const modelRoots: string[] = [];
  let full = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--model-root":
        modelRoots.push(readStringFlag(arg, argv[index + 1]));
        index += 1;
        break;
      case "--full":
        full = true;
        break;
      default:
        if (arg?.startsWith("--")) {
          throw new Error(`Unknown argument: ${arg}`);
        }
        throw new Error(`Unexpected positional argument: ${arg ?? "<missing>"}`);
    }
  }

  if (modelRoots.length === 0) {
    throw new Error("Missing required --model-root <directory>.");
  }

  return { modelRoots: [...new Set(modelRoots)], full };
}

export function parseServeDiscoverArgs(argv: readonly string[]): ServeDiscoverCliParseResult {
  if (argv.includes("--help")) {
    return { kind: "help", exitCode: 0 };
  }

  try {
    return { kind: "discover", options: parseDiscoverState(argv) };
  } catch (error) {
    return {
      kind: "help",
      exitCode: 2,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

function toon(value: string | number | boolean): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatHelp(commands: readonly string[]): string[] {
  if (commands.length === 0) {
    return [];
  }
  return [`help[${commands.length}]:`, ...commands.map((command) => `  ${command}`)];
}

function formatRoots(results: readonly ServeDiscoverRootResult[]): string[] {
  return [
    `roots[${results.length}]{path,count}:`,
    ...results.map((result) => `  ${toon(result.root)},${result.models.length}`),
  ];
}

function formatModelTable(models: readonly DiscoveredLocalModelSource[], full: boolean): string[] {
  const columns = full
    ? "model_id,source,model_type,has_vision,architectures"
    : "model_id,source,model_type,has_vision";
  return [
    `models[${models.length}]{${columns}}:`,
    ...models.map((model) => {
      const base = [
        toon(model.modelId),
        toon(model.source),
        toon(model.modelType),
        toon(model.hasVisionConfig),
      ];
      if (full) {
        base.push(toon(model.architectures.join("|")));
      }
      return `  ${base.join(",")}`;
    }),
  ];
}

function formatServeCommand(modelRoots: readonly string[]): string {
  const flags = modelRoots.map((root) => `--model-root ${JSON.stringify(root)}`).join(" ");
  return `Run \`mlxts-serve ${flags}\` to serve discovered models lazily`;
}

export function formatServeDiscoverUsage(): string {
  return [
    "description: Discover supported local autoregressive checkpoints under one or more roots",
    "usage[2]:",
    "  mlxts-serve discover --model-root <directory>",
    "  mlxts-serve discover --model-root <directory> --full",
    "options[3]{flag,description}:",
    '  "--model-root <directory>","Root to scan; repeat for multiple roots"',
    '  "--full","Include architecture metadata for each discovered model"',
    '  "--help","Show this help"',
  ].join("\n");
}

export function formatServeDiscoverError(message: string, code = "usage"): string {
  return [
    "error:",
    `  code: ${toon(code)}`,
    `  message: ${toon(message)}`,
    ...formatHelp(["Run `mlxts-serve discover --model-root <directory>` to list checkpoints"]),
  ].join("\n");
}

export function formatServeDiscoverResults(
  results: readonly ServeDiscoverRootResult[],
  full: boolean,
): string {
  const models = results.flatMap((result) => result.models);
  if (models.length === 0) {
    return [
      ...formatRoots(results),
      "models: 0 supported autoregressive checkpoints found",
      ...formatHelp(["Add a checkpoint folder containing config.json and .safetensors"]),
    ].join("\n");
  }

  return [
    ...formatRoots(results),
    `count: ${models.length} of ${models.length} total`,
    ...formatModelTable(models, full),
    ...formatHelp([
      formatServeCommand(results.map((result) => result.root)),
      "Run `mlxts-serve discover --model-root <directory> --full` to include architecture metadata",
    ]),
  ].join("\n");
}

export function discoverServeModelRoots(
  options: ServeDiscoverCliOptions,
  discover: (root: string) => readonly DiscoveredLocalModelSource[] = discoverLocalModelSources,
): ServeDiscoverRootResult[] {
  return options.modelRoots.map((root) => ({
    root,
    models: discover(root),
  }));
}

export function runServeDiscoverCli(
  argv: readonly string[],
  runtime: ServeDiscoverCliRuntime = {},
): void {
  const log = runtime.log ?? console.log;
  const exit = runtime.exit ?? process.exit;
  const parsed = parseServeDiscoverArgs(argv);
  if (parsed.kind === "help") {
    log(
      parsed.message === undefined
        ? formatServeDiscoverUsage()
        : formatServeDiscoverError(parsed.message),
    );
    exit(parsed.exitCode);
    return;
  }

  try {
    const discover = runtime.discoverLocalModelSources ?? discoverLocalModelSources;
    const results = discoverServeModelRoots(parsed.options, discover);
    log(formatServeDiscoverResults(results, parsed.options.full));
    exit(0);
  } catch (exception) {
    log(
      formatServeDiscoverError(
        exception instanceof Error ? exception.message : String(exception),
        "discovery",
      ),
    );
    exit(1);
  }
}
