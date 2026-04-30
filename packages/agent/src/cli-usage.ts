import { DEFAULT_AGENT_MAX_TOKENS } from "./chat-model";

type AgentCliOptionRow = {
  flag: string;
  value: string;
  defaultValue: string;
  description: string;
};

const OPTION_ROWS: readonly AgentCliOptionRow[] = [
  {
    flag: "--endpoint",
    value: "<url>",
    defaultValue: "http://127.0.0.1:8000",
    description: "Base endpoint.",
  },
  { flag: "--model", value: "<id>", defaultValue: "required", description: "Served model id." },
  {
    flag: "--prompt",
    value: "<text>",
    defaultValue: "interactive",
    description: "Run one finite agent turn and exit.",
  },
  {
    flag: "--cwd",
    value: "<path>",
    defaultValue: "current directory",
    description: "Directory exposed to read-only file tools.",
  },
  { flag: "--api-key", value: "<key>", defaultValue: "unset", description: "Bearer token." },
  {
    flag: "--max-tokens",
    value: "<n>",
    defaultValue: String(DEFAULT_AGENT_MAX_TOKENS),
    description: "Max assistant tokens per loop step.",
  },
  {
    flag: "--temperature",
    value: "<n>",
    defaultValue: "model config",
    description: "Sampling temperature, 0 to 2.",
  },
  { flag: "--greedy", value: "", defaultValue: "off", description: "Alias for --temperature 0." },
  {
    flag: "--deterministic",
    value: "",
    defaultValue: "off",
    description: "Alias for --temperature 0.",
  },
  {
    flag: "--thinking",
    value: "",
    defaultValue: "template default",
    description: "Enable compatible thinking templates.",
  },
  {
    flag: "--no-thinking",
    value: "",
    defaultValue: "template default",
    description: "Disable compatible thinking templates.",
  },
  { flag: "--stream", value: "", defaultValue: "on", description: "Use streaming chat transport." },
  {
    flag: "--no-stream",
    value: "",
    defaultValue: "off",
    description: "Use non-streaming chat completions.",
  },
  {
    flag: "--max-iterations",
    value: "<n>",
    defaultValue: "8",
    description: "Max model/tool loop steps per turn.",
  },
  { flag: "--verbose", value: "", defaultValue: "off", description: "Enable fetch diagnostics." },
  { flag: "--help", value: "", defaultValue: "off", description: "Show this help." },
];

/** Format `mlxts-agent --help` as compact agent-readable stdout. */
export function formatAgentUsage(): string {
  const lines = [
    "agent_cli:",
    '  description: "Talk to an OpenAI-compatible local chat endpoint with read-only tools."',
    '  usage: "mlxts-agent [run] --model <served-model-id> [--prompt <text>] [options]"',
    `options[${OPTION_ROWS.length}]{flag,value,default,description}:`,
  ];
  for (const row of OPTION_ROWS) {
    lines.push(
      `  ${JSON.stringify(row.flag)},${JSON.stringify(row.value)},${JSON.stringify(
        row.defaultValue,
      )},${JSON.stringify(row.description)}`,
    );
  }
  lines.push("examples[2]:");
  lines.push('  "mlxts-agent --model mlx-community/Qwen3.6-27B-4bit --cwd ."');
  lines.push(
    '  "mlxts-agent run --model mlx-community/Qwen3.6-27B-4bit --prompt \\"List the top-level files.\\" --cwd ."',
  );
  return lines.join("\n");
}
