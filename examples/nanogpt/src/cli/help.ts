import type { TrainEvent } from "../train";

export function formatHelp(): string {
  return [
    "description: Train GPT models, generate text, and export nanoGPT weights",
    "usage[3]:",
    "  nanogpt train [options]",
    "  nanogpt generate --checkpoint <path> [options]",
    "  nanogpt export --checkpoint <path> --output <file>.safetensors",
    "commands[3]{command,description}:",
    '  "train","Train a GPT model on text data"',
    '  "generate","Generate text from a checkpoint"',
    '  "export","Export model weights as safetensors"',
    "help[3]:",
    '  "nanogpt train --help"',
    '  "nanogpt generate --help"',
    '  "nanogpt export --help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"success or help"',
    '  1,"runtime failure"',
    '  2,"usage error"',
  ].join("\n");
}

export function printHelp(): void {
  process.stdout.write(`${formatHelp()}\n`);
}

export function formatTrainHelp(): string {
  return [
    "description: Train a GPT model on text data",
    "usage[1]:",
    "  nanogpt train [options]",
    "options[30]{flag,description}:",
    '  "--preset <name>","Model preset: gpt-tiny or gpt-small; default gpt-tiny"',
    '  "--gradient-checkpointing <true|false>","Override the preset gradient-checkpointing setting"',
    '  "--data <path>","Training text file; default cached/downloaded Shakespeare"',
    '  "--max-steps <n>","Maximum training steps; default preset-specific"',
    '  "--batch-size <n>","Batch size; default preset-specific"',
    '  "--grad-accum <n>","Gradient accumulation steps; default preset-specific"',
    '  "--eval-interval <n>","Evaluate every N steps; default preset-specific"',
    '  "--eval-steps <n>","Evaluation batch count; default preset-specific"',
    '  "--log-interval <n>","Emit step progress every N steps; default preset-specific"',
    '  "--lr <n>","Peak learning rate; default 3e-4"',
    '  "--weight-decay <n>","Weight decay; default 0.1"',
    '  "--max-grad-norm <n|none>","Global gradient clipping threshold; default 1.0"',
    '  "--warmup-steps <n>","Warmup steps; default preset-specific"',
    '  "--min-lr <n>","Minimum learning rate; default preset-specific"',
    '  "--seed <n>","Training seed for MLX and batching; default 42"',
    '  "--resume <path>","Resume exact training state from a resume checkpoint"',
    '  "--warm-start <path>","Initialize weights from a checkpoint with fresh optimizer state"',
    '  "--checkpoint-dir <path>","Checkpoint directory; default .nanogpt-checkpoints"',
    '  "--snapshot-interval <n>","Save snapshot checkpoints every N eval steps; default 250"',
    '  "--resume-interval <n>","Save resumable checkpoints every N eval steps; default 1000"',
    '  "--sample-interval <n>","Emit generated samples every N training steps"',
    '  "--sample-tokens <n>","Generated sample token count; default 200"',
    '  "--early-stop-patience <n|none>","Stop after N evals without meaningful val-loss improvement; default 8"',
    '  "--early-stop-min-delta <n>","Minimum val-loss improvement to reset patience; default 0.02"',
    '  "--memory-limit-mb <n>","Set the MLX allocator memory limit in MB"',
    '  "--cache-limit-mb <n>","Set the MLX allocator cache limit in MB"',
    '  "--wired-limit-mb <n>","Set the MLX wired-memory limit in MB"',
    '  "--run-dir <path>","Supervised-run control directory"',
    '  "--json","Emit JSON event lines to stdout"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"training completed, stopped cleanly, cancelled cleanly, or help"',
    '  1,"runtime failure"',
    '  2,"usage error"',
  ].join("\n");
}

export function printTrainHelp(): void {
  process.stdout.write(`${formatTrainHelp()}\n`);
}

export function formatGenerateHelp(): string {
  return [
    "description: Generate text from a nanoGPT checkpoint",
    "usage[1]:",
    "  nanogpt generate --checkpoint <path> [options]",
    "options[6]{flag,description}:",
    '  "--checkpoint <path>","Checkpoint directory; required"',
    '  "--prompt <text>","Prompt text; default newline"',
    '  "--max-tokens <n>","Tokens to generate; default 500"',
    '  "--temperature <n>","Sampling temperature; 0 is greedy; default 0.8"',
    '  "--json","Emit a JSON result object to stdout"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"generation completed or help"',
    '  1,"runtime failure"',
    '  2,"usage error"',
  ].join("\n");
}

export function printGenerateHelp(): void {
  process.stdout.write(`${formatGenerateHelp()}\n`);
}

export function formatExportHelp(): string {
  return [
    "description: Export nanoGPT checkpoint weights as safetensors",
    "usage[1]:",
    "  nanogpt export --checkpoint <path> --output <file>.safetensors",
    "options[3]{flag,description}:",
    '  "--checkpoint <path>","Checkpoint directory; required"',
    '  "--output <path>","Output safetensors path; required"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"export completed or help"',
    '  1,"runtime failure"',
    '  2,"usage error"',
  ].join("\n");
}

export function printExportHelp(): void {
  process.stdout.write(`${formatExportHelp()}\n`);
}

export function trainingTableHeader(): string {
  return (
    "  Step    Loss    Val     LR        Tokens/sec\n" +
    "  ──────────────────────────────────────────────\n"
  );
}

export function formatStepEvent(event: Extract<TrainEvent, { type: "step" }>): string {
  return (
    `  ${String(event.step).padStart(5)}  ${event.loss.toFixed(4).padStart(7)}` +
    `                ${event.learningRate.toExponential(1).padStart(8)}  ${Math.round(
      event.tokensPerSec,
    )
      .toLocaleString()
      .padStart(10)}\n`
  );
}

export function formatEvalEvent(event: Extract<TrainEvent, { type: "eval" }>): string {
  return `  ${String(event.step).padStart(5)}  ${event.trainLoss
    .toFixed(4)
    .padStart(7)}  ${event.valLoss.toFixed(4).padStart(7)}\n`;
}
