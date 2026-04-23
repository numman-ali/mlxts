import type { TrainEvent } from "../train";

export function printHelp(): void {
  process.stdout.write(`nanogpt — Train GPT models and generate text

Usage:
  nanogpt train [options]     Train a GPT model on text data
  nanogpt generate [options]  Generate text from a checkpoint
  nanogpt export [options]    Export model weights as safetensors

Train options:
  --preset <name>            Model preset: gpt-tiny (default), gpt-small
                             gpt-small enables gradient checkpointing by default
  --gradient-checkpointing <true|false>
                             Override the preset's gradient checkpointing setting
  --data <path>              Path to training text file (default: cached/downloaded Shakespeare)
  --max-steps <n>            Maximum training steps (default: preset-specific safe default)
  --batch-size <n>           Batch size (default: preset-specific safe default)
  --grad-accum <n>           Gradient accumulation steps (default: preset-specific safe default)
  --lr <n>                   Peak learning rate (default: 3e-4)
  --weight-decay <n>         Weight decay (default: 0.1)
  --seed <n>                 Training seed for MLX + batching (default: 42)
  --resume <path>            Resume training from a checkpoint directory
  --warm-start <path>        Initialize weights from a checkpoint and start with a fresh optimizer
  --checkpoint-dir <path>    Directory for periodic/final checkpoints (default: .nanogpt-checkpoints)
  --snapshot-interval <n>    Save snapshot checkpoints every N eval steps (default: 250)
  --resume-interval <n>      Save resumable checkpoints every N eval steps (default: 1000)
  --early-stop-patience <n|none>
                             Stop after N evals without meaningful val-loss improvement (default: 8)
  --early-stop-min-delta <n> Minimum val-loss improvement required to reset patience (default: 0.02)
  --memory-limit-mb <n>      Set the MLX allocator memory limit in MB
  --cache-limit-mb <n>       Set the MLX allocator cache limit in MB
  --wired-limit-mb <n>       Set the MLX wired-memory limit in MB
  --json                     Emit JSON events to stdout
  --help                     Show this help

Generate options:
  --checkpoint <path>        Path to checkpoint directory (required)
  --prompt <text>            Prompt text (default: newline)
  --max-tokens <n>           Tokens to generate (default: 500)
  --temperature <n>          Sampling temperature, 0=greedy (default: 0.8)
  --json                     Emit a JSON result object to stdout
  --help                     Show this help

Examples:
  nanogpt train --preset gpt-tiny
  nanogpt train --resume .nanogpt-checkpoints/gpt-small-resume-step-500
  nanogpt train --warm-start .nanogpt-checkpoints/gpt-small-snapshot-step-50 --max-steps 500
  nanogpt train --preset gpt-small --max-steps 5000 --grad-accum 8
  nanogpt generate --checkpoint .nanogpt-checkpoints/gpt-tiny-resume-step-500 --prompt "To be or"
  nanogpt export --checkpoint .nanogpt-checkpoints/gpt-small-resume-step-500 --output model.safetensors
`);
}

export function printTrainHelp(): void {
  process.stdout.write(`nanogpt train — Train a GPT model on text data

Usage:
  nanogpt train [options]

Options:
  --preset <name>            Model preset: gpt-tiny (default), gpt-small
                             gpt-small enables gradient checkpointing by default
  --gradient-checkpointing <true|false>
                             Override the preset's gradient checkpointing setting
  --data <path>              Path to training text file (default: cached/downloaded Shakespeare)
  --max-steps <n>            Maximum training steps (default: preset-specific safe default)
  --batch-size <n>           Batch size (default: preset-specific safe default)
  --grad-accum <n>           Gradient accumulation steps (default: preset-specific safe default)
  --lr <n>                   Peak learning rate (default: 3e-4)
  --weight-decay <n>         Weight decay (default: 0.1)
  --max-grad-norm <n|none>   Global gradient clipping threshold (default: 1.0)
  --seed <n>                 Training seed for MLX + batching (default: 42)
  --resume <path>            Resume training from a checkpoint directory
  --warm-start <path>        Initialize weights from a checkpoint and start with a fresh optimizer
  --checkpoint-dir <path>    Directory for periodic/final checkpoints (default: .nanogpt-checkpoints)
  --snapshot-interval <n>    Save snapshot checkpoints every N eval steps (default: 250)
  --resume-interval <n>      Save resumable checkpoints every N eval steps (default: 1000)
  --sample-interval <n>      Emit a generated sample every N training steps (default: 0 for plain train, snapshot interval for supervised runs)
  --sample-tokens <n>        Number of tokens to generate per sample (default: 200)
  --early-stop-patience <n|none>
                             Stop after N evals without meaningful val-loss improvement (default: 8)
  --early-stop-min-delta <n> Minimum val-loss improvement required to reset patience (default: 0.02)
  --memory-limit-mb <n>      Set the MLX allocator memory limit in MB
  --cache-limit-mb <n>       Set the MLX allocator cache limit in MB
  --wired-limit-mb <n>       Set the MLX wired-memory limit in MB
  --json                     Emit JSON events to stdout
  --help                     Show this help
`);
}

export function printGenerateHelp(): void {
  process.stdout.write(`nanogpt generate — Generate text from a checkpoint

Usage:
  nanogpt generate --checkpoint <path> [options]

Options:
  --checkpoint <path>        Path to checkpoint directory (required)
  --prompt <text>            Prompt text (default: newline)
  --max-tokens <n>           Tokens to generate (default: 500)
  --temperature <n>          Sampling temperature, 0=greedy (default: 0.8)
  --json                     Emit a JSON result object to stdout
  --help                     Show this help
`);
}

export function printExportHelp(): void {
  process.stdout.write(`nanogpt export — Export model weights as safetensors

Usage:
  nanogpt export --checkpoint <path> --output <file>.safetensors

Options:
  --checkpoint <path>        Path to checkpoint directory (required)
  --output <path>            Output safetensors path (required)
  --help                     Show this help
`);
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
