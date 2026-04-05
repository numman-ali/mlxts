#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from safetensors.numpy import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fixed LLaMA forward-pass parity fixture from MLX Python.",
    )
    parser.add_argument("model", help="Local MLX model path or repo id")
    parser.add_argument("output_dir", help="Directory that will receive fixture.json and logits.safetensors")
    parser.add_argument(
        "--prompt-text",
        default="Hello, world!",
        help="Prompt text to tokenize when --token-ids is not provided",
    )
    parser.add_argument(
        "--token-ids",
        default="",
        help="Comma-separated token ids. When provided, this overrides --prompt-text.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max absolute diff tolerated by the Bun-side verifier",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(args.model)
    if args.token_ids.strip():
        token_ids = [int(part) for part in args.token_ids.split(",") if part.strip()]
        prompt_text = None
    else:
        prompt_text = args.prompt_text
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    input_ids = mx.array([token_ids], dtype=mx.int32)
    logits = model(input_ids)
    mx.eval(logits)

    save_file({"logits": np.array(logits)}, output_dir / "logits.safetensors")
    (output_dir / "fixture.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "promptText": prompt_text,
                "tokenIds": token_ids,
                "tolerance": args.tolerance,
                "logitsTensorName": "logits",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
