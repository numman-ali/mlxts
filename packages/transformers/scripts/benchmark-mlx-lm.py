#!/usr/bin/env python3

import argparse
import json
from datetime import date

import mlx.core as mx
from mlx_lm import load, stream_generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MLX-LM decode benchmark over explicit prompt token IDs."
    )
    parser.add_argument("--model", required=True, help="Local snapshot path or Hugging Face repo id.")
    parser.add_argument(
        "--prompt-token-ids-json",
        required=True,
        help="JSON array of integer prompt token IDs.",
    )
    parser.add_argument("--max-tokens", required=True, type=int, help="Maximum number of tokens to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_token_ids = json.loads(args.prompt_token_ids_json)
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token, int) for token in prompt_token_ids
    ):
        raise ValueError("--prompt-token-ids-json must decode to a JSON array of integers.")

    model, tokenizer = load(args.model)
    final_response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt_token_ids,
        max_tokens=args.max_tokens,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    ):
        final_response = response

    if final_response is None:
        raise RuntimeError("MLX-LM benchmark produced no response.")

    print(
        json.dumps(
            {
                "prompt_tps": final_response.prompt_tps,
                "generation_tps": final_response.generation_tps,
                "peak_memory_gb": final_response.peak_memory,
                "captured_at": date.today().isoformat(),
                "finish_reason": final_response.finish_reason,
                "generation_tokens": final_response.generation_tokens,
            }
        )
    )


if __name__ == "__main__":
    main()
