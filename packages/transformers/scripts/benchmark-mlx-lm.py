#!/usr/bin/env python3

import argparse
import glob
import json
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import stream_generate
from mlx_lm.utils import _download, _get_classes, load_tokenizer
from mlx.utils import tree_flatten


def is_allowed_extra_weight(name: str) -> bool:
    return (
        name.startswith("language_model.model.layers.")
        and (
            name.endswith(".self_attn.k_norm.weight")
            or name.endswith(".self_attn.k_proj.weight")
            or name.endswith(".self_attn.v_proj.weight")
        )
    )


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
    parser.add_argument("--prefill-step-size", required=True, type=int, help="Prompt prefill chunk size.")
    parser.add_argument("--trials", required=True, type=int, help="Measured trials to average.")
    parser.add_argument("--warmup-trials", required=True, type=int, help="Warmup trials before timing.")
    parser.add_argument(
        "--allow-extra-weights",
        action="store_true",
        help="Filter known extra checkpoint tensors before strict MLX-LM reference loading.",
    )
    return parser.parse_args()


def validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def load_reference_model(
    model_path: Path,
    allow_extra: Callable[[str], bool] | None,
) -> tuple[nn.Module, dict[str, Any]]:
    config = json.loads((model_path / "config.json").read_text())
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for weight_file in weight_files:
        weights.update(mx.load(weight_file))

    model_class, model_args_class = _get_classes(config)
    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]

    model = model_class(model_args_class.from_dict(config))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if allow_extra is not None:
        expected_names = {name for name, _ in tree_flatten(model.parameters())}
        weights = {
            name: value
            for name, value in weights.items()
            if name in expected_names or not allow_extra(name)
        }

    quantization = config.get("quantization")
    if quantization is not None:
        def class_predicate(path, module):
            if path in config["quantization"]:
                return config["quantization"][path]
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    model.eval()
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, config


def run_once(model, tokenizer, prompt_token_ids, args):
    mx.reset_peak_memory()
    final_response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt_token_ids,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    ):
        final_response = response

    if final_response is None:
        raise RuntimeError("MLX-LM benchmark produced no response.")
    return final_response


def average_responses(responses, key: str) -> float:
    return fmean(getattr(response, key) for response in responses)


def main() -> None:
    args = parse_args()
    validate_positive(args.max_tokens, "--max-tokens")
    validate_positive(args.prefill_step_size, "--prefill-step-size")
    validate_positive(args.trials, "--trials")
    if args.warmup_trials < 0:
        raise ValueError("--warmup-trials must be non-negative.")

    prompt_token_ids = json.loads(args.prompt_token_ids_json)
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token, int) for token in prompt_token_ids
    ):
        raise ValueError("--prompt-token-ids-json must decode to a JSON array of integers.")

    model_path = Path(_download(args.model))
    model, config = load_reference_model(
        model_path,
        is_allowed_extra_weight if args.allow_extra_weights else None,
    )
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    # Match mlx-lm's own benchmark behavior: throughput timing should run the
    # full requested decode window instead of stopping on EOS.
    tokenizer._eos_token_ids = {}

    for _ in range(args.warmup_trials):
        run_once(model, tokenizer, prompt_token_ids, args)
        mx.clear_cache()

    responses = []
    for _ in range(args.trials):
        responses.append(run_once(model, tokenizer, prompt_token_ids, args))
        mx.clear_cache()

    generation_tokens = responses[-1].generation_tokens
    finish_reason = responses[-1].finish_reason
    if any(response.generation_tokens != generation_tokens for response in responses):
        raise RuntimeError("MLX-LM benchmark measured trials with different generation token counts.")
    if any(response.finish_reason != finish_reason for response in responses):
        raise RuntimeError("MLX-LM benchmark measured trials with different finish reasons.")

    print(
        json.dumps(
            {
                "prompt_tps": average_responses(responses, "prompt_tps"),
                "generation_tps": average_responses(responses, "generation_tps"),
                "peak_memory_gb": average_responses(responses, "peak_memory"),
                "captured_at": date.today().isoformat(),
                "finish_reason": finish_reason,
                "generation_tokens": generation_tokens,
                "trial_count": len(responses),
            }
        )
    )


if __name__ == "__main__":
    main()
