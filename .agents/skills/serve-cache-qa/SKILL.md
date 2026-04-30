---
name: serve-cache-qa
description: Diagnose and validate @mlxts/serve prompt-prefix cache behavior across Pi-style agent sessions, Gemma/Qwen dense and MoE checkpoints, exact-boundary caches, and multi-model serving.
---

# Serve Cache QA

Use this skill when changing or debugging `@mlxts/serve` prompt-prefix caching,
continuous scheduling, Pi/OpenAI-compatible agent loops, or Qwen/Gemma serving
regressions.

## Core Model

Prompt-prefix cache hits are completed retained snapshots. Cold concurrent
requests cannot reuse an in-flight snapshot. Exact-boundary caches such as Qwen
hybrid and Gemma layer-pattern caches can reuse only retained exact prompt
boundaries unless the family snapshot explicitly supports shorter forks.

Serve owns matching, retention, eviction, event accounting, and protocol usage.
Transformers owns cache state shape, `layerKinds`, snapshot/fork validity, and
disposal.

## Debug Ladder

1. Read `packages/serve/AGENTS.md`, `docs/runtime-safety.md`, and the serving
   entries in `MEMORY.md`.
2. Capture route and cache events before changing code:
   - `[route] ... route=continuous|single ... model_type=...`
   - `[cache] ... miss|hit|write ... read_tokens=... write_tokens=...`
   - `generation_scheduler_phase` queued, prefill, admitted, first-token, and
     finished phases.
3. Separate cold-concurrent misses from warm-retention failures:
   - cold A+B simultaneous: misses are expected when no snapshot exists yet
   - warm exact A repeat: must hit
   - warm A/B/A with divergent sessions: must hit when retention keeps both
     exact prompt boundaries
4. Check protocol shape. Prompt-prefix caching applies to message/content
   prompts with compatible prompt-cache identity; plain text completions are not
   the Pi chat-loop proof path.
5. Check cache semantics through `CacheLayerKind`, not family names:
   - full KV can usually trim/prefix-fork
   - Gemma layer-pattern full+sliding is exact-boundary unless the snapshot says
     otherwise
   - Qwen hybrid full+linear-recurrent is exact-boundary unless the family-owned
     cache says otherwise

## Required Coverage

For a cache remediation, add the narrowest unit test that reproduces the product
failure without real checkpoints, then run the focused package tests.

Minimum unit coverage:

- exact-boundary A/B/A retention with Gemma-style `["full", "sliding"]`
  `CacheLayerKind`
- Qwen-style exact continuation with `["linear-recurrent", "full"]`
- routing tests that prove cache-shape decisions use `CacheLayerKind`
- multi-model router evidence that model engines keep separate cache state

Real checkpoint proof, when cached models and runtime budget are available:

- dense Gemma and Gemma MoE
- dense Qwen and Qwen MoE
- one Gemma model and one Qwen model served by the same process
- repeated Pi-style chat turn for one session, then two divergent sessions, then
  exact A/B/A replay

Use `bun run regression:agent-cache -- --scenarios qwen-dense,gemma-dense,multi-dense`
for the automated dense proof. Add `--include-moe` when the Qwen/Gemma MoE
checkpoints and memory budget are available.

Use `cmux` for Pi/server smokes so server logs and two client terminals remain
visible. Heavy MLX commands remain exclusive.

## Evidence

Runtime-sensitive cache changes need a `docs/reviews/` artifact with:

- `## Files Reviewed` naming every changed runtime-sensitive file
- what was reproduced or ruled out
- focused tests and real checkpoint/CMux evidence
- memory or performance tradeoff, especially retained snapshot count/bytes
- independent review result
- remaining risks and explicit out-of-scope work

Do not claim shared AGENTS-prefix reuse for exact-boundary caches unless there is
an exact retained snapshot at that boundary or a family-owned cache backend that
supports that fork.
