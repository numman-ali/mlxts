# Serve Completions Example

This example is a deterministic harness for `@mlxts/serve`'s OpenAI-compatible
`/v1/completions` route.

It starts a local Bun server with:

- a model router for two model ids
- a micro-batching wrapper for nearby non-streaming requests
- four pseudo-agents making concurrent OpenAI-style completion calls

Run it from the repo root:

```bash
bun run examples/serve-completions/index.ts
```

The example intentionally does not shell out to Codex, opencode, or pi agent.
Those tools are better as later external smoke tests once this stable harness
proves the endpoint contract, batching seam, and model routing behavior.
