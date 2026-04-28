# examples/nanogpt

`examples/nanogpt/` is the committed GPT proof surface. It is not a publishable package.

`src/run/` is production code for the supervised GPT run path. Manager, supervisor, file, and status plumbing use `@mlxts/train/supervised-run`; nanoGPT owns only the GPT trainer command, `.nanogpt-runs` root, supervisor lock label, status wording, GPT config typing, acceptance thresholds, soak defaults, and sample generation.

`RunStatus.config` is GPT-shaped only in this example boundary. The reusable supervised-run contract remains model-family agnostic.

Snapshot checkpoints and resume checkpoints keep distinct meanings. Acceptance and soak flows treat resume checkpoints as exact-continuation state and snapshots as lightweight model saves.

Reusable training or run-management abstractions move into `@mlxts/*`. The example keeps GPT-specific policy and operator commands.
