# Development Setup

## Prerequisites

### Hardware
- Mac with Apple Silicon (M1/M2/M3/M4)

### Software

| Dependency | Min Version | Install | Purpose |
|---|---|---|---|
| macOS | 14.0+ | — | Apple Silicon + Metal support |
| Xcode | 16.0+ | Mac App Store | C/C++ compiler, Metal toolchain |
| Metal Toolchain | — | See below | Compiles Metal GPU shaders for MLX |
| CMake | 3.24+ | `brew install cmake` | Builds mlx-c and MLX from source |
| Bun | 1.3+ | `curl -fsSL https://bun.sh/install \| bash` | Runtime, FFI, package manager, test runner |

### Metal Toolchain

Xcode 16+ does not bundle the Metal compiler by default. MLX's GPU kernels require it. Install with:

```bash
xcodebuild -downloadComponent MetalToolchain
```

This downloads ~700MB. You only need to do this once. Without it, the native build will fail with:
```
error: cannot execute tool 'metal' due to missing Metal Toolchain
```

### Verify prerequisites

```bash
# C++ compiler (from Xcode)
c++ --version

# Metal compiler
xcrun --sdk macosx --find metal

# CMake
cmake --version

# Bun
bun --version
```

If `xcrun --find metal` fails, install the Metal Toolchain as described above.

## Building

### 1. Install dependencies

```bash
bun install
```

### 2. Build native bindings

```bash
cd packages/core
bun run build:native
```

This does the following automatically:
1. Runs CMake to fetch **mlx-c** v0.6.0 (Apple's C API for MLX)
2. mlx-c's CMake fetches **MLX** (Apple's ML framework) via FetchContent
3. Compiles both into shared libraries (`libmlxc.dylib`, `libmlx.dylib`)
4. Copies them to `packages/core/native/lib/`
5. Fixes rpaths so the libraries can find each other at runtime

**First build takes 5-15 minutes** (downloads + compiles C++). Subsequent builds are cached — if `native/lib/` already contains the dylibs, the script skips the build.

To force a rebuild:
```bash
rm -rf packages/core/native/build packages/core/native/lib
bun run build:native
```

### 3. Verify

```bash
bun run validate
```

This runs typecheck + lint + assertion checks + coverage-backed tests across all packages.
It also runs the 500-line file-size gate, tensor-lifetime checks, and runtime-review validation. The current package-first coverage posture hard-enforces `@mlxts/core` and reports the newly extracted auxiliary packages while the migration settles.

Longer acceptance runs are separate on purpose:

```bash
bun run acceptance:gpt-tiny
bun run acceptance:gpt-small
```

Before an overnight run, use the shorter runtime checks:

```bash
bun run bench:memory
bun run soak:gpt-tiny
bun run soak:gpt-small
```

For overnight or laptop-safe long runs, use the detached supervisor:

```bash
bun run run:nanogpt start --preset gpt-small --max-steps 5000
bun run run:nanogpt status --name <run-id>
bun run run:nanogpt watch --name <run-id> --interval 600
bun run run:nanogpt stop --name <run-id>
bun run run:nanogpt resume --from <run-id> --max-steps 10000
```

This supervised path is the canonical long-run surface. It writes a run-local directory under `.nanogpt-runs/` with structured events, status snapshots, stderr logs, and checkpoint directories.

For now that operator surface still lives in `packages/nanogpt` because the
current app is a temporary validation fixture. The package location is
transitional even though the operator behavior is still canonical.

Checkpoint kinds are explicit:

- `*-snapshot-step-N/` stores model/config/tokenizer for frequent saves
- `*-resume-step-N/` additionally stores optimizer state for exact continuation

## Build details

### What gets built

The native build produces two shared libraries:

| Library | Source | Purpose |
|---|---|---|
| `libmlx.dylib` | [ml-explore/mlx](https://github.com/ml-explore/mlx) | MLX core — Metal GPU kernels, tensor ops, autograd |
| `libmlxc.dylib` | [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c) | C API wrapper over MLX C++ — what our FFI binds to |

Both live in `packages/core/native/lib/` (gitignored).

### Why the Xcode SDK matters

The build script explicitly points CMake at the Xcode SDK (`-DCMAKE_OSX_SYSROOT`), not the Command Line Tools SDK. This is because the Metal compiler (`metal`) is only available through Xcode, and CMake needs it to compile MLX's `.metal` GPU shader files into a `.metallib` bundle.

If you see SDK-related errors, ensure Xcode is installed (not just Command Line Tools) and the Metal Toolchain is downloaded.

### Version pinning

| Dependency | Pinned Version | Controlled By |
|---|---|---|
| mlx-c | v0.6.0 | `native/CMakeLists.txt` `GIT_TAG` |
| MLX | v0.31.1 (approx) | mlx-c's own CMakeLists.txt (FetchContent) |
| Bun | 1.3+ | `package.json` engine field (future) |

To update mlx-c, change the `GIT_TAG` in `packages/core/native/CMakeLists.txt` and force a rebuild.
