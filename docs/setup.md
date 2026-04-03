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
cd packages/mlx-ts
bun run build:native
```

This does the following automatically:
1. Runs CMake to fetch **mlx-c** v0.6.0 (Apple's C API for MLX)
2. mlx-c's CMake fetches **MLX** (Apple's ML framework) via FetchContent
3. Compiles both into shared libraries (`libmlxc.dylib`, `libmlx.dylib`)
4. Copies them to `packages/mlx-ts/native/lib/`
5. Fixes rpaths so the libraries can find each other at runtime

**First build takes 5-15 minutes** (downloads + compiles C++). Subsequent builds are cached — if `native/lib/` already contains the dylibs, the script skips the build.

To force a rebuild:
```bash
rm -rf packages/mlx-ts/native/build packages/mlx-ts/native/lib
bun run build:native
```

### 3. Verify

```bash
bun run validate
```

This runs typecheck + lint + tests across all packages.

## Build details

### What gets built

The native build produces two shared libraries:

| Library | Source | Purpose |
|---|---|---|
| `libmlx.dylib` | [ml-explore/mlx](https://github.com/ml-explore/mlx) | MLX core — Metal GPU kernels, tensor ops, autograd |
| `libmlxc.dylib` | [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c) | C API wrapper over MLX C++ — what our FFI binds to |

Both live in `packages/mlx-ts/native/lib/` (gitignored).

### Why the Xcode SDK matters

The build script explicitly points CMake at the Xcode SDK (`-DCMAKE_OSX_SYSROOT`), not the Command Line Tools SDK. This is because the Metal compiler (`metal`) is only available through Xcode, and CMake needs it to compile MLX's `.metal` GPU shader files into a `.metallib` bundle.

If you see SDK-related errors, ensure Xcode is installed (not just Command Line Tools) and the Metal Toolchain is downloaded.

### Version pinning

| Dependency | Pinned Version | Controlled By |
|---|---|---|
| mlx-c | v0.6.0 | `native/CMakeLists.txt` `GIT_TAG` |
| MLX | v0.31.1 (approx) | mlx-c's own CMakeLists.txt (FetchContent) |
| Bun | 1.3+ | `package.json` engine field (future) |

To update mlx-c, change the `GIT_TAG` in `packages/mlx-ts/native/CMakeLists.txt` and force a rebuild.
