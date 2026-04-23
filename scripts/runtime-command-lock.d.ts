/**
 * Acquire the shared MLX runtime lock for a heavy command.
 *
 * Benchmark, soak, acceptance, and long-running training entrypoints should use
 * this so the repo cannot accidentally run multiple heavy MLX programs at the
 * same time on one machine.
 */
export declare function acquireRuntimeCommandLock(command: string): Disposable;
