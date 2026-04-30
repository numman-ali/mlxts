import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../errors";
import { DDIMScheduler } from "../schedulers/ddim";
import { EulerScheduler } from "../schedulers/euler";
import { FlowMatchEulerScheduler } from "../schedulers/flow-match-euler";

import {
  createDiffusionScheduler,
  loadDiffusionSchedulerConfig,
  loadDiffusionSchedulerFromSnapshot,
  parseDiffusionSchedulerConfig,
} from "./scheduler-config";

async function withTempDirectory<T>(
  prefix: string,
  run: (directory: string) => Promise<T>,
): Promise<T> {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  try {
    return await run(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function writeSchedulerConfig(directory: string, payload: unknown): void {
  const schedulerDirectory = join(directory, "scheduler");
  mkdirSync(schedulerDirectory, { recursive: true });
  writeFileSync(join(schedulerDirectory, "scheduler_config.json"), JSON.stringify(payload));
}

function writeRootSchedulerConfig(directory: string, payload: unknown): void {
  writeFileSync(join(directory, "scheduler_config.json"), JSON.stringify(payload));
}

describe("diffusion scheduler config loading", () => {
  test("parses DDIM scheduler config from Diffusers-style metadata", () => {
    const parsed = parseDiffusionSchedulerConfig({
      _class_name: "DDIMScheduler",
      beta_start: 0.00085,
      beta_end: 0.012,
      beta_schedule: "scaled_linear",
      num_train_timesteps: 1000,
      set_alpha_to_one: false,
      clip_sample: false,
      clip_sample_range: 2,
      timestep_spacing: "trailing",
      steps_offset: 1,
      prediction_type: "epsilon",
    });

    expect(parsed).toEqual({
      kind: "ddim",
      className: "DDIMScheduler",
      config: {
        betaStart: 0.00085,
        betaEnd: 0.012,
        betaSchedule: "scaled_linear",
        numTrainTimesteps: 1000,
        setAlphaToOne: false,
        clipSample: false,
        clipSampleRange: 2,
        timestepSpacing: "trailing",
        stepsOffset: 1,
      },
    });
    expect(createDiffusionScheduler(parsed)).toBeInstanceOf(DDIMScheduler);
  });

  test("parses Euler scheduler config only when unsupported Diffusers knobs are inactive", () => {
    const parsed = parseDiffusionSchedulerConfig({
      _class_name: "EulerDiscreteScheduler",
      beta_start: 0.00085,
      beta_end: 0.012,
      beta_schedule: "scaled_linear",
      num_train_timesteps: 1000,
      prediction_type: "epsilon",
      interpolation_type: "linear",
      timestep_spacing: "linspace",
      timestep_type: "discrete",
      final_sigmas_type: "zero",
      use_karras_sigmas: false,
      use_exponential_sigmas: false,
      use_beta_sigmas: false,
      steps_offset: 0,
    });

    expect(parsed).toEqual({
      kind: "euler",
      className: "EulerDiscreteScheduler",
      config: {
        betaStart: 0.00085,
        betaEnd: 0.012,
        betaSchedule: "scaled_linear",
        numTrainTimesteps: 1000,
      },
    });
    expect(createDiffusionScheduler(parsed)).toBeInstanceOf(EulerScheduler);
  });

  test("parses FlowMatch Euler scheduler config for Flux snapshots", () => {
    const parsed = parseDiffusionSchedulerConfig({
      _class_name: "FlowMatchEulerDiscreteScheduler",
      _diffusers_version: "0.34.0.dev0",
      base_image_seq_len: 256,
      base_shift: 0.5,
      invert_sigmas: false,
      max_image_seq_len: 4096,
      max_shift: 1.15,
      num_train_timesteps: 1000,
      shift: 3,
      shift_terminal: null,
      stochastic_sampling: false,
      time_shift_type: "exponential",
      use_beta_sigmas: false,
      use_dynamic_shifting: true,
      use_exponential_sigmas: false,
      use_karras_sigmas: false,
    });

    expect(parsed).toEqual({
      kind: "flow-match-euler",
      className: "FlowMatchEulerDiscreteScheduler",
      config: {
        baseImageSeqLen: 256,
        baseShift: 0.5,
        maxImageSeqLen: 4096,
        maxShift: 1.15,
        numTrainTimesteps: 1000,
        shift: 3,
        timeShiftType: "exponential",
        useDynamicShifting: true,
      },
    });
    expect(createDiffusionScheduler(parsed)).toBeInstanceOf(FlowMatchEulerScheduler);
  });

  test("loads scheduler config and scheduler instances from a local snapshot", async () => {
    await withTempDirectory("mlxts-diffusion-scheduler-", async (directory) => {
      writeSchedulerConfig(directory, {
        _class_name: "DDIMScheduler",
        beta_schedule: "linear",
        beta_start: 0.1,
        beta_end: 0.2,
        num_train_timesteps: 4,
      });

      const parsed = await loadDiffusionSchedulerConfig(directory);
      const loaded = await loadDiffusionSchedulerFromSnapshot(directory);

      expect(parsed.kind).toBe("ddim");
      expect(loaded.scheduler).toBeInstanceOf(DDIMScheduler);
      expect(loaded.className).toBe("DDIMScheduler");
      expect(loaded.configPath).toBe(join(directory, "scheduler", "scheduler_config.json"));
      expect(loaded.rawConfig._class_name).toBe("DDIMScheduler");
    });
  });

  test("loads scheduler config from an explicit Diffusers subfolder", async () => {
    await withTempDirectory("mlxts-diffusion-scheduler-subfolder-", async (directory) => {
      const custom = join(directory, "custom-scheduler");
      mkdirSync(custom, { recursive: true });
      writeFileSync(
        join(custom, "scheduler_config.json"),
        JSON.stringify({
          _class_name: "EulerDiscreteScheduler",
          beta_schedule: "linear",
          beta_start: 0.1,
          beta_end: 0.2,
          num_train_timesteps: 4,
        }),
      );

      const loaded = await loadDiffusionSchedulerFromSnapshot(directory, {
        subfolder: "custom-scheduler",
      });

      expect(loaded.scheduler).toBeInstanceOf(EulerScheduler);
    });
  });

  test("loads root-level scheduler config when the subfolder is disabled", async () => {
    await withTempDirectory("mlxts-diffusion-root-scheduler-", async (directory) => {
      writeRootSchedulerConfig(directory, {
        _class_name: "DDIMScheduler",
        beta_schedule: "linear",
        beta_start: 0.1,
        beta_end: 0.2,
        num_train_timesteps: 4,
      });

      const loaded = await loadDiffusionSchedulerFromSnapshot(directory, { subfolder: "" });

      expect(loaded.configPath).toBe(join(directory, "scheduler_config.json"));
      expect(loaded.parsedConfig.kind).toBe("ddim");
    });
  });

  test("rejects configs whose scheduler math is not implemented yet", () => {
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        prediction_type: "v_prediction",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        beta_schedule: "squaredcos_cap_v2",
      }),
    ).toThrow("not supported");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "EulerDiscreteScheduler",
        use_karras_sigmas: true,
      }),
    ).toThrow("use_karras_sigmas");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        thresholding: true,
      }),
    ).toThrow("thresholding");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        trained_betas: [0.1, 0.2],
      }),
    ).toThrow("trained_betas");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "EulerDiscreteScheduler",
        sigma_min: 0.01,
      }),
    ).toThrow("sigma_min");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "EulerDiscreteScheduler",
        steps_offset: 1,
      }),
    ).toThrow("steps_offset");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "EulerDiscreteScheduler",
        final_sigmas_type: "sigma_min",
      }),
    ).toThrow("final_sigmas_type");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "FlowMatchEulerDiscreteScheduler",
        invert_sigmas: true,
      }),
    ).toThrow("invert_sigmas");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "FlowMatchEulerDiscreteScheduler",
        shift_terminal: 0.1,
      }),
    ).toThrow("shift_terminal");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "FlowMatchEulerDiscreteScheduler",
        stochastic_sampling: true,
      }),
    ).toThrow("stochastic_sampling");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "FlowMatchEulerDiscreteScheduler",
        use_beta_sigmas: true,
      }),
    ).toThrow("use_beta_sigmas");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "FlowMatchEulerDiscreteScheduler",
        prediction_type: "epsilon",
      }),
    ).toThrow("prediction_type");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "FlowMatchEulerDiscreteScheduler",
        beta_schedule: "linear",
      }),
    ).toThrow("beta_schedule");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "FlowMatchEulerDiscreteScheduler",
        time_shift_type: "cosine",
      }),
    ).toThrow("time_shift_type");
  });

  test("rejects unknown classes, malformed fields, and missing config files", async () => {
    expect(() => parseDiffusionSchedulerConfig(null)).toThrow("JSON object");
    expect(() => parseDiffusionSchedulerConfig([])).toThrow("JSON object");
    expect(() => parseDiffusionSchedulerConfig({})).toThrow("_class_name");
    expect(() => parseDiffusionSchedulerConfig({ _class_name: "PNDMScheduler" })).toThrow(
      "not supported",
    );
    expect(() => parseDiffusionSchedulerConfig({ _class_name: "" })).toThrow("_class_name");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        num_train_timesteps: 1.5,
      }),
    ).toThrow("num_train_timesteps");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        beta_start: [0.1],
      }),
    ).toThrow("array");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        set_alpha_to_one: "false",
      }),
    ).toThrow("set_alpha_to_one");
    expect(() =>
      parseDiffusionSchedulerConfig({
        _class_name: "DDIMScheduler",
        timestep_spacing: "middle",
      }),
    ).toThrow("timestep_spacing");

    await withTempDirectory("mlxts-diffusion-missing-scheduler-", async (directory) => {
      await expect(loadDiffusionSchedulerConfig(directory)).rejects.toThrow("missing");
      await expect(
        loadDiffusionSchedulerFromSnapshot(directory, { subfolder: "../outside" }),
      ).rejects.toThrow("subfolder");
    });

    await withTempDirectory("mlxts-diffusion-invalid-json-", async (directory) => {
      const schedulerDirectory = join(directory, "scheduler");
      mkdirSync(schedulerDirectory, { recursive: true });
      writeFileSync(join(schedulerDirectory, "scheduler_config.json"), "{");

      await expect(loadDiffusionSchedulerConfig(directory)).rejects.toThrow("valid JSON");
    });
  });
});
