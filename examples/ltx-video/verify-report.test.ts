import { describe, expect, test } from "bun:test";
import { array, MxArray } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";

import { writeImageProofBmp } from "../image-proof/artifact";
import { writeLtx2AudioWav } from "./audio-output";
import {
  formatUsage,
  type LtxVideoProofReport,
  loadLtxVideoProofReport,
  parseVerifyCommand,
  runLtxVideoVerifyCommand,
  verifyLtxVideoProofReport,
} from "./verify-report";

function temporaryDirectory(): string {
  return mkdtempSync(join(import.meta.dir, ".tmp-verify-"));
}

function writePreview(directory: string) {
  using image = array(
    [
      [
        [
          [0, 0, 1],
          [1, 0, 0],
        ],
      ],
    ],
    "float32",
  );
  return writeImageProofBmp(image, join(directory, "preview.bmp"), {
    label: "LTX-Video test preview",
  });
}

function writeAudio(directory: string) {
  using waveform = MxArray.fromData([0, 0.5, -0.5, 1], [1, 1, 4], "float32");
  return writeLtx2AudioWav(waveform, join(directory, "audio.wav"), 16000);
}

function reportFor(
  pipeline: LtxVideoProofReport["pipeline"],
  artifact: ReturnType<typeof writePreview>,
  audioArtifact?: ReturnType<typeof writeAudio>,
): LtxVideoProofReport {
  const dtype: LtxVideoProofReport["dtype"] = "float32";
  const common = {
    source: "/models/ltx",
    snapshotPath: "/resolved/ltx",
    resolvedRevision: "abc123",
    pipeline,
    prompt: "a quiet library",
    negativePrompt: "",
    outputPath: artifact.path,
    imageSize: { width: artifact.width, height: artifact.height },
    videoSize: { width: 2, height: 1, frames: 1, channels: 3 },
    latentSize: { width: 1, height: 1, frames: 1, channels: 128 },
    outputBytes: artifact.bytes,
    artifact,
    steps: 2,
    guidanceScale: 3,
    maxSequenceLength: 128,
    requestedFrames: 1,
    decodedFrames: 1,
    frameRate: 25,
    seed: 11,
    dtype,
    promptTruncated: false,
    negativePromptTruncated: false,
    elapsedMs: 12.34,
  };
  if (pipeline === "ltx2" && audioArtifact !== undefined) {
    return {
      ...common,
      pipeline,
      audioOutputPath: audioArtifact.path,
      audioSize: {
        sampleRate: audioArtifact.sampleRate,
        channels: audioArtifact.channels,
        samples: audioArtifact.samples,
        durationSeconds: audioArtifact.durationSeconds,
      },
      audioOutputBytes: audioArtifact.bytes,
      audioArtifact,
      audioGuidanceScale: 2.5,
    };
  }
  return { ...common, pipeline };
}

function writeReport(path: string, report: LtxVideoProofReport): void {
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);
}

describe("LTX video proof report verifier", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses verifier expectations", () => {
    const command = parseVerifyCommand([
      ".tmp/report.json",
      "--expect-pipeline",
      "ltx2",
      "--expect-source",
      "Lightricks/LTX-2",
      "--expect-resolved-revision",
      "abc123",
      "--expect-preview-sha256",
      "a".repeat(64),
      "--expect-audio-sha256",
      "b".repeat(64),
    ]);

    expect(command).toEqual({
      kind: "run",
      options: {
        reportPath: ".tmp/report.json",
        expectPipeline: "ltx2",
        expectSource: "Lightricks/LTX-2",
        expectResolvedRevision: "abc123",
        expectPreviewSha256: "a".repeat(64),
        expectAudioSha256: "b".repeat(64),
      },
    });
  });

  test("verifies classic LTX preview reports and LTX-2 preview plus audio reports", () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writePreview(directory);
      const audioArtifact = writeAudio(directory);
      const classic = verifyLtxVideoProofReport(reportFor("ltx-video", artifact), {
        reportPath: join(directory, "classic.json"),
        expectPipeline: "ltx-video",
        expectPreviewSha256: artifact.sha256,
      });
      const ltx2 = verifyLtxVideoProofReport(reportFor("ltx2", artifact, audioArtifact), {
        reportPath: join(directory, "ltx2.json"),
        expectPipeline: "ltx2",
        expectPreviewSha256: artifact.sha256,
        expectAudioSha256: audioArtifact.sha256,
      });

      expect(classic.status).toBe("passed");
      expect(classic.failedChecks).toBe(0);
      expect(ltx2.status).toBe("passed");
      expect(ltx2.failedChecks).toBe(0);
      expect(ltx2.audioSha256).toBe(audioArtifact.sha256);
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("loads report JSON and verifies through the CLI", async () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writePreview(directory);
      const audioArtifact = writeAudio(directory);
      const reportPath = join(directory, "report.json");
      writeReport(reportPath, reportFor("ltx2", artifact, audioArtifact));
      const stdout: string[] = [];

      const exitCode = await runLtxVideoVerifyCommand([reportPath, "--expect-pipeline", "ltx2"], {
        stdout: (line) => stdout.push(line),
      });

      expect(exitCode).toBe(0);
      expect(stdout.join("\n")).toContain("ltx_video_report:");
      expect(stdout.join("\n")).toContain('status: "passed"');
      expect(loadLtxVideoProofReport(reportPath).pipeline).toBe("ltx2");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("returns usage errors on stdout", async () => {
    const stdout: string[] = [];

    const exitCode = await runLtxVideoVerifyCommand(["--expect-pipeline", "ltx2"], {
      stdout: (line) => stdout.push(line),
    });

    expect(exitCode).toBe(2);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("Missing report JSON path");
  });

  test("fails expectation mismatches without rerunning generation", () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writePreview(directory);
      const result = verifyLtxVideoProofReport(reportFor("ltx-video", artifact), {
        reportPath: join(directory, "report.json"),
        expectPipeline: "ltx2",
      });

      expect(result.status).toBe("failed");
      expect(result.failedCheckNames).toContain("expect.pipeline");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("fails truncated LTX-2 WAV artifacts", async () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writePreview(directory);
      const audioArtifact = writeAudio(directory);
      const reportPath = join(directory, "report.json");
      writeReport(reportPath, reportFor("ltx2", artifact, audioArtifact));
      writeFileSync(audioArtifact.path, new Uint8Array([82, 73, 70]));
      const stdout: string[] = [];

      const exitCode = await runLtxVideoVerifyCommand([reportPath], {
        stdout: (line) => stdout.push(line),
      });

      expect(exitCode).toBe(1);
      expect(stdout.join("\n")).toContain('status: "failed"');
      expect(stdout.join("\n")).toContain("audio_artifact.byteLengthMatches");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("fails schema errors with actionable output", async () => {
    const directory = temporaryDirectory();
    try {
      const reportPath = join(directory, "bad.json");
      writeFileSync(reportPath, JSON.stringify({ pipeline: "ltx2" }));
      const stdout: string[] = [];

      const exitCode = await runLtxVideoVerifyCommand([reportPath], {
        stdout: (line) => stdout.push(line),
      });

      expect(exitCode).toBe(1);
      expect(stdout.join("\n")).toContain('code: "validation"');
      expect(stdout.join("\n")).toContain("audioArtifact is required for ltx2 reports");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });
});
