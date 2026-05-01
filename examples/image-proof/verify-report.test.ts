import { describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";

import { writeImageProofBmp } from "./artifact";
import {
  formatUsage,
  type ImageProofExampleReport,
  loadImageProofReport,
  parseVerifyCommand,
  runImageProofVerifyCommand,
  verifyImageProofReport,
} from "./verify-report";

function temporaryDirectory(): string {
  return mkdtempSync(join(import.meta.dir, ".tmp-verify-"));
}

function writeArtifact(directory: string) {
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
  return writeImageProofBmp(image, join(directory, "sample.bmp"), { label: "test image" });
}

function reportFor(
  pipeline: ImageProofExampleReport["pipeline"],
  artifact: ReturnType<typeof writeImageProofBmp>,
): ImageProofExampleReport {
  const dtype: ImageProofExampleReport["dtype"] = "float32";
  const common = {
    source: "/models/image",
    snapshotPath: "/resolved/image",
    resolvedRevision: "abc123",
    pipeline,
    prompt: "a red apple",
    outputPath: artifact.path,
    imageSize: { width: artifact.width, height: artifact.height },
    outputBytes: artifact.bytes,
    steps: 4,
    seed: 11,
    dtype,
    promptTruncated: false,
    elapsedMs: 12.34,
    artifact,
  };
  if (pipeline === "stable-diffusion" || pipeline === "stable-diffusion-xl") {
    return {
      ...common,
      guidanceScale: 5,
      negativePrompt: "",
      negativePromptTruncated: false,
    };
  }
  if (pipeline === "flux") {
    return {
      ...common,
      guidanceScale: null,
      maxSequenceLength: 512,
      prompt2: "a red apple",
      prompt2Truncated: false,
    };
  }
  if (pipeline === "flux2-klein") {
    return {
      ...common,
      guidanceScale: 4,
      negativePrompt: "",
      negativePromptTruncated: false,
      maxSequenceLength: 512,
    };
  }
  if (pipeline === "z-image") {
    return {
      ...common,
      guidanceScale: 0,
      maxSequenceLength: 512,
    };
  }
  return {
    ...common,
    trueCfgScale: 4,
    negativePrompt: " ",
    negativePromptTruncated: false,
    maxSequenceLength: 1024,
  };
}

function writeReport(path: string, report: ImageProofExampleReport): void {
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);
}

describe("image proof report verifier", () => {
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
      "qwen-image",
      "--expect-source",
      "Qwen/Qwen-Image-2512",
      "--expect-resolved-revision",
      "abc123",
      "--expect-sha256",
      "a".repeat(64),
    ]);

    expect(command).toEqual({
      kind: "run",
      options: {
        reportPath: ".tmp/report.json",
        expectPipeline: "qwen-image",
        expectSource: "Qwen/Qwen-Image-2512",
        expectResolvedRevision: "abc123",
        expectSha256: "a".repeat(64),
      },
    });
  });

  test("verifies all current image proof report families", () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writeArtifact(directory);
      const pipelines: ImageProofExampleReport["pipeline"][] = [
        "stable-diffusion",
        "stable-diffusion-xl",
        "flux",
        "flux2-klein",
        "z-image",
        "qwen-image",
      ];

      for (const pipeline of pipelines) {
        const report = reportFor(pipeline, artifact);
        const result = verifyImageProofReport(report, {
          reportPath: join(directory, `${pipeline}.json`),
          expectPipeline: pipeline,
          expectResolvedRevision: "abc123",
          expectSha256: artifact.sha256,
        });

        expect(result.status).toBe("passed");
        expect(result.failedChecks).toBe(0);
      }
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("loads report JSON and verifies through the CLI", async () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writeArtifact(directory);
      const reportPath = join(directory, "report.json");
      writeReport(reportPath, reportFor("qwen-image", artifact));
      const stdout: string[] = [];

      const exitCode = await runImageProofVerifyCommand(
        [reportPath, "--expect-pipeline", "qwen-image"],
        { stdout: (line) => stdout.push(line) },
      );

      expect(exitCode).toBe(0);
      expect(stdout.join("\n")).toContain("image_proof_report:");
      expect(stdout.join("\n")).toContain('status: "passed"');
      expect(loadImageProofReport(reportPath).pipeline).toBe("qwen-image");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("returns usage errors on stdout", async () => {
    const stdout: string[] = [];

    const exitCode = await runImageProofVerifyCommand(["--expect-pipeline", "flux"], {
      stdout: (line) => stdout.push(line),
    });

    expect(exitCode).toBe(2);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("Missing report JSON path");
  });

  test("fails expectation mismatches without rerunning generation", () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writeArtifact(directory);
      const result = verifyImageProofReport(reportFor("flux", artifact), {
        reportPath: join(directory, "report.json"),
        expectPipeline: "qwen-image",
      });

      expect(result.status).toBe("failed");
      expect(result.failedCheckNames).toContain("expect.pipeline");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("fails missing, bad-header, and truncated artifacts", async () => {
    const directory = temporaryDirectory();
    try {
      const artifact = writeArtifact(directory);
      const reportPath = join(directory, "report.json");
      writeReport(reportPath, reportFor("stable-diffusion", artifact));
      writeFileSync(artifact.path, new Uint8Array([66, 77, 0]));
      const stdout: string[] = [];

      const exitCode = await runImageProofVerifyCommand([reportPath], {
        stdout: (line) => stdout.push(line),
      });

      expect(exitCode).toBe(1);
      expect(stdout.join("\n")).toContain('status: "failed"');
      expect(stdout.join("\n")).toContain("artifact.byteLengthMatches");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });

  test("fails schema errors with actionable output", async () => {
    const directory = temporaryDirectory();
    try {
      const reportPath = join(directory, "bad.json");
      writeFileSync(reportPath, JSON.stringify({ pipeline: "flux" }));
      const stdout: string[] = [];

      const exitCode = await runImageProofVerifyCommand([reportPath], {
        stdout: (line) => stdout.push(line),
      });

      expect(exitCode).toBe(1);
      expect(stdout.join("\n")).toContain('code: "validation"');
      expect(stdout.join("\n")).toContain("artifact must be an object");
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
  });
});
