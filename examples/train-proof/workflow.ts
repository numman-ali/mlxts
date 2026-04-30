import type { LoadSourceOptions } from "@mlxts/transformers";
import { mkdirSync } from "fs";
import { dirname } from "path";

import type { TrainingProofArgs } from "./args";
import { prepareTrainingProofData } from "./prepare";
import { loadAssets } from "./runtime";
import { printStage, runDPOStage, runLoRAStage, runQLoRAStage, runSFTStage } from "./stages";
import type { StageReport, TrainingProofReport } from "./types";
import { assertTrainingProofReport, verificationOptionsFromArgs } from "./verification";

export async function runTrainingProof(
  parsed: TrainingProofArgs,
  progress: (line: string) => void = console.error,
): Promise<TrainingProofReport> {
  const sourceOptions: LoadSourceOptions = {};

  progress(`Training proof source: ${parsed.source}`);
  progress(`Dataset source: ${parsed.datasetSource}`);
  progress(`Train limit: ${parsed.trainLimit}`);
  progress(`Eval limit: ${parsed.evalLimit}`);
  progress(`Batch size: ${parsed.batchSize}`);
  progress(`Steps per stage: ${parsed.steps}`);
  progress(`Max sequence length: ${parsed.maxSequenceLength}`);
  progress(`Stages: ${parsed.stages.join(",")}`);
  progress(`DPO profile: ${parsed.dpoProfile}`);
  progress(`Quantized snapshot output: ${parsed.quantizedOutputDir}`);
  progress(`Adapter output: ${parsed.adapterOutputDir}`);
  progress(`Report path: ${parsed.reportPath}`);

  const denseAssets = await loadAssets(parsed.source, sourceOptions);
  const preparedData = await prepareTrainingProofData(
    denseAssets.tokenizer,
    denseAssets.profile,
    parsed,
  );
  for (const note of preparedData.notes) {
    progress(`  data: ${note}`);
  }

  const stages: StageReport[] = [];
  if (parsed.stages.includes("lora")) {
    const loraStage = await runLoRAStage(
      parsed.source,
      denseAssets.tokenizer,
      denseAssets.profile,
      preparedData,
      parsed,
    );
    printStage(loraStage, progress);
    stages.push(loraStage);
  }
  if (parsed.stages.includes("qlora")) {
    const qloraStage = await runQLoRAStage(
      parsed.source,
      parsed.quantizedOutputDir,
      denseAssets.tokenizer,
      denseAssets.profile,
      preparedData,
      parsed,
    );
    printStage(qloraStage, progress);
    stages.push(qloraStage);
  }
  if (parsed.stages.includes("sft")) {
    const sftStage = await runSFTStage(
      parsed.source,
      denseAssets.tokenizer,
      denseAssets.profile,
      preparedData,
      parsed,
    );
    printStage(sftStage, progress);
    stages.push(sftStage);
  }
  if (parsed.stages.includes("dpo")) {
    const dpoStage = await runDPOStage(
      parsed.source,
      denseAssets.tokenizer,
      denseAssets.profile,
      preparedData,
      parsed,
    );
    printStage(dpoStage, progress);
    stages.push(dpoStage);
  }

  const report: TrainingProofReport = {
    source: parsed.source,
    quantizedOutputDir: parsed.quantizedOutputDir,
    adapterOutputDir: parsed.adapterOutputDir,
    datasetSource: parsed.datasetSource,
    trainLimit: parsed.trainLimit,
    evalLimit: parsed.evalLimit,
    batchSize: parsed.batchSize,
    steps: parsed.steps,
    maxSequenceLength: parsed.maxSequenceLength,
    seed: parsed.seed,
    dataNotes: preparedData.notes,
    stages,
  };
  const verification = assertTrainingProofReport(report, verificationOptionsFromArgs(parsed));
  const verifiedReport: TrainingProofReport = {
    ...report,
    verification,
  };

  mkdirSync(dirname(parsed.reportPath), { recursive: true });
  await Bun.write(parsed.reportPath, `${JSON.stringify(verifiedReport, null, 2)}\n`);
  progress(`Report written to ${parsed.reportPath}`);
  progress(`Report verification passed (${verification.checks.length} checks).`);
  return verifiedReport;
}
