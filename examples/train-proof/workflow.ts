import type { LoadSourceOptions } from "@mlxts/transformers";
import { mkdirSync } from "fs";
import { dirname } from "path";

import { parseTrainingProofArgs, type TrainingProofArgs } from "./args";
import { prepareTrainingProofData } from "./prepare";
import { loadAssets } from "./runtime";
import { printStage, runDPOStage, runLoRAStage, runQLoRAStage, runSFTStage } from "./stages";
import type { StageReport, TrainingProofReport } from "./types";
import { assertTrainingProofReport, verificationOptionsFromArgs } from "./verification";

export async function runTrainingProof(argv: readonly string[]): Promise<void> {
  const parsed: TrainingProofArgs = parseTrainingProofArgs(argv);
  const sourceOptions: LoadSourceOptions = {};

  console.log(`Training proof source: ${parsed.source}`);
  console.log(`Dataset source: ${parsed.datasetSource}`);
  console.log(`Train limit: ${parsed.trainLimit}`);
  console.log(`Eval limit: ${parsed.evalLimit}`);
  console.log(`Batch size: ${parsed.batchSize}`);
  console.log(`Steps per stage: ${parsed.steps}`);
  console.log(`Max sequence length: ${parsed.maxSequenceLength}`);
  console.log(`Stages: ${parsed.stages.join(",")}`);
  console.log(`DPO profile: ${parsed.dpoProfile}`);
  console.log(`Quantized snapshot output: ${parsed.quantizedOutputDir}`);
  console.log(`Report path: ${parsed.reportPath}`);

  const denseAssets = await loadAssets(parsed.source, sourceOptions);
  const preparedData = await prepareTrainingProofData(
    denseAssets.tokenizer,
    denseAssets.profile,
    parsed,
  );
  for (const note of preparedData.notes) {
    console.log(`  data: ${note}`);
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
    printStage(loraStage);
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
    printStage(qloraStage);
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
    printStage(sftStage);
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
    printStage(dpoStage);
    stages.push(dpoStage);
  }

  const report: TrainingProofReport = {
    source: parsed.source,
    quantizedOutputDir: parsed.quantizedOutputDir,
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
  console.log(`Report written to ${parsed.reportPath}`);
  console.log(`Report verification passed (${verification.checks.length} checks).`);
}
