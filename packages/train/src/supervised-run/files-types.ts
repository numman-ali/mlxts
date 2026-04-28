export const DEFAULT_STALL_TIMEOUT_SECONDS = 600;

export type EmptyRunStatusExtras = Record<never, never>;
export type RunStatusConfig = unknown;

export type RunState =
  | "starting"
  | "running"
  | "stopping"
  | "cancelling"
  | "stalled"
  | "stopped"
  | "completed"
  | "failed"
  | "cancelled";

export type RunControlCommand = "stop" | "cancel";
export type OperatorHealth = "healthy" | "dead-supervisor" | "dead-trainer" | "dead-both";

export type RunSpec = {
  runId: string;
  createdAt: string;
  repoRoot: string;
  packageRoot: string;
  checkpointDir: string;
  stallTimeoutSeconds?: number | undefined;
  trainerArgs: string[];
  resumedFrom?: string | undefined;
};

export type RunControl = {
  command: RunControlCommand;
  requestedAt: string;
};

export type RunStatus<TExtras extends object = EmptyRunStatusExtras> = {
  runId: string;
  state: RunState;
  startedAt: string;
  updatedAt: string;
  supervisorHeartbeatAt: string;
  trainerHeartbeatAt?: string | undefined;
  lastProgressAt?: string | undefined;
  stallTimeoutSeconds?: number | undefined;
  supervisorPid?: number | undefined;
  trainerPid?: number | undefined;
  preset?: string | undefined;
  config?: RunStatusConfig | undefined;
  parameterCount?: number | undefined;
  step?: number | undefined;
  maxSteps?: number | undefined;
  batchSize?: number | undefined;
  gradAccumSteps?: number | undefined;
  warmupSteps?: number | undefined;
  lastStepLoss?: number | undefined;
  lastTrainLoss?: number | undefined;
  lastValLoss?: number | undefined;
  bestValLoss?: number | undefined;
  lastTokensPerSec?: number | undefined;
  latestCheckpoint?: string | undefined;
  latestSnapshotCheckpoint?: string | undefined;
  latestResumeCheckpoint?: string | undefined;
  bestCheckpoint?: string | undefined;
  bestCheckpointStep?: number | undefined;
  latestCheckpointKind?: string | undefined;
  activeMemoryBytes?: number | undefined;
  cacheMemoryBytes?: number | undefined;
  peakMemoryBytes?: number | undefined;
  memoryLimitBytes?: number | undefined;
  earlyStopPatience?: number | null | undefined;
  earlyStopMinDelta?: number | undefined;
  earlyStopConsecutiveBadEvals?: number | undefined;
  earlyStopReason?: string | undefined;
  exitCode?: number | null | undefined;
  signal?: string | null | undefined;
  resumeFrom?: string | undefined;
  controlCommand?: RunControlCommand | undefined;
  controlRequestedAt?: string | undefined;
  stallReason?: string | undefined;
} & TExtras;

export type RunHealth = {
  supervisorAlive: boolean;
  trainerAlive: boolean;
  operatorHealth: OperatorHealth;
};
