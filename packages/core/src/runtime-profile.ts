type TimingMetric = {
  count: number;
  totalNs: number;
};

export type CoreRuntimeProfileSnapshot = {
  enabled: boolean;
  outSlot: TimingMetric;
  ffiInvoke: TimingMetric;
  wrapperConstruct: TimingMetric;
  registryRegister: TimingMetric;
  explicitFree: TimingMetric;
  registryUnregister: TimingMetric;
  nativeFree: TimingMetric;
  ffiLabels: Record<string, TimingMetric>;
};

const ENABLED_FLAG = "MLXTS_RUNTIME_PROFILE";

const state: Omit<CoreRuntimeProfileSnapshot, "enabled"> = {
  outSlot: { count: 0, totalNs: 0 },
  ffiInvoke: { count: 0, totalNs: 0 },
  wrapperConstruct: { count: 0, totalNs: 0 },
  registryRegister: { count: 0, totalNs: 0 },
  explicitFree: { count: 0, totalNs: 0 },
  registryUnregister: { count: 0, totalNs: 0 },
  nativeFree: { count: 0, totalNs: 0 },
  ffiLabels: {},
};

function nowNs(): number {
  return Bun.nanoseconds();
}

function zeroMetric(metric: TimingMetric): void {
  metric.count = 0;
  metric.totalNs = 0;
}

function recordMetric(metric: TimingMetric, durationNs: number): void {
  metric.count += 1;
  metric.totalNs += durationNs;
}

export function isCoreRuntimeProfilingEnabled(): boolean {
  return process.env[ENABLED_FLAG] === "1";
}

export function coreRuntimeProfileTimestamp(): number {
  return isCoreRuntimeProfilingEnabled() ? nowNs() : 0;
}

export function resetCoreRuntimeProfile(): void {
  zeroMetric(state.outSlot);
  zeroMetric(state.ffiInvoke);
  zeroMetric(state.wrapperConstruct);
  zeroMetric(state.registryRegister);
  zeroMetric(state.explicitFree);
  zeroMetric(state.registryUnregister);
  zeroMetric(state.nativeFree);
  for (const metric of Object.values(state.ffiLabels)) {
    zeroMetric(metric);
  }
}

export function recordOutSlotDuration(durationNs: number): void {
  if (!isCoreRuntimeProfilingEnabled()) {
    return;
  }
  recordMetric(state.outSlot, durationNs);
}

export function recordFfiInvokeDuration(label: string, durationNs: number): void {
  if (!isCoreRuntimeProfilingEnabled()) {
    return;
  }
  recordMetric(state.ffiInvoke, durationNs);
  let metric = state.ffiLabels[label];
  if (metric === undefined) {
    metric = { count: 0, totalNs: 0 };
    state.ffiLabels[label] = metric;
  }
  recordMetric(metric, durationNs);
}

export function recordWrapperConstructDuration(durationNs: number, registerNs: number): void {
  if (!isCoreRuntimeProfilingEnabled()) {
    return;
  }
  recordMetric(state.wrapperConstruct, durationNs);
  recordMetric(state.registryRegister, registerNs);
}

export function recordExplicitFreeDuration(
  durationNs: number,
  unregisterNs: number,
  nativeFreeNs: number,
): void {
  if (!isCoreRuntimeProfilingEnabled()) {
    return;
  }
  recordMetric(state.explicitFree, durationNs);
  recordMetric(state.registryUnregister, unregisterNs);
  recordMetric(state.nativeFree, nativeFreeNs);
}

export function snapshotCoreRuntimeProfile(): CoreRuntimeProfileSnapshot {
  return {
    enabled: isCoreRuntimeProfilingEnabled(),
    outSlot: { ...state.outSlot },
    ffiInvoke: { ...state.ffiInvoke },
    wrapperConstruct: { ...state.wrapperConstruct },
    registryRegister: { ...state.registryRegister },
    explicitFree: { ...state.explicitFree },
    registryUnregister: { ...state.registryUnregister },
    nativeFree: { ...state.nativeFree },
    ffiLabels: Object.fromEntries(
      Object.entries(state.ffiLabels).map(([label, metric]) => [label, { ...metric }]),
    ),
  };
}
