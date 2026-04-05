import type { RunHealth, RunState, RunStatus } from "./files-types";

function runStateIsTerminal(state: RunState): boolean {
  return (
    state === "stopped" || state === "completed" || state === "failed" || state === "cancelled"
  );
}

export function activePid(pid: number | undefined): boolean {
  if (pid === undefined) {
    return false;
  }

  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function deriveOperatorHealth(status: RunStatus): RunHealth {
  const supervisorAlive = activePid(status.supervisorPid);
  const trainerAlive = activePid(status.trainerPid);

  if (status.state === "starting") {
    return {
      supervisorAlive,
      trainerAlive,
      operatorHealth: supervisorAlive ? "healthy" : "dead-supervisor",
    };
  }

  if (supervisorAlive && trainerAlive) {
    return { supervisorAlive, trainerAlive, operatorHealth: "healthy" };
  }

  if (!supervisorAlive && !trainerAlive) {
    return {
      supervisorAlive,
      trainerAlive,
      operatorHealth: runStateIsTerminal(status.state) ? "healthy" : "dead-both",
    };
  }

  if (!supervisorAlive) {
    return { supervisorAlive, trainerAlive, operatorHealth: "dead-supervisor" };
  }

  return { supervisorAlive, trainerAlive, operatorHealth: "dead-trainer" };
}
