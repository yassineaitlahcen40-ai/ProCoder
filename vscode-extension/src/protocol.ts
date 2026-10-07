export const PROTOCOL_VERSION = 1;
export const MAX_PROTOCOL_LINE_BYTES = 1_000_000;

export interface BridgeEvent {
  protocol_version: number;
  type: string;
  [key: string]: unknown;
}

export type WorkflowStatus =
  | "idle"
  | "planning"
  | "generating"
  | "testing"
  | "reviewing"
  | "repairing"
  | "recording"
  | "transcribing"
  | "awaiting_confirmation"
  | "success"
  | "failed";

export interface WorkflowState {
  status: WorkflowStatus;
  stage: string;
  runId?: string;
  repairAttempt?: number;
  maxRepairAttempts?: number;
  result?: BridgeEvent;
  error?: string;
}

export function parseBridgeEvent(line: string): BridgeEvent | undefined {
  if (Buffer.byteLength(line, "utf8") > MAX_PROTOCOL_LINE_BYTES) {
    return undefined;
  }
  try {
    const parsed: unknown = JSON.parse(line);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      !("protocol_version" in parsed) ||
      !("type" in parsed)
    ) {
      return undefined;
    }
    const candidate = parsed as Record<string, unknown>;
    if (
      candidate.protocol_version !== PROTOCOL_VERSION ||
      typeof candidate.type !== "string" ||
      candidate.type.length > 80
    ) {
      return undefined;
    }
    return candidate as BridgeEvent;
  } catch {
    return undefined;
  }
}

export function transition(
  current: WorkflowState,
  event: BridgeEvent,
): WorkflowState {
  const runId = typeof event.run_id === "string" ? event.run_id : current.runId;
  switch (event.type) {
    case "workflow_started":
      return { status: "planning", stage: "Starting workflow", runId };
    case "planning_started":
      return { ...current, status: "planning", stage: "Planning", runId };
    case "planning_completed":
      return { ...current, status: "generating", stage: "Planning complete", runId };
    case "generation_started":
      return { ...current, status: "generating", stage: "Generating code", runId };
    case "generation_completed":
      return { ...current, status: "testing", stage: "Generation complete", runId };
    case "testing_started":
      return { ...current, status: "testing", stage: "Testing in Docker", runId };
    case "testing_completed":
      return { ...current, status: "reviewing", stage: "Docker testing complete", runId };
    case "review_started":
      return { ...current, status: "reviewing", stage: "Reviewing test evidence", runId };
    case "review_completed":
      return { ...current, status: "reviewing", stage: "Review complete", runId };
    case "repair_started": {
      const attempt = typeof event.attempt === "number" ? event.attempt : undefined;
      const maximum =
        typeof event.max_attempts === "number" ? event.max_attempts : undefined;
      return {
        ...current,
        status: "repairing",
        stage: attempt === undefined ? "Repairing" : `Repairing (${attempt}/${maximum ?? "?"})`,
        runId,
        repairAttempt: attempt,
        maxRepairAttempts: maximum,
      };
    }
    case "repair_completed":
      return { ...current, status: "reviewing", stage: "Repair cycle complete", runId };
    case "voice_starting":
      return { status: "recording", stage: "Starting microphone" };
    case "voice_recording_started":
      return { status: "recording", stage: "Recording (bounded)" };
    case "voice_transcription_started":
      return { status: "transcribing", stage: "Transcribing locally" };
    case "transcript_ready":
      return { ...current, status: "awaiting_confirmation", stage: "Confirm transcript" };
    case "transcript_approved":
      return { status: "planning", stage: "Transcript approved; starting workflow" };
    case "transcript_rejected":
    case "voice_recording_cancelled":
      return { status: "idle", stage: "Idle" };
    case "workflow_completed":
      return {
        ...current,
        status: event.success === true ? "success" : "failed",
        stage: event.success === true ? "Success" : "Failed",
        runId,
        result: event,
      };
    case "workflow_failed":
    case "workflow_stage_failed":
      return {
        ...current,
        status: "failed",
        stage: "Failed",
        runId,
        error: typeof event.message === "string" ? event.message : "Backend failed",
      };
    case "voice_failed":
    case "protocol_error":
      return {
        ...current,
        status: "failed",
        stage: "Failed",
        error: typeof event.message === "string" ? event.message : "Request failed",
      };
    case "cancel_unavailable":
      return {
        ...current,
        error:
          typeof event.message === "string"
            ? event.message
            : "Cancellation is unavailable",
      };
    default:
      return current;
  }
}

const RUN_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isSafeGeneratedRelativePath(path: string): boolean {
  if (!path || path.length > 240 || path.includes("\0") || path.includes("\\")) {
    return false;
  }
  if (path.startsWith("/") || /^[a-zA-Z]:/.test(path)) {
    return false;
  }
  const parts = path.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) {
    return false;
  }
  const sensitiveNames = new Set([
    ".env",
    ".git",
    ".docker",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".aws",
    ".azure",
    ".gcloud",
    ".ssh",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.toml",
    "id_rsa",
    "id_ed25519",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    ".dockerignore",
  ]);
  return !parts.some((part) => {
    const name = part.toLowerCase();
    return (
      name.startsWith(".env") ||
      sensitiveNames.has(name) ||
      ["secret", "credential", "token", "api_key", "apikey", "access_key", "private_key"].some(
        (marker) => name.includes(marker),
      ) ||
      name.endsWith(".pem") ||
      name.endsWith(".key")
    );
  });
}

export function isSafeRunId(runId: string): boolean {
  return RUN_ID.test(runId);
}
