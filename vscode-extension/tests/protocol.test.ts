import assert from "node:assert/strict";
import test from "node:test";

import {
  isSafeGeneratedRelativePath,
  isSafeRunId,
  parseBridgeEvent,
  transition,
  type WorkflowState,
} from "../src/protocol";

test("protocol parser accepts versioned events and rejects malformed or future versions", () => {
  assert.equal(parseBridgeEvent('{"protocol_version":1,"type":"ready"}')?.type, "ready");
  assert.equal(parseBridgeEvent("{bad json"), undefined);
  assert.equal(parseBridgeEvent('{"protocol_version":2,"type":"ready"}'), undefined);
  assert.equal(parseBridgeEvent('{"protocol_version":1}'), undefined);
});

test("workflow state follows planning, repair, retest, and final completion", () => {
  let state: WorkflowState = { status: "idle", stage: "Idle" };
  state = transition(state, { protocol_version: 1, type: "workflow_started", run_id: "run" });
  assert.equal(state.status, "planning");
  state = transition(state, { protocol_version: 1, type: "repair_started", attempt: 1, max_attempts: 3 });
  assert.equal(state.stage, "Repairing (1/3)");
  state = transition(state, { protocol_version: 1, type: "repair_completed" });
  assert.equal(state.status, "reviewing");
  state = transition(state, { protocol_version: 1, type: "workflow_completed", success: true });
  assert.equal(state.status, "success");
});

test("generated-file paths reject traversal, absolute paths, and secret files", () => {
  assert.equal(isSafeGeneratedRelativePath("src/main.py"), true);
  for (const unsafe of [
    "../outside.txt",
    "src/../../outside",
    "/absolute",
    "C:/outside",
    ".env",
    ".env.example",
    "src/.env",
    "src/.git/config",
    "Dockerfile",
    "src/api_key.txt",
    "src\\main.py",
  ]) {
    assert.equal(isSafeGeneratedRelativePath(unsafe), false, unsafe);
  }
  assert.equal(isSafeRunId("bcf3a60b-f031-41a3-830d-ae941f8b948d"), true);
  assert.equal(isSafeRunId("../escape"), false);
});
