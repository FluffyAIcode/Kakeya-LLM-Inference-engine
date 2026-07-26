import assert from "node:assert/strict";
import test from "node:test";

import { PAGE, executionSnapshot } from "./src/page.js";


test("blocked idle overrides retained resumable role", () => {
  const state = executionSnapshot({
    fresh: true,
    state: "idle",
    phase: "blocked_idle",
    adapter_status: "ADAPTER_BLOCKED",
    orchestration_state: "DEFINITION_AUDITOR",
    active_role: "definition_auditor",
    resume_role: "definition_auditor",
  });
  assert.equal(state.blocked, true);
  assert.equal(state.activeInference, false);
  assert.equal(state.resumeRole, "definition_auditor");
  assert.match(PAGE, /Blocked \/ idle/);
  assert.match(PAGE, /Resume role:/);
});


test("only fresh inference heartbeat marks a role active", () => {
  const fresh = executionSnapshot({
    fresh: true,
    state: "decode",
    phase: "definition_auditor_decode",
    active_role: "definition_auditor",
  });
  const stale = executionSnapshot({
    fresh: false,
    state: "decode",
    phase: "definition_auditor_decode",
    active_role: "definition_auditor",
  });
  assert.equal(fresh.activeInference, true);
  assert.equal(stale.activeInference, false);
});


test("same-run state changes produce a new render snapshot", () => {
  const run = {fresh: true, run_id: "same", sequence: 1};
  const active = executionSnapshot({
    ...run,
    state: "prefill",
    phase: "definition_auditor_prefill",
  });
  const blocked = executionSnapshot({
    ...run,
    sequence: 2,
    state: "idle",
    phase: "blocked_idle",
    adapter_status: "ADAPTER_BLOCKED",
    resume_role: "definition_auditor",
  });
  assert.equal(active.activeInference, true);
  assert.equal(blocked.activeInference, false);
  assert.equal(blocked.blocked, true);
  assert.match(PAGE, /setInterval\(load,1800\)/);
}
);
