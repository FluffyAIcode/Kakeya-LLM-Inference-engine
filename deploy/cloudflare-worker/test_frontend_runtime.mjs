import assert from "node:assert/strict";
import test from "node:test";

import { JSDOM, VirtualConsole } from "jsdom";

import { PAGE } from "./src/page.js";

const proofBase = {
  ledger_version: 91,
  nodes: [
    {
      id: "root",
      parent_id: "",
      statement: "Root proof obligation",
      status: "UNRESOLVED",
      formal_status: "UNFORMALIZED",
    },
    {
      id: "leaf",
      parent_id: "root",
      statement: "Active graph obligation",
      status: "UNRESOLVED",
      formal_status: "TYPECHECKED",
    },
  ],
  longest_chain: ["root", "leaf"],
  active_leaf_id: "leaf",
  active_obligation: { id: "leaf", statement: "Active graph obligation" },
  unresolved_leaves: 1,
  status_counts: { PROVED: 4 },
  reasoning_runs: [
    {
      run_id: "completed-1",
      outcome: "host_gate_rejected",
      target_statement: "Previous obligation",
      mathematical_summary: "The latest completed reasoning result.",
      updated_at: 1784969000,
    },
  ],
  collaboration: { chain_participants: [], authoritative_gates: [], open_slots: 3 },
  review_health: { status: "healthy", latest_errors: [] },
};

const networkNodes = [
  {
    id: "allens-mini",
    prefill_worker: { tokens_per_second: 2.75 },
  },
];

function inlineScripts(html) {
  return [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
    .map(match => match[1]);
}

function assertStandaloneBrowserScripts(html) {
  const scripts = inlineScripts(html);
  assert.ok(scripts.length, "served HTML must contain an inline application script");
  for (const script of scripts) {
    assert.doesNotMatch(script, /\b__name\b/);
  }
}

async function render(liveExecution) {
  const runtimeErrors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("error", error => runtimeErrors.push(error));
  virtualConsole.on("jsdomError", error => runtimeErrors.push(error));

  const dom = new JSDOM(PAGE, {
    runScripts: "dangerously",
    url: "https://kakeya.ai/",
    virtualConsole,
    beforeParse(window) {
      window.matchMedia = () => ({
        matches: false,
        addEventListener() {},
        removeEventListener() {},
      });
      window.requestAnimationFrame = callback => {
        callback(0);
        return 1;
      };
      window.fetch = async input => {
        const path = new URL(String(input), window.location.href).pathname;
        const body = path === "/v1/proof/progress"
          ? { ...proofBase, live_execution: liveExecution }
          : path === "/v1/network/nodes"
            ? networkNodes
            : null;
        if (body === null) return { ok: false, status: 404, json: async () => ({}) };
        return { ok: true, status: 200, json: async () => structuredClone(body) };
      };
      window.addEventListener("error", event => {
        runtimeErrors.push(event.error || event.message);
      });
      window.addEventListener("unhandledrejection", event => {
        runtimeErrors.push(event.reason);
      });
    },
  });

  await new Promise(resolve => setTimeout(resolve, 30));
  return {
    dom,
    document: dom.window.document,
    runtimeErrors,
  };
}

test("static scan rejects the exact Worker helper regression fixture", () => {
  assertStandaloneBrowserScripts(PAGE);
  const brokenFixture = "<script>const callback = __name(() => 1, 'callback');</script>";
  assert.throws(
    () => assertStandaloneBrowserScripts(brokenFixture),
    /did not match|__name/,
  );
});

test("served inline script renders active telemetry, role pipeline, graph, and metrics", async t => {
  const result = await render({
    fresh: true,
    state: "prefill",
    phase: "decomposer_prefill",
    execution_state: "prefill",
    execution_phase: "decomposer_prefill",
    orchestration_state: "DECOMPOSER",
    active_role: "decomposer",
    run_id: "active-run",
    worker: "allens",
    sequence: 12,
    progress: { current: 5, total: 10 },
  });
  t.after(() => result.dom.window.close());

  assert.deepEqual(result.runtimeErrors, []);
  assert.equal(result.document.querySelector("#backendAnswer").textContent, "Backend live");
  assert.match(result.document.querySelector("#roleAnswer").textContent, /Decomposer/);
  assert.match(result.document.querySelector("#progressLabel").textContent, /5 \/ 10/);
  assert.equal(result.document.querySelectorAll(".node").length, 2);
  assert.equal(result.document.querySelector("#graphActiveTitle").textContent, "Active graph obligation");
  assert.equal(result.document.querySelector("#ledgerVersion").textContent, "v91");
  assert.equal(result.document.querySelector("#prefillRate").textContent, "2.75");
});

test("served inline script renders blocked idle without claiming active execution", async t => {
  const result = await render({
    fresh: true,
    state: "idle",
    phase: "blocked_idle",
    adapter_status: "ADAPTER_BLOCKED",
    blocked_category: "DEFINITION_REQUIRED",
    blocked_reason: "Typed definition required",
    orchestration_state: "DEFINITION_AUDITOR",
    active_role: "definition_auditor",
    resume_role: "definition_auditor",
    progress: { current: 8, total: 10 },
  });
  t.after(() => result.dom.window.close());

  assert.deepEqual(result.runtimeErrors, []);
  assert.equal(result.document.querySelector("#backendAnswer").textContent, "Blocked / idle");
  assert.equal(result.document.querySelector("#roleAnswer").textContent, "Blocked / idle");
  assert.match(result.document.querySelector("#roleDetail").textContent, /Resume role: Definition Auditor/);
  assert.equal(result.document.querySelector("#progressLabel").textContent, "0 / 0 · no active iteration");
  assert.equal(result.document.querySelectorAll(".pipeline-step.current").length, 0);
  assert.equal(result.document.querySelectorAll(".node").length, 2);
});
