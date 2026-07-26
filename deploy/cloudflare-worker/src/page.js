export function executionSnapshot(live) {
  const canonical = value => String(value ?? "").trim().toLowerCase()
    .replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "")
    .replace(/^agent_/, "");
  const state = String(live?.execution_state || live?.state || "idle").toLowerCase();
  const phase = String(live?.execution_phase || live?.phase || "").toLowerCase();
  const adapter = String(live?.adapter_status || "").toUpperCase();
  const blockedCategory = String(live?.blocked_category || "").toUpperCase();
  const blocked = Boolean(adapter || blockedCategory) || phase === "blocked_idle";
  const activeInference = live?.fresh === true && !blocked
    && ["queued", "prefill", "decode", "review"].includes(state);
  const resumeRole = canonical(
    live?.resume_role || live?.active_role || live?.orchestration_state,
  );
  return {
    state, phase, adapter, blockedCategory, blocked, activeInference, resumeRole,
  };
}

export const PAGE = String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kakeya Proof Observatory</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/core@1.4.0/dist/css/tabler.min.css" crossorigin="anonymous" onerror="this.remove()">
<style>
:root{color-scheme:dark;--bg:#090c0a;--surface:#111612;--surface-2:#171e18;--line:#303a31;--line-strong:#465549;--text:#f2efe7;--muted:#a7b0a6;--green:#91d6a2;--green-bg:#14251a;--amber:#e1bb67;--amber-bg:#2b2415;--red:#e18a78;--blue:#8fc1d4;--focus:#b7ddeb}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;max-width:100%;background:var(--bg);color:var(--text);font-family:"IBM Plex Mono","SFMono-Regular",Consolas,monospace}body{overflow-x:hidden;background-image:radial-gradient(circle at 50% -16%,#19251c 0,transparent 44%)}button{font:inherit}.page{min-height:100vh}.container-xl{width:100%;max-width:1500px;margin-inline:auto;padding-inline:24px}.observatory-nav{position:sticky;top:0;z-index:30;min-height:58px;border-bottom:1px solid var(--line);background:rgba(9,12,10,.96);backdrop-filter:blur(12px)}.observatory-nav .container-xl{min-height:58px;display:flex;align-items:center;gap:16px}.navbar-brand{display:flex;align-items:center;gap:11px;min-width:0;color:inherit;text-decoration:none}.mark{width:35px;height:35px;flex:0 0 auto;display:grid;place-items:center;border:1px solid var(--green);border-radius:50%;font:italic 21px Georgia;color:var(--green)}h1{margin:0;font:600 18px/1.15 Georgia,serif}.kicker,.eyebrow,.label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.11em}.nav-meta{margin-left:auto;display:flex;align-items:center;justify-content:flex-end;gap:7px;min-width:0}.badge{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:5px 8px;color:var(--muted);font-size:9px;letter-spacing:.04em;white-space:nowrap}.build-badge{border-color:#705f36;color:var(--amber)}.status-dot{width:7px;height:7px;border-radius:50%;background:var(--muted)}.status-badge.live{border-color:#3c6d49;color:var(--green);background:var(--green-bg)}.status-badge.live .status-dot{background:var(--green);animation:pulse 1.6s infinite}.status-badge.stale{border-color:#725d2f;color:var(--amber);background:var(--amber-bg)}.status-badge.stale .status-dot{background:var(--amber)}.status-badge.error{border-color:#743f38;color:var(--red)}.page-wrapper{min-width:0}.page-body{padding:18px 0 40px}
.answer-strip{display:grid;grid-template-columns:1.05fr 1.3fr 2fr 1.55fr;border:1px solid var(--line);background:var(--line);gap:1px}.answer{min-width:0;padding:15px;background:var(--surface)}.answer.primary{background:linear-gradient(135deg,#17231a,var(--surface))}.answer .eyebrow{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}.answer-value{font:600 17px/1.25 Georgia,serif;overflow-wrap:anywhere}.answer.primary .answer-value{font-size:21px;color:var(--green)}.answer-detail{margin-top:7px;color:var(--muted);font:11px/1.45 Georgia,serif;overflow-wrap:anywhere}.answer-detail strong{color:var(--text);font-family:inherit}.telemetry-tag{font:8px/1 monospace;border:1px solid var(--line);border-radius:999px;padding:3px 5px}.telemetry-tag.fresh{color:var(--green);border-color:#3c6d49}.telemetry-tag.stale{color:var(--amber);border-color:#725d2f}.execution-progress{height:5px;margin:11px 0 7px;background:#273028;overflow:hidden}.execution-progress i{display:block;height:100%;width:0;background:var(--green);transition:width .35s}.progress-meta{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:9px}.math{white-space:normal;overflow-wrap:anywhere;font-family:Georgia,"Times New Roman",serif}.math.compact{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:3;overflow:hidden}.latest-change{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:4;overflow:hidden}
.pipeline-card{margin-top:12px;border:1px solid var(--line);background:var(--surface)}.pipeline-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:9px 12px;border-bottom:1px solid var(--line)}.pipeline{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));padding:11px 12px;overflow:hidden}.pipeline-step{position:relative;min-width:0;padding-top:18px;text-align:center}.pipeline-step:before{content:"";position:absolute;top:5px;left:0;right:0;height:1px;background:var(--line-strong)}.pipeline-step:first-child:before{left:50%}.pipeline-step:last-child:before{right:50%}.pipeline-step i{position:absolute;z-index:1;top:0;left:calc(50% - 6px);width:12px;height:12px;border:2px solid var(--line-strong);border-radius:50%;background:var(--surface)}.pipeline-step.done i{border-color:var(--green);background:var(--green)}.pipeline-step.reused i{border-color:var(--blue);background:var(--blue)}.pipeline-step.skipped i{border-style:dashed;opacity:.6}.pipeline-step.current i{border-color:var(--amber);background:var(--surface);box-shadow:0 0 0 4px rgba(225,187,103,.16)}.pipeline-step.failed i{border-color:var(--red);background:var(--red)}.pipeline-step span{display:block;padding-inline:3px;color:var(--muted);font-size:8px;line-height:1.25;overflow-wrap:anywhere}.pipeline-step small{display:block;margin-top:3px;color:#758077;font-size:7px;text-transform:uppercase}.pipeline-step.current span,.pipeline-step.current small{color:var(--amber)}.pipeline-step.done span{color:#c7d4c8}.pipeline-step.reused span,.pipeline-step.reused small{color:var(--blue)}.pipeline-step.skipped span{text-decoration:line-through;opacity:.65}
.workspace{margin-top:14px;border:1px solid var(--line);background:var(--surface)}.nav-tabs{display:flex;align-items:center;gap:3px;min-height:52px;padding:8px 10px;border-bottom:1px solid var(--line);overflow-x:auto;scrollbar-width:thin}.tab-btn{flex:0 0 auto;border:1px solid transparent;border-radius:4px;padding:8px 12px;background:transparent;color:var(--muted);cursor:pointer}.tab-btn[aria-selected="true"]{border-color:var(--line-strong);background:var(--surface-2);color:var(--text)}.tab-btn:hover{color:var(--text)}.tab-btn .count{margin-left:5px;color:var(--muted);font-size:9px}.tab-panel{display:none}.tab-panel.active{display:block}.proof-layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;min-width:0}.graph-area{min-width:0;border-right:1px solid var(--line)}.graph-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 12px;border-bottom:1px solid var(--line)}.graph-title{min-width:0}.graph-title b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:600 13px Georgia,serif}.controls{display:flex;align-items:center;gap:5px;flex:0 0 auto}.icon-btn{min-width:34px;min-height:32px;border:1px solid var(--line);border-radius:3px;background:transparent;color:var(--muted);cursor:pointer}.icon-btn:hover{border-color:var(--blue);color:var(--text)}.icon-btn.active{color:var(--green);border-color:#3c6d49}.graph-viewport{position:relative;width:100%;height:570px;overflow:hidden;touch-action:none;cursor:grab;background-image:linear-gradient(rgba(48,58,49,.55) 1px,transparent 1px),linear-gradient(90deg,rgba(48,58,49,.55) 1px,transparent 1px);background-size:32px 32px}.graph-viewport.dragging{cursor:grabbing}.graph-scene{position:absolute;left:0;top:0;width:1200px;height:620px;transform-origin:0 0}.graph-scene svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.edge{stroke:#48554a;stroke-width:1.5;fill:none}.edge.active{stroke:var(--green);stroke-width:2.2}.node{position:absolute;width:205px;min-height:84px;max-height:112px;padding:10px;border:1px solid var(--line-strong);border-radius:5px;background:#111612;color:var(--text);text-align:left;cursor:pointer;overflow:hidden}.node:hover{border-color:var(--blue)}.node.active{border-color:var(--green);background:#142219}.node.selected{outline:2px solid var(--focus);outline-offset:3px}.node.rejected{opacity:.55}.node .id{color:var(--muted);font-size:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.node .title{display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:3;margin-top:6px;overflow:hidden;font:600 12px/1.3 Georgia,serif}.node .state{margin-top:6px;color:var(--amber);font-size:8px}.node.active .state{color:var(--green)}.riemann{position:absolute;z-index:4;width:28px;height:28px;display:grid;place-items:center;border:1px solid #70644f;border-radius:50%;background:#171c18;color:#d8c8aa;font:italic 16px Georgia;pointer-events:none;transition:transform .8s ease}.riemann:after{content:"ζ";}.legend{position:absolute;left:10px;bottom:10px;z-index:6;display:flex;gap:10px;flex-wrap:wrap;padding:6px 8px;border:1px solid var(--line);background:rgba(9,12,10,.9);color:var(--muted);font-size:8px}.legend i{display:inline-block;width:8px;height:8px;margin-right:4px;border:1px solid var(--line-strong)}.legend .lg-active{border-color:var(--green);background:var(--green-bg)}.legend .lg-selected{outline:1px solid var(--focus)}.legend .lg-rejected{opacity:.45}
.inspector{min-width:0;display:flex;flex-direction:column}.inspector-head{padding:12px;border-bottom:1px solid var(--line)}.inspector-head b{display:block;margin-top:5px;font:600 15px/1.25 Georgia,serif;overflow-wrap:anywhere}.inspector-body{padding:12px}.fact{display:grid;grid-template-columns:82px minmax(0,1fr);gap:8px;padding:7px 0;border-bottom:1px solid #252d26;font-size:9px}.fact span:first-child{color:var(--muted)}.fact span:last-child{overflow-wrap:anywhere}.evidence{margin:13px 0 0;color:#d7d3c9;font:12px/1.5 Georgia,serif;overflow-wrap:anywhere}.inspector-empty{color:var(--muted)}.advisory{margin-top:auto;padding:12px;border-top:1px solid var(--line);background:#0d120e}.advisory .chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.chip{max-width:100%;padding:4px 6px;border:1px dashed var(--line-strong);color:var(--muted);font-size:8px;overflow-wrap:anywhere}.chip.gate{border-style:solid;border-color:#3c6d49;color:var(--green)}.chip.warning{border-color:#725d2f;color:var(--amber)}
.history{padding:16px}.history-intro{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:14px}.history-intro h2{margin:4px 0 0;font:600 20px Georgia,serif}.timeline{position:relative;margin-left:9px;padding-left:24px;border-left:1px solid var(--line-strong)}.run-item{position:relative;margin-bottom:12px;padding:14px;border:1px solid var(--line);background:#0f140f}.run-item:before{content:"";position:absolute;left:-31px;top:18px;width:12px;height:12px;border:2px solid var(--line-strong);border-radius:50%;background:var(--surface)}.run-item:first-child:before{border-color:var(--amber);background:var(--amber-bg)}.run-meta{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:9px}.run-item h3{margin:8px 0 6px;font:600 15px/1.35 Georgia,serif}.run-item p{margin:0;color:#d7d3c9;font:12px/1.5 Georgia,serif}.run-item small{display:block;margin-top:9px;color:var(--amber);font-size:9px}.infra{padding:16px}.infra-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.metric{padding:14px;border:1px solid var(--line);background:#0f140f}.metric strong{display:block;font-size:22px;font-weight:550;overflow:hidden;text-overflow:ellipsis}.metric span{display:block;margin-top:4px;color:var(--muted);font-size:9px}.infra details{margin-top:14px;border:1px solid var(--line);padding:12px}.infra summary{cursor:pointer;color:var(--muted);font-size:10px}.infra-copy{margin-top:10px;color:var(--muted);font:12px/1.5 Georgia,serif}.footer{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding-top:14px;color:var(--muted);font-size:9px}
.graph-title{width:100%;max-width:100%;overflow:hidden}.graph-title b{max-width:100%}
button:focus-visible,[tabindex]:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
@keyframes pulse{50%{opacity:.35;transform:scale(.72)}}
@media(max-width:1100px){.answer-strip{grid-template-columns:1fr 1fr}.proof-layout{grid-template-columns:minmax(0,1fr) 270px}.pipeline-step span{font-size:7px}}
@media(max-width:900px){.proof-layout{display:block}.graph-area{border-right:0;border-bottom:1px solid var(--line)}.inspector{min-height:230px}.graph-viewport{height:520px}}
@media(max-width:767.98px){.container-xl{padding-inline:12px}.observatory-nav,.observatory-nav .container-xl{min-height:52px}.brand-copy .kicker{display:none}h1{font-size:16px}.build-badge{display:none}.page-body{padding-top:12px}.answer-strip{grid-template-columns:1fr}.answer{padding:12px}.answer.primary .answer-value{font-size:19px}.pipeline{display:flex;flex-direction:column;padding:10px 14px}.pipeline-step{min-height:32px;padding:3px 0 3px 25px;text-align:left}.pipeline-step:before{left:5px!important;right:auto!important;top:0;bottom:0;width:1px;height:auto}.pipeline-step:first-child:before{top:8px}.pipeline-step:last-child:before{bottom:22px}.pipeline-step i{left:0;top:6px}.pipeline-step span{font-size:9px}.proof-layout{display:block}.graph-area{border-right:0;border-bottom:1px solid var(--line)}.graph-toolbar{align-items:flex-start;flex-direction:column}.controls{width:100%}.controls .icon-btn{flex:1}.graph-viewport{height:480px}.inspector{min-height:230px}.infra-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.history{padding:12px}.run-meta{flex-direction:column;gap:3px}}
@media(max-width:420px){.container-xl{padding-inline:8px}.navbar-brand{gap:8px}.mark{width:31px;height:31px;font-size:18px}.nav-meta{gap:4px}.badge{padding:4px 6px}.status-badge span:last-child{max-width:106px;overflow:hidden;text-overflow:ellipsis}.nav-tabs{padding:6px}.tab-btn{padding:8px 9px;font-size:10px}.graph-viewport{height:430px}.infra-grid{grid-template-columns:1fr}.footer{display:block}.footer span{display:block;margin-top:5px}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}.riemann{display:none}}
</style>
</head>
<body>
<div class="page">
  <header class="observatory-nav"><div class="container-xl">
    <a class="navbar-brand" href="/" aria-label="Kakeya Proof Observatory home"><span class="mark">ζ</span><span class="brand-copy"><span class="kicker">Kakeya Proof Observatory</span><h1>Riemann Hypothesis · proof operations</h1></span></a>
    <div class="nav-meta"><span class="badge build-badge" id="buildBadge">BUILD 2026.07.25-FRONTEND-RUNTIME-R2</span><span class="badge status-badge stale" id="backendBadge"><i class="status-dot"></i><span id="backendLabel">CONNECTING</span></span></div>
  </div></header>
  <main class="page-body"><div class="container-xl">
    <section class="answer-strip" aria-label="Current execution summary">
      <article class="answer primary"><div class="eyebrow"><span>Backend</span><span class="telemetry-tag stale" id="freshnessTag">CHECKING</span></div><div class="answer-value" id="backendAnswer">Connecting…</div><div class="answer-detail" id="backendDetail">Waiting for authoritative telemetry</div></article>
      <article class="answer"><div class="eyebrow"><span>Exact role / phase</span></div><div class="answer-value" id="roleAnswer">No active execution</div><div class="answer-detail" id="roleDetail">Run and worker unavailable</div></article>
      <article class="answer"><div class="eyebrow"><span>Obligation under attack</span></div><div class="answer-value math compact" id="obligationAnswer">Loading active obligation…</div><div class="answer-detail" id="obligationId">—</div></article>
      <article class="answer"><div class="eyebrow"><span>Latest completed change</span></div><div class="answer-value latest-change" id="latestAnswer">Waiting for completed history</div><div class="answer-detail" id="latestMeta">—</div></article>
    </section>
    <section class="pipeline-card" aria-label="Role and iteration progress">
      <div class="pipeline-head"><span class="label">Role pipeline</span><span class="label" id="progressLabel">0 / 0 · no active iteration</span></div>
      <div class="pipeline" id="rolePipeline"></div>
    </section>
    <section class="workspace">
      <nav class="nav-tabs" role="tablist" aria-label="Observatory views">
        <button class="tab-btn" id="tabProof" role="tab" aria-selected="true" aria-controls="proofPanel" data-tab="proofPanel">Proof Graph <span class="count" id="graphCount">—</span></button>
        <button class="tab-btn" id="tabHistory" role="tab" aria-selected="false" aria-controls="historyPanel" data-tab="historyPanel">Run History <span class="count" id="historyCount">—</span></button>
        <button class="tab-btn" id="tabInfra" role="tab" aria-selected="false" aria-controls="infraPanel" data-tab="infraPanel">Infrastructure</button>
      </nav>
      <section class="tab-panel active" id="proofPanel" role="tabpanel" aria-labelledby="tabProof">
        <div class="proof-layout">
          <div class="graph-area">
            <div class="graph-toolbar"><div class="graph-title"><span class="label">Decomposition graph · active chain emphasized</span><b id="graphActiveTitle">Loading graph…</b></div><div class="controls" aria-label="Graph controls"><button class="icon-btn" id="zoomOut" aria-label="Zoom out">−</button><button class="icon-btn" id="zoomIn" aria-label="Zoom in">+</button><button class="icon-btn" id="fitChain" aria-label="Fit active chain">Fit</button><button class="icon-btn" id="resetView" aria-label="Reset graph view">Reset</button></div></div>
            <div class="graph-viewport" id="graphViewport" aria-label="Interactive proof graph. Drag to pan.">
              <div class="graph-scene" id="graphScene"><svg id="edges" aria-hidden="true"></svg><div class="riemann" id="riemann" aria-hidden="true"></div></div>
              <div class="legend" aria-label="Graph legend"><span><i class="lg-active"></i>active chain</span><span><i class="lg-selected"></i>selected</span><span><i class="lg-rejected"></i>rejected</span></div>
            </div>
          </div>
          <aside class="inspector" aria-label="Selected proof node">
            <div class="inspector-head"><span class="label">Selected-node inspector</span><b id="inspectorTitle">Select a proof task</b></div>
            <div class="inspector-body" id="inspectorBody"><p class="inspector-empty">Choose a node to inspect its formal status and latest evidence.</p></div>
            <div class="advisory"><span class="label">Collaboration policy</span><div class="chips" id="collaborationChips"><span class="chip warning">External slots are advisory only</span></div></div>
          </aside>
        </div>
      </section>
      <section class="tab-panel" id="historyPanel" role="tabpanel" aria-labelledby="tabHistory">
        <div class="history"><div class="history-intro"><div><span class="label">Completed-run reasoning timeline</span><h2>What changed, and why it stopped</h2></div><span class="label">Newest first</span></div><div class="timeline" id="runTimeline"><div class="run-item"><p>Waiting for completed proof runs.</p></div></div></div>
      </section>
      <section class="tab-panel" id="infraPanel" role="tabpanel" aria-labelledby="tabInfra">
        <div class="infra"><div class="infra-grid">
          <div class="metric"><strong id="ledgerVersion">—</strong><span>LEDGER VERSION</span></div><div class="metric"><strong id="nodeCount">—</strong><span>PROOF TASKS</span></div><div class="metric"><strong id="leafCount">—</strong><span>OPEN LEAVES</span></div><div class="metric"><strong id="chainLength">—</strong><span>ACTIVE CHAIN DEPTH</span></div><div class="metric"><strong id="proofCount">—</strong><span>PROVED</span></div><div class="metric"><strong id="prefillRate">—</strong><span>ALLENS PREFILL TOK/S</span></div>
        </div><details><summary>Telemetry provenance and review health</summary><div class="infra-copy" id="infraDetail">Waiting for telemetry.</div></details></div>
      </section>
    </section>
    <footer class="footer"><span>Primary + Allens · host and Lean gates remain authoritative</span><span>kakeya.ai · build 2026.07.25-frontend-runtime-r2</span></footer>
  </div></main>
</div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clean=s=>String(s??'').replace(/\*\*/g,'').replace(/\r?\n+/g,' ').replace(/\s+/g,' ').trim();
const short=(s,n=130)=>{s=clean(s);return s.length>n?s.slice(0,n-1)+'…':s};
const label=s=>String(s??'').replace(/^agent_/,'').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
const roleStages=[
  {id:'strategy_tournament',label:'Strategy Tournament'},
  {id:'research_contract_gate',label:'Research Contract'},
  {id:'premise_auditor',label:'Premise Auditor',branch:true},
  {id:'definition_auditor',label:'Definition Auditor'},
  {id:'counterexample_worker',label:'Counterexample Worker'},
  {id:'synthesis',label:'Synthesis / Reframe'},
  {id:'decomposer',label:'Decomposer'},
  {id:'formalizer',label:'Formalizer'},
  {id:'proof_search',label:'Proof Search'},
  {id:'adversarial_proponent',label:'Adversarial Proponent'},
  {id:'judge',label:'Judge'},
  {id:'commit',label:'Commit'}
];
const roleAliases={
  strategy_tournament:'strategy_tournament',
  research_contract_gate:'research_contract_gate',
  premise_audit:'premise_auditor',premise_auditor:'premise_auditor',
  definition_audit:'definition_auditor',definition_auditor:'definition_auditor',
  counterexample:'counterexample_worker',counterexample_worker:'counterexample_worker',
  synthesis:'synthesis',reframe:'synthesis',
  decomposer:'decomposer',formalizer:'formalizer',math_ir_translation:'formalizer',
  host_typed_ir_gate:'formalizer',lean_elaboration_gate:'formalizer',
  proof_search:'proof_search',
  adversarial_review:'adversarial_proponent',adversarial_proponent:'adversarial_proponent',
  judge:'judge',commit:'commit'
};
function canonicalRoleId(value){
  const id=String(value??'').trim().toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'').replace(/^agent_/,'');
  return roleAliases[id]||'';
}
function activeRoleId(live){
  const machine=canonicalRoleId(live?.orchestration_state);
  return machine||canonicalRoleId(live?.active_role)||canonicalRoleId(live?.role);
}
// Keep browser code explicit: Worker bundlers may rewrite Function#toString() output
// to reference private bundle helpers that do not exist in this document.
function executionSnapshot(live){
  const canonical=value=>String(value??'').trim().toLowerCase()
    .replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'')
    .replace(/^agent_/,'');
  const state=String(live?.execution_state||live?.state||'idle').toLowerCase();
  const phase=String(live?.execution_phase||live?.phase||'').toLowerCase();
  const adapter=String(live?.adapter_status||'').toUpperCase();
  const blockedCategory=String(live?.blocked_category||'').toUpperCase();
  const blocked=Boolean(adapter||blockedCategory)||phase==='blocked_idle';
  const activeInference=live?.fresh===true&&!blocked
    &&['queued','prefill','decode','review'].includes(state);
  const resumeRole=canonical(
    live?.resume_role||live?.active_role||live?.orchestration_state,
  );
  return {state,phase,adapter,blockedCategory,blocked,activeInference,resumeRole};
}
const reduceMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
let proofData=null,positions=new Map(),activeChain=[],selectedId=null,sceneW=1200,sceneH=620,view={x:20,y:20,scale:1},drag=null,hop=0;
function telemetryKind(live){
  if(live?.fresh===true)return 'fresh';
  if(live?.fallback_reason)return 'fallback';
  if(live?.state==='completed')return 'history';
  return live?'stale':'idle';
}
function renderStatus(proof){
  const live=proof.live_execution,kind=telemetryKind(live),fresh=kind==='fresh',execution=executionSnapshot(live),badge=$('backendBadge'),tag=$('freshnessTag');
  badge.className='badge status-badge '+(execution.blocked?'stale':execution.activeInference?'live':kind==='fallback'?'error':'stale');
  tag.className='telemetry-tag '+(fresh?'fresh':'stale');
  const state=label(live?.state||'idle'),age=Number(live?.age_s);
  $('backendLabel').textContent=execution.blocked?'BLOCKED / IDLE':execution.activeInference?'LIVE · '+state.toUpperCase():kind==='fallback'?'FALLBACK':kind==='history'?'COMPLETED HISTORY':kind==='stale'?'STALE TELEMETRY':'IDLE';
  $('backendAnswer').textContent=execution.blocked?'Blocked / idle':execution.activeInference?'Backend live':kind==='fallback'?'Fallback active':kind==='stale'?'Telemetry stale':'Backend idle';
  $('freshnessTag').textContent=fresh?'AUTHORITATIVE · FRESH':kind==='fallback'?'FALLBACK':kind==='history'?'COMPLETED':kind==='stale'?'STALE':'IDLE';
  $('backendDetail').textContent=fresh?(Number.isFinite(age)?age.toFixed(1)+'s old · ':'')+(live.freshness_source||live.hit_source||live.writer_source||'authoritative source')+(execution.blocked&&live?.blocked_category?' · '+label(live.blocked_category):''):live?.fallback_reason||'No fresh execution heartbeat';
  const machine=live?.orchestration_state||'',activeRole=live?.active_role||live?.role||'';
  const machineRole=canonicalRoleId(machine),activeRoleCanonical=canonicalRoleId(activeRole),roleParts=machineRole&&machineRole===activeRoleCanonical?[label(machine)]:[label(machine),label(activeRole)].filter(Boolean);
  $('roleAnswer').textContent=execution.blocked?'Blocked / idle':execution.activeInference?[...roleParts,label(live.phase)].filter(Boolean).join(' · '):kind==='history'?'No active role · completed':'No active role';
  const decomposition=Number(live?.decomposition_iteration)?' · decomposition '+live.decomposition_iteration+(live.viewpoint?' · '+label(live.viewpoint):'')+(live.semantic_rejection?' · semantic proposal rejected':''):'';
  const synthesis=Number(live?.synthesis_iteration)?' · synthesis '+live.synthesis_iteration+' · '+Number(live.candidate_count||0)+' candidates'+(live.selected_move?' · selected '+label(live.selected_move):'')+(live.stagnation_reason?' · '+label(live.stagnation_reason):''):'';
  const resolution=live?.definition_resolution,resolutionDetail=resolution?.current_gap?' · definition '+label(resolution.current_gap)+' · sources '+Object.keys(resolution.source_statuses||{}).length+' · candidates '+Number(resolution.candidate_count||0)+' · branches '+Number(resolution.branch_count||0)+(resolution.lean_status?' · '+label(resolution.lean_status):'')+(resolution.backjump_target?' · typed backjump':''):'';
  $('roleDetail').textContent=execution.blocked?'Resume role: '+label(execution.resumeRole||'unavailable')+' · '+clean(live?.blocked_reason||live?.transition_reason||'blocked'):execution.activeInference?(live.run_id||'run unknown')+' · seq '+(live.sequence??'—')+' · '+(live.worker||'worker unknown')+' · '+(live.transition_reason||state)+(live.resume_origin?' · from '+label(live.resume_origin):'')+(live.strategy_reused?' · Strategy reused':'')+decomposition+synthesis+resolutionDetail+(Number(live.retry_count)?' · retry '+live.retry_count:''):kind==='stale'?'Latest telemetry is historical, not live':'Waiting for a fresh authoritative heartbeat';
}
function renderPipeline(live){
  const execution=executionSnapshot(live),current=execution.activeInference?activeRoleId(live):'',idx=roleStages.findIndex(r=>r.id===current),machine=String(live?.orchestration_state||'').toUpperCase(),failed=execution.activeInference&&(String(live?.state).toLowerCase()==='failed'||machine==='BLOCKED'||machine==='APPROACH_FAILED');
  $('rolePipeline').innerHTML=roleStages.map((r,i)=>{
    let status='pending';
    if(idx>=0&&i<idx)status=r.branch&&current!=='premise_auditor'?'skipped':'completed';
    if(i===idx)status=failed?'failed':'current';
    if(r.id==='strategy_tournament'&&live?.strategy_reused&&['completed','current'].includes(status))status='reused';
    const cls=status==='completed'?'done':status;
    return '<div class="pipeline-step '+cls+'" data-role="'+r.id+'"><i></i><span>'+esc(r.label)+'</span><small>'+status+'</small></div>';
  }).join('');
  const p=live?.progress||{},cur=Number(p.current||0),total=Number(p.total||0),pct=total?Math.min(100,100*cur/total):0;
  $('progressLabel').textContent=execution.activeInference?(cur+' / '+total+' · '+Math.round(pct)+'% through '+label(current||live.role)):'0 / 0 · no active iteration';
}
function hierarchy(nodes){
  const by=new Map(nodes.map(n=>[n.id,n])),depth=new Map(),visiting=new Set();
  function d(n){if(depth.has(n.id))return depth.get(n.id);if(visiting.has(n.id))return 0;visiting.add(n.id);const p=by.get(n.parent_id),v=p?d(p)+1:0;visiting.delete(n.id);depth.set(n.id,v);return v}nodes.forEach(d);return depth;
}
function renderGraph(proof){
  const nodes=proof.nodes||[],depth=hierarchy(nodes),levels={};nodes.forEach(n=>(levels[depth.get(n.id)]??=[]).push(n));
  activeChain=proof.longest_chain||[];const activeSet=new Set(activeChain),maxDepth=Math.max(0,...depth.values());
  sceneW=Math.max(1000,(maxDepth+1)*245+100);const maxLevel=Math.max(1,...Object.values(levels).map(x=>x.length));sceneH=Math.max(620,maxLevel*130+100);
  const scene=$('graphScene'),svg=$('edges');scene.style.width=sceneW+'px';scene.style.height=sceneH+'px';svg.setAttribute('viewBox','0 0 '+sceneW+' '+sceneH);scene.querySelectorAll('.node').forEach(n=>n.remove());svg.innerHTML='';positions=new Map();
  Object.entries(levels).forEach(([dv,list])=>{
    list.sort((a,b)=>(activeSet.has(b.id)?1:0)-(activeSet.has(a.id)?1:0)||String(a.id).localeCompare(String(b.id)));
    const active=list.filter(n=>activeSet.has(n.id)),other=list.filter(n=>!activeSet.has(n.id));let ys=[];
    if(active.length)ys.push({n:active[0],y:Math.max(40,sceneH/2-45)});
    other.forEach((n,i)=>{const row=Math.floor(i/2)+1,side=i%2?1:-1;ys.push({n,y:Math.max(25,Math.min(sceneH-125,sceneH/2-45+side*row*125))})});
    active.slice(1).forEach((n,i)=>ys.push({n,y:Math.max(25,Math.min(sceneH-125,sceneH/2+90+i*115))}));
    ys.forEach(({n,y})=>{const x=45+Number(dv)*245;positions.set(n.id,{x,y});const el=document.createElement('button');el.type='button';el.className='node '+(activeSet.has(n.id)?'active ':'')+(n.id===selectedId?'selected ':'')+(n.status==='REJECTED_DUPLICATE'?'rejected':'');el.dataset.id=n.id;el.style.transform='translate('+x+'px,'+y+'px)';el.setAttribute('aria-label',short(n.statement,100)+' · '+(n.formal_status||n.status));el.innerHTML='<div class="id">'+esc(n.id)+'</div><div class="title">'+esc(short(n.statement,92))+'</div><div class="state">'+esc(n.formal_status||n.status||'UNKNOWN')+'</div>';el.onclick=()=>selectNode(n.id);scene.appendChild(el)});
  });
  nodes.forEach(n=>{const a=positions.get(n.parent_id),b=positions.get(n.id);if(!a||!b)return;const p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d','M '+(a.x+205)+' '+(a.y+42)+' C '+(a.x+225)+' '+(a.y+42)+', '+(b.x-20)+' '+(b.y+42)+', '+b.x+' '+(b.y+42));p.setAttribute('class','edge '+(activeSet.has(n.id)?'active':''));svg.appendChild(p)});
  $('graphCount').textContent=nodes.length+' tasks';$('graphActiveTitle').textContent=short(nodes.find(n=>n.id===proof.active_leaf_id)?.statement||'No active leaf',150);
  const current=proof.active_leaf_id&&nodes.find(n=>n.id===proof.active_leaf_id);if(!selectedId&&current)selectNode(current.id,false);fitActiveChain();
}
function selectNode(id,refit=false){
  selectedId=id;document.querySelectorAll('.node').forEach(n=>n.classList.toggle('selected',n.dataset.id===id));const n=(proofData?.nodes||[]).find(x=>x.id===id);if(!n)return;
  $('inspectorTitle').textContent=short(n.statement,120);$('inspectorBody').innerHTML='<div class="fact"><span>Task ID</span><span>'+esc(n.id)+'</span></div><div class="fact"><span>Task state</span><span>'+esc(n.status||'unknown')+'</span></div><div class="fact"><span>Formal status</span><span>'+esc(n.formal_status||'unknown')+'</span></div><div class="fact"><span>Certificate</span><span>'+esc(n.certificate_status||'none')+'</span></div><p class="evidence math">'+esc(clean(n.last_evidence||'No evidence recorded.'))+'</p>';if(refit){const p=positions.get(id);if(p)centerOn(p.x+102,p.y+45)}
}
function applyView(){view.scale=Math.max(.35,Math.min(1.8,view.scale));$('graphScene').style.transform='translate('+view.x+'px,'+view.y+'px) scale('+view.scale+')'}
function centerOn(x,y){const vp=$('graphViewport');view.x=vp.clientWidth/2-x*view.scale;view.y=vp.clientHeight/2-y*view.scale;applyView()}
function fitActiveChain(){
  const pts=activeChain.map(id=>positions.get(id)).filter(Boolean),vp=$('graphViewport');if(!pts.length){resetGraph();return}
  const minX=Math.min(...pts.map(p=>p.x)),maxX=Math.max(...pts.map(p=>p.x+205)),minY=Math.min(...pts.map(p=>p.y)),maxY=Math.max(...pts.map(p=>p.y+112)),pad=55;
  view.scale=Math.max(.35,Math.min(1.15,Math.min((vp.clientWidth-pad*2)/(maxX-minX),(vp.clientHeight-pad*2)/(maxY-minY))));
  view.x=(vp.clientWidth-(minX+maxX)*view.scale)/2;view.y=(vp.clientHeight-(minY+maxY)*view.scale)/2;applyView()
}
function resetGraph(){view={x:20,y:20,scale:Math.min(1,$('graphViewport').clientWidth/1000)};applyView()}
function zoomBy(f){const vp=$('graphViewport'),cx=vp.clientWidth/2,cy=vp.clientHeight/2,wx=(cx-view.x)/view.scale,wy=(cy-view.y)/view.scale;view.scale*=f;view.scale=Math.max(.35,Math.min(1.8,view.scale));view.x=cx-wx*view.scale;view.y=cy-wy*view.scale;applyView()}
function renderRuns(runs){
  runs=(runs||[]).slice(0,12);$('historyCount').textContent=runs.length+' runs';const latest=runs[0];
  if(latest){$('latestAnswer').textContent=short(latest.mathematical_summary||latest.target_statement||latest.outcome,180);$('latestMeta').textContent=(latest.outcome||'completed').replaceAll('_',' ')+' · '+new Date((latest.updated_at||0)*1000).toLocaleString()}
  $('runTimeline').innerHTML=runs.length?runs.map((r,i)=>'<article class="run-item"><div class="run-meta"><span>'+esc(r.run_id||'completed run')+'</span><time>'+esc(new Date((r.updated_at||0)*1000).toLocaleString())+'</time></div><h3 class="math">'+esc(short(r.target_statement||r.target_obligation_id,220))+'</h3><p class="math">'+esc(clean(r.mathematical_summary||'No mathematical conclusion recorded.'))+'</p><small>'+esc(clean(r.gate_summary||r.outcome||'completed'))+'</small></article>').join(''):'<div class="run-item"><p>Waiting for completed proof runs.</p></div>';
}
function renderCollaboration(c){
  c=c||{};const html=(c.chain_participants||[]).map(m=>'<span class="chip">'+esc(m.name)+' · advisory</span>').join('')+(c.authoritative_gates||[]).map(g=>'<span class="chip gate">'+esc(g.name)+' · authoritative gate</span>').join('')+'<span class="chip warning">'+Number(c.open_slots||0)+' open slots · not executing</span>';$('collaborationChips').innerHTML=html;
}
function renderInfra(proof,nodes){
  $('ledgerVersion').textContent='v'+(proof.ledger_version??'—');$('nodeCount').textContent=(proof.nodes||[]).length;$('leafCount').textContent=proof.unresolved_leaves??'—';$('chainLength').textContent=(proof.longest_chain||[]).length;$('proofCount').textContent=proof.status_counts?.PROVED||0;
  const worker=(nodes||[]).find(n=>n.id==='allens-mini')?.prefill_worker,rate=Number(worker?.tokens_per_second);$('prefillRate').textContent=Number.isFinite(rate)?rate.toFixed(2):'—';
  const h=proof.review_health||{};$('infraDetail').textContent=(h.fallback_active?'Fallback active. ':'')+'Review gate: '+label(h.status||'unknown')+'. '+clean((h.latest_errors||[]).join('; ')||'No current review errors reported.');
}
async function json(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error(url+' returned '+r.status);return r.json()}
async function load(){
  try{const [proof,nodes]=await Promise.all([json('/v1/proof/progress'),json('/v1/network/nodes')]);proofData=proof;renderStatus(proof);renderPipeline(proof.live_execution);const obligation=proof.active_obligation||{};$('obligationAnswer').textContent=clean(obligation.statement||'No active obligation');$('obligationId').textContent=obligation.id||proof.active_leaf_id||'—';renderRuns(proof.reasoning_runs);renderCollaboration(proof.collaboration);renderInfra(proof,nodes);renderGraph(proof)}
  catch(e){const badge=$('backendBadge');badge.className='badge status-badge error';$('backendLabel').textContent='RECONNECTING';$('backendAnswer').textContent='Telemetry unavailable';$('backendDetail').textContent=String(e);$('freshnessTag').textContent='CONNECTION ERROR';console.error(e)}
}
document.querySelectorAll('.tab-btn').forEach((btn,i,all)=>{btn.onclick=()=>{all.forEach(x=>x.setAttribute('aria-selected',String(x===btn)));document.querySelectorAll('.tab-panel').forEach(p=>p.classList.toggle('active',p.id===btn.dataset.tab));if(btn.dataset.tab==='proofPanel')requestAnimationFrame(fitActiveChain)};btn.onkeydown=e=>{if(!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();const j=(i+(e.key==='ArrowRight'?1:-1)+all.length)%all.length;all[j].focus();all[j].click()}});
$('zoomIn').onclick=()=>zoomBy(1.2);$('zoomOut').onclick=()=>zoomBy(1/1.2);$('fitChain').onclick=fitActiveChain;$('resetView').onclick=resetGraph;
const vp=$('graphViewport');vp.addEventListener('pointerdown',e=>{if(e.target.closest('.node'))return;drag={x:e.clientX,y:e.clientY,vx:view.x,vy:view.y};vp.classList.add('dragging');vp.setPointerCapture(e.pointerId)});vp.addEventListener('pointermove',e=>{if(!drag)return;view.x=drag.vx+e.clientX-drag.x;view.y=drag.vy+e.clientY-drag.y;applyView()});vp.addEventListener('pointerup',()=>{drag=null;vp.classList.remove('dragging')});vp.addEventListener('wheel',e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();zoomBy(e.deltaY<0?1.1:1/1.1)},{passive:false});
setInterval(()=>{if(reduceMotion||!activeChain.length)return;hop=(hop+1)%activeChain.length;const p=positions.get(activeChain[hop]);if(p)$('riemann').style.transform='translate('+(p.x+170)+'px,'+(p.y-36)+'px)'},2600);
load();setInterval(load,1800);addEventListener('resize',()=>{if(document.querySelector('#proofPanel.active'))fitActiveChain()});
</script>
</body></html>`;
