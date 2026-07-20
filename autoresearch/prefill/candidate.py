"""The only strategy file the AutoResearch agent may edit."""

CANDIDATE_ID = "baseline-v1"
TARGET_OBLIGATION_ID = "RH-C1"
HYPOTHESIS = (
    "Force each experiment to attack one unresolved proof obligation with a "
    "concrete construction or counterexample."
)
GENERATOR_DIRECTIVE = (
    "Focus on RH-C1. Propose one explicit non-circular operator definition, "
    "including domain, kernel/action, and the exact theorem still required."
)
CRITIC_DIRECTIVE = (
    "Attempt to falsify the proposed RH-C1 operator. Reject placeholders and "
    "identify the first invalid domain, self-adjointness, or spectrum step."
)
PREFILL_COMPUTE_CHUNK_TOKENS = 256
SNAPSHOT_MODE = "final_only"
MAX_SEGMENT_SECONDS = 300.0
REQUIRE_FULL_CONTEXT = True
ALLOW_FALLBACK = False
