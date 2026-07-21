import Lake

open Lake DSL

package «kakeya_lean_gate»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.32.0-rc1"

@[default_target]
lean_lib KakeyaLeanGate
