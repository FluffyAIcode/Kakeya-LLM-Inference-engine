"""Fail-closed contracts for model-authored Lean declarations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


_SIGNATURE_BLOCK = re.compile(
    r"^### LEAN_SIGNATURE(?:\s+(\S+))?\s*$"
    r"\s*```lean\s*(?P<source>.*?)```",
    re.MULTILINE | re.DOTALL,
)
_FORBIDDEN = re.compile(
    r"(^|\s)(?:import|axiom|opaque|unsafe|macro|syntax|elab|run_cmd|run_tac|"
    r"set_option|def|abbrev|instance|structure|inductive|class|namespace|"
    r"section|variable|open|attribute)\b|#(?:eval|check|print|reduce)|"
    r"\b(?:IO|System|FilePath)\b",
    re.MULTILINE,
)
_DECLARATION = re.compile(
    r"\A(?P<kind>theorem|lemma)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_']*)"
    r"(?P<tail>.*?)\Z",
    re.DOTALL,
)
_DECLARATION_LINE = re.compile(
    r"^\s*(?:theorem|lemma|def|axiom|opaque|abbrev|instance|structure|"
    r"inductive|class)\b",
    re.MULTILINE,
)
_SCAFFOLD = re.compile(r"\s*:=\s*by\b")
_PLACEHOLDER = re.compile(
    r"\b(?:sorry|admit|placeholder|todo)\b|"
    r"\.\.\.|…|<[A-Za-z_][A-Za-z0-9_ -]{0,40}>|\?\w*",
    re.IGNORECASE,
)
_LATEX_ESCAPE = re.compile(r"\\[A-Za-z]+|[$]")


@dataclass(frozen=True)
class LeanDeclaration:
    kind: str
    name: str
    binders: str
    proposition: str
    source: str
    declaration_hash: str
    proposition_hash: str


@dataclass(frozen=True)
class LeanContractUser:
    role: str
    file: str
    policy: str
    contract_id: str
    contract_version: int
    fixtures: tuple[str, ...]


@dataclass(frozen=True)
class LeanContractDefinition:
    contract_id: str
    version: int
    schema_version: int
    allowed_kinds: tuple[str, ...]
    exact_scaffold: str
    forbidden_constructs: tuple[str, ...]
    signature_only: bool
    content_sha256: str


@dataclass(frozen=True)
class LeanSymbol:
    name: str
    lean_type: str
    source_symbol_hash: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeanSymbolTableDefinition:
    symbol_table_id: str
    version: int
    schema_version: int
    parent_statement_hash: str
    symbols: tuple[LeanSymbol, ...]
    content_sha256: str


def _contract_definition(
    *,
    name: str,
    version: int,
    signature_only: bool,
) -> LeanContractDefinition:
    content = {
        "name": name,
        "version": version,
        "schema_version": 1,
        "allowed_kinds": ["theorem", "lemma"],
        "exact_scaffold": ":= by",
        "forbidden_constructs": [
            "prose",
            "markdown_fence",
            "commands",
            "multiple_declarations",
            "placeholders",
            "latex_escapes",
            "proof_body" if signature_only else "missing_proof_body",
        ],
        "signature_only": signature_only,
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    return LeanContractDefinition(
        contract_id=f"{name}-{digest[:16]}",
        version=version,
        schema_version=1,
        allowed_kinds=("theorem", "lemma"),
        exact_scaffold=":= by",
        forbidden_constructs=tuple(content["forbidden_constructs"]),
        signature_only=signature_only,
        content_sha256=digest,
    )


_SIGNATURE_CONTRACT_DEFINITION = _contract_definition(
    name="lean-signature",
    version=1,
    signature_only=True,
)
_PROOF_CONTRACT_DEFINITION = _contract_definition(
    name="lean-proof",
    version=1,
    signature_only=False,
)
LEAN_CONTRACT_REGISTRY = {
    (contract.contract_id, contract.version): contract
    for contract in (
        _SIGNATURE_CONTRACT_DEFINITION,
        _PROOF_CONTRACT_DEFINITION,
    )
}
LEAN_SYMBOL_TABLE_REGISTRY: dict[
    tuple[str, int],
    LeanSymbolTableDefinition,
] = {}

_AUDITED_SYMBOL_NAMES = {
    r"\epsilon": ("epsilon", r"\epsilon"),
    "p": ("p",),
    r"\rho_c": ("rho_c", r"\rho_c"),
    r"\{z_n\}": ("z",),
    r"\rho": ("rho", r"\rho"),
    "s": ("s",),
    "s_0": ("s0",),
    r"\delta": ("delta", r"\delta"),
    "m": ("m",),
    "f(s)": ("f",),
    "growth order": ("growthOrder",),
}
_AUDITED_TYPE_MAP = {
    "real constant": "ℝ",
    "integer (genus)": "ℕ",
    "critical density threshold": "ℝ",
    "sequence of complex numbers": "ℕ → ℂ",
    "density of sequence": "ℝ",
    "complex variable": "ℂ",
    "singularity point": "ℂ",
    "neighborhood radius": "ℝ",
    "integer constant": "ℤ",
    "meromorphic/analytic function": "ℂ → ℂ",
    "order of growth (complex analysis)": "(ℂ → ℂ) → ℝ",
}
_AUDITED_SYMBOL_ID_MAP = {
    "SYM_EPSILON": ("epsilon", "ℝ"),
    "SYM_GENUS_P": ("p", "ℕ"),
    "SYM_CRITICAL_DENSITY": ("rho_c", "ℝ"),
    "SYM_SEQUENCE": ("z", "ℕ → ℂ"),
    "SYM_SEQUENCE_DENSITY": ("rho", "ℝ"),
    "SYM_COMPLEX_VARIABLE": ("s", "ℂ"),
    "SYM_POLE_LOCATION": ("s0", "ℂ"),
    "SYM_POLE_MULTIPLICITY": ("m", "ℤ"),
    "SYM_FUNCTION": ("f", "ℂ → ℂ"),
}
_MISSING_DEFINITION_ID_MAP = {
    "DEF_EPSILON": ("positiveEpsilon", "ℝ → Prop"),
    "DEF_GENUS": ("fixedGenus", "ℕ → Prop"),
    "DEF_CRITICAL_DENSITY": ("criticalDensity", "ℝ"),
    "DEF_SEQUENCE_DENSITY": ("density", "(ℕ → ℂ) → ℝ"),
    "DEF_SERIES_CONVERGENCE": (
        "localConvergence",
        "(ℕ → ℂ) → ℂ → ℤ → ℝ → Prop",
    ),
    "DEF_POLE_NEIGHBORHOOD": ("poleNeighborhood", "ℂ → ℝ → Prop"),
    "DEF_FUNCTION_BINDING": ("canonicalProductBinding", "(ℕ → ℂ) → (ℂ → ℂ) → Prop"),
    "DEF_GROWTH_ORDER": ("growthConstraint", "(ℂ → ℂ) → ℕ → Prop"),
}


def register_lean_symbol_table(
    definitions: list[dict],
    *,
    parent_statement_hash: str,
    missing_definitions: list[dict] | None = None,
    version: int = 1,
) -> LeanSymbolTableDefinition:
    symbols = []
    seen_symbol_ids = set()
    for item in definitions:
        symbol_ids = item.get("symbol_ids")
        if isinstance(symbol_ids, list):
            for symbol_id in symbol_ids:
                symbol_id = str(symbol_id)
                if symbol_id in seen_symbol_ids:
                    continue
                if symbol_id not in _AUDITED_SYMBOL_ID_MAP:
                    raise ValueError(
                        f"unregistered audited symbol ID: {symbol_id}",
                    )
                name, lean_type = _AUDITED_SYMBOL_ID_MAP[symbol_id]
                symbols.append(LeanSymbol(
                    name=name,
                    lean_type=lean_type,
                    source_symbol_hash=hashlib.sha256(
                        symbol_id.encode(),
                    ).hexdigest(),
                ))
                seen_symbol_ids.add(symbol_id)
            continue
        raw_symbol = str(item.get("symbol", ""))
        raw_type = str(item.get("type", ""))
        if raw_symbol not in _AUDITED_SYMBOL_NAMES:
            raise ValueError(f"unregistered audited symbol: {raw_symbol}")
        if raw_type not in _AUDITED_TYPE_MAP:
            raise ValueError(f"unregistered audited symbol type: {raw_type}")
        names = _AUDITED_SYMBOL_NAMES[raw_symbol]
        symbols.append(LeanSymbol(
            name=names[0],
            lean_type=_AUDITED_TYPE_MAP[raw_type],
            source_symbol_hash=hashlib.sha256(raw_symbol.encode()).hexdigest(),
            aliases=tuple(names[1:]),
        ))
    missing_symbol_specs = {
        "L1": ("density", "(ℕ → ℂ) → ℝ"),
        "L2": (
            "localConvergence",
            "(ℕ → ℂ) → ℂ → ℤ → ℝ → Prop",
        ),
        "L3": ("growthConstraint", "(ℂ → ℂ) → ℕ → Prop"),
    }
    for item in missing_definitions or []:
        definition_id = str(item.get("definition_id", ""))
        if definition_id:
            if definition_id not in _MISSING_DEFINITION_ID_MAP:
                raise ValueError(
                    f"unregistered missing definition ID: {definition_id}",
                )
            name, lean_type = _MISSING_DEFINITION_ID_MAP[definition_id]
            symbols.append(LeanSymbol(
                name=name,
                lean_type=lean_type,
                source_symbol_hash=hashlib.sha256(
                    definition_id.encode(),
                ).hexdigest(),
            ))
            continue
        label = str(item.get("obligation_label", ""))
        if label not in missing_symbol_specs:
            raise ValueError(f"unregistered missing definition label: {label}")
        name, lean_type = missing_symbol_specs[label]
        symbols.append(LeanSymbol(
            name=name,
            lean_type=lean_type,
            source_symbol_hash=hashlib.sha256(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            ).hexdigest(),
        ))
    content = {
        "version": version,
        "schema_version": 1,
        "parent_statement_hash": parent_statement_hash,
        "symbols": [
            {
                "name": symbol.name,
                "lean_type": symbol.lean_type,
                "source_symbol_hash": symbol.source_symbol_hash,
                "aliases": list(symbol.aliases),
            }
            for symbol in symbols
        ],
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    table = LeanSymbolTableDefinition(
        symbol_table_id=f"lean-symbols-{digest[:16]}",
        version=version,
        schema_version=1,
        parent_statement_hash=parent_statement_hash,
        symbols=tuple(symbols),
        content_sha256=digest,
    )
    LEAN_SYMBOL_TABLE_REGISTRY[(table.symbol_table_id, table.version)] = table
    return table


def resolve_lean_symbol_table(
    symbol_table_id: str,
    version: int,
) -> LeanSymbolTableDefinition:
    try:
        return LEAN_SYMBOL_TABLE_REGISTRY[
            (str(symbol_table_id), int(version))
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("unknown or stale Lean symbol table ID/version") from exc


def lean_symbol_semantic_hash(
    text: str,
    table: LeanSymbolTableDefinition,
) -> str:
    canonical = str(text)
    aliases = sorted(
        (
            (alias, symbol.name)
            for symbol in table.symbols
            for alias in symbol.aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, name in aliases:
        canonical = canonical.replace(alias, name)
    return hashlib.sha256(canonical.encode()).hexdigest()


def normalize_registered_latex_identifiers(
    text: str,
    table: LeanSymbolTableDefinition,
) -> str:
    """Normalize only audited identifier tokens; reject semantic LaTeX."""
    original = str(text)
    normalized = original
    aliases = sorted(
        (
            (alias, symbol.name)
            for symbol in table.symbols
            for alias in symbol.aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, name in aliases:
        pattern = (
            rf"(?<![A-Za-z0-9_\\]){re.escape(alias)}"
            rf"(?![A-Za-z0-9_])"
        )
        normalized = re.sub(pattern, name, normalized)
    remaining = re.findall(r"\\[A-Za-z]+|\\[{}]", normalized)
    if remaining:
        raise ValueError(
            "unapproved LaTeX command(s): " + ", ".join(sorted(set(remaining))),
        )
    if (
        lean_symbol_semantic_hash(original, table)
        != lean_symbol_semantic_hash(normalized, table)
    ):
        raise ValueError("identifier normalization changed semantic hash")
    return normalized


def lint_lean_model_prompt(messages: list[dict]) -> None:
    for message in messages:
        content = str(message.get("content", ""))
        escapes = re.findall(r"\\[A-Za-z]+|\\[{}]", content)
        if escapes:
            raise ValueError(
                "Lean-producing prompt contains raw LaTeX: "
                + ", ".join(sorted(set(escapes))),
            )


def resolve_lean_contract(
    contract_id: str,
    version: int,
    *,
    signature_only: bool | None = None,
) -> LeanContractDefinition:
    try:
        contract = LEAN_CONTRACT_REGISTRY[(str(contract_id), int(version))]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("unknown or stale Lean contract ID/version") from exc
    if signature_only is not None and contract.signature_only != signature_only:
        raise ValueError("Lean contract policy mismatch")
    return contract


def lean_signature_contract_ref() -> dict[str, object]:
    return {
        "contract_id": _SIGNATURE_CONTRACT_DEFINITION.contract_id,
        "version": _SIGNATURE_CONTRACT_DEFINITION.version,
    }


def lean_proof_contract_ref() -> dict[str, object]:
    return {
        "contract_id": _PROOF_CONTRACT_DEFINITION.contract_id,
        "version": _PROOF_CONTRACT_DEFINITION.version,
    }


LEAN_SIGNATURE_FIXTURES = (
    "zero_declarations",
    "multiple_declarations",
    "prose",
    "fence",
    "def",
    "forbidden_command",
    "missing_scaffold",
    "duplicate_scaffold",
    "proof_body",
    "placeholder",
    "unknown_type",
    "changed_theorem_target",
    "latex_escape",
    "valid_multiline_binders",
    "latest_parent_prose",
    "latest_child_missing_scaffold",
    "latest_reduction_hash",
    "latex_identifier_epsilon",
    "latex_identifier_rho",
    "latex_identifier_delta",
    "forbidden_latex_sum",
    "forbidden_latex_frac",
    "forbidden_latex_set",
    "mixed_prose_lean",
    "unknown_latex_command",
    "escaped_json_backslashes",
    "valid_unicode_ascii",
)

# CI treats this as the exhaustive registry of model roles that may carry Lean.
LEAN_CONTRACT_USER_REGISTRY = {
    "formalizer": LeanContractUser(
        "formalizer",
        "scripts/agent_gan_repl.py",
        "signature",
        _SIGNATURE_CONTRACT_DEFINITION.contract_id,
        _SIGNATURE_CONTRACT_DEFINITION.version,
        LEAN_SIGNATURE_FIXTURES,
    ),
    "prover": LeanContractUser(
        "prover",
        "scripts/agent_gan_repl.py",
        "complete_proof",
        _PROOF_CONTRACT_DEFINITION.contract_id,
        _PROOF_CONTRACT_DEFINITION.version,
        LEAN_SIGNATURE_FIXTURES,
    ),
    "premise_auditor": LeanContractUser(
        "premise_auditor",
        "scripts/agent_gan_repl.py",
        "reject_unbound_lean",
        _PROOF_CONTRACT_DEFINITION.contract_id,
        _PROOF_CONTRACT_DEFINITION.version,
        LEAN_SIGNATURE_FIXTURES,
    ),
}


@dataclass(frozen=True)
class LeanSignatureResult:
    source: str
    signature_hash: str
    ok: bool
    status: str = "FORMALIZED"
    error: str = ""
    attempts: int = 1
    elapsed_s: float = 0.0
    output: str = ""
    normalized_source: str = ""
    declaration_name: str = ""
    binders: str = ""
    proposition: str = ""
    proposition_hash: str = ""


@dataclass(frozen=True)
class _LeanRun:
    returncode: int | None
    timed_out: bool
    elapsed_s: float
    output: str


class LeanSignatureContract:
    """Syntactic contract shared by every model-authored Lean declaration."""

    allowed_kinds = ("theorem", "lemma")
    exact_scaffold = ":= by"

    @staticmethod
    def _split_tail(tail: str) -> tuple[str, str]:
        depth = 0
        pairs = {"(": ")", "{": "}", "[": "]"}
        closers = set(pairs.values())
        quote = False
        escape = False
        for index, char in enumerate(tail):
            if quote:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    quote = False
                continue
            if char == '"':
                quote = True
            elif char in pairs:
                depth += 1
            elif char in closers:
                depth -= 1
                if depth < 0:
                    raise ValueError("unbalanced declaration binders")
            elif char == ":" and depth == 0:
                return tail[:index].strip(), tail[index + 1:].strip()
        raise ValueError("declaration is missing a proposition separator `:`")

    def parse(
        self,
        source: str,
        *,
        signature_only: bool,
        allow_scaffold_normalization: bool = False,
        expected: dict[str, str] | None = None,
    ) -> LeanDeclaration:
        raw = str(source)
        if not raw.strip():
            raise ValueError("empty Lean declaration")
        if len(raw) > 12_000:
            raise ValueError("Lean declaration is too large")
        if "```" in raw:
            raise ValueError("Markdown fences are forbidden")
        if "--" in raw or "/-" in raw or "-/" in raw:
            raise ValueError("comments or prose are forbidden")
        if _LATEX_ESCAPE.search(raw):
            raise ValueError("LaTeX escapes are forbidden in Lean source")
        if _FORBIDDEN.search(raw):
            raise ValueError("forbidden Lean command in generated declaration")
        if _PLACEHOLDER.search(raw):
            raise ValueError("Lean declaration contains a placeholder")
        if len(_DECLARATION_LINE.findall(raw)) != 1:
            raise ValueError("expected exactly one theorem or lemma declaration")

        text = raw.strip()
        match = _DECLARATION.fullmatch(text)
        if match is None:
            raise ValueError(
                "artifact must contain only one theorem or lemma declaration",
            )
        scaffold_matches = list(_SCAFFOLD.finditer(match.group("tail")))
        if len(scaffold_matches) > 1:
            raise ValueError("duplicate `:= by` proof scaffold")
        if not scaffold_matches:
            if not allow_scaffold_normalization:
                raise ValueError(
                    "theorem signature must end with `:= by` proof scaffold",
                )
            declaration_tail = match.group("tail").rstrip()
            proof_body = ""
        else:
            scaffold = scaffold_matches[0]
            declaration_tail = match.group("tail")[:scaffold.start()].rstrip()
            proof_body = match.group("tail")[scaffold.end():]
            if signature_only and proof_body.strip():
                raise ValueError("signature proof scaffold must have no proof body")
            if not signature_only and not proof_body.strip():
                raise ValueError("complete proof must contain a proof body")

        binders, proposition = self._split_tail(declaration_tail)
        if not proposition:
            raise ValueError("Lean declaration proposition is empty")
        kind = match.group("kind")
        name = match.group("name")
        if expected:
            comparisons = {
                "kind": kind,
                "name": name,
                "binders": binders,
                "proposition": proposition,
            }
            for field, actual in comparisons.items():
                if field in expected and str(expected[field]).strip() != actual:
                    raise ValueError(f"{field} changed during signature transport")

        declaration_core = f"{kind} {name}"
        if binders:
            declaration_core += f" {binders}"
        declaration_core += f" : {proposition}"
        normalized = (
            declaration_core + " := by"
            if signature_only
            else text
        )
        declaration_hash = hashlib.sha256(
            " ".join(declaration_core.split()).encode(),
        ).hexdigest()
        proposition_hash = hashlib.sha256(proposition.encode()).hexdigest()
        normalized_parsed = _DECLARATION.fullmatch(normalized)
        if normalized_parsed is None:
            raise ValueError("normalized Lean declaration is malformed")
        normalized_tail = normalized_parsed.group("tail")
        normalized_scaffold = _SCAFFOLD.search(normalized_tail)
        assert normalized_scaffold is not None
        normalized_binders, normalized_proposition = self._split_tail(
            normalized_tail[:normalized_scaffold.start()].rstrip(),
        )
        if (
            normalized_parsed.group("name") != name
            or normalized_binders != binders
            or normalized_proposition != proposition
            or hashlib.sha256(normalized_proposition.encode()).hexdigest()
            != proposition_hash
        ):
            raise ValueError("mechanical normalization changed mathematical content")
        return LeanDeclaration(
            kind=kind,
            name=name,
            binders=binders,
            proposition=proposition,
            source=normalized,
            declaration_hash=declaration_hash,
            proposition_hash=proposition_hash,
        )

    def normalize_signature(
        self,
        source: str,
        *,
        expected: dict[str, str] | None = None,
    ) -> LeanDeclaration:
        return self.parse(
            source,
            signature_only=True,
            allow_scaffold_normalization=True,
            expected=expected,
        )

    def validate_signature(
        self,
        source: str,
        *,
        expected: dict[str, str] | None = None,
    ) -> LeanDeclaration:
        return self.parse(
            source,
            signature_only=True,
            allow_scaffold_normalization=False,
            expected=expected,
        )

    def validate_proof(self, source: str) -> LeanDeclaration:
        return self.parse(source, signature_only=False)

    def example(self, name: str = "contractExample") -> dict[str, str]:
        source = f"theorem {name} (P : Prop) (h : P) : P := by"
        parsed = self.validate_signature(source)
        return {
            "kind": parsed.kind,
            "name": parsed.name,
            "binders": parsed.binders,
            "proposition": parsed.proposition,
            "source": parsed.source,
        }


LEAN_SIGNATURE_CONTRACT = LeanSignatureContract()


def extract_lean_signature_blocks(text: str) -> list[tuple[str, str]]:
    return [
        ((match.group(1) or "").strip(), match.group("source").strip())
        for match in _SIGNATURE_BLOCK.finditer(text)
    ]


def _signature_only(source: str) -> str:
    match = re.search(r"\s*:=\s*by\b", source)
    return source[:match.start()].strip() if match else source.strip()


def lean_theorem_signature_hash(source: str) -> str:
    try:
        return LEAN_SIGNATURE_CONTRACT.normalize_signature(
            source,
        ).declaration_hash
    except ValueError:
        signature = " ".join(_signature_only(source).split())
        return hashlib.sha256(signature.encode()).hexdigest() if signature else ""


def normalize_lean_signature(
    source: str,
    *,
    expected: dict[str, str] | None = None,
) -> LeanDeclaration:
    """Add/canonicalize only the terminal scaffold, never mathematics."""
    return LEAN_SIGNATURE_CONTRACT.normalize_signature(source, expected=expected)


def lean_signature_contract_example(name: str = "contractExample") -> dict[str, str]:
    return LEAN_SIGNATURE_CONTRACT.example(name)


def _run_lean(
    content: str,
    *,
    project_root: Path,
    timeout_s: float,
) -> _LeanRun:
    started = time.monotonic()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".lean",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(content)
        path = Path(handle.name)
    process = None
    try:
        process = subprocess.Popen(
            ["lake", "env", "lean", str(path)],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout_s)
            return _LeanRun(
                process.returncode,
                False,
                time.monotonic() - started,
                output or "",
            )
        except subprocess.TimeoutExpired as exc:
            partial = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
            remainder, _ = process.communicate()
            return _LeanRun(
                None,
                True,
                time.monotonic() - started,
                partial + (remainder or ""),
            )
    except OSError as exc:
        return _LeanRun(
            None,
            False,
            time.monotonic() - started,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        path.unlink(missing_ok=True)


def warm_lean_environment(
    project_root: Path,
    *,
    timeout_s: float = 120.0,
) -> LeanSignatureResult:
    source = "theorem kakeyaLeanWarmup : True := by trivial"
    content = (
        "import KakeyaLeanGate\n\n"
        "set_option autoImplicit false\n\n"
        + source
        + "\n"
    )
    run = _run_lean(
        content,
        project_root=project_root,
        timeout_s=timeout_s,
    )
    if run.timed_out:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="TYPECHECK_TIMEOUT",
            error=f"Lean warmup timed out after {timeout_s:.1f}s",
            elapsed_s=run.elapsed_s,
            output=run.output,
        )
    if run.returncode != 0:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="ENVIRONMENT_FAILED",
            error=f"Lean warmup failed: {run.output[-2000:]}",
            elapsed_s=run.elapsed_s,
            output=run.output,
        )
    return LeanSignatureResult(
        source,
        "",
        True,
        status="ENVIRONMENT_READY",
        elapsed_s=run.elapsed_s,
        output=run.output,
    )


def validate_lean_signature(
    source: str,
    *,
    project_root: Path,
    timeout_s: float = 45.0,
    retry_timeout_s: float = 120.0,
) -> LeanSignatureResult:
    source = source.strip()
    try:
        declaration = LEAN_SIGNATURE_CONTRACT.validate_signature(source)
    except ValueError as exc:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="CONTRACT_FAILED",
            error=str(exc),
        )
    signature_hash = declaration.declaration_hash
    content = (
        "import KakeyaLeanGate\n\n"
        "set_option autoImplicit false\n\n"
        + declaration.source
        + "\n  sorry\n"
    )
    first = _run_lean(
        content,
        project_root=project_root,
        timeout_s=timeout_s,
    )
    attempts = 1
    total_elapsed = first.elapsed_s
    output = first.output
    run = first
    if first.timed_out:
        warmup = warm_lean_environment(
            project_root,
            timeout_s=retry_timeout_s,
        )
        total_elapsed += warmup.elapsed_s
        output += warmup.output
        if not warmup.ok:
            return LeanSignatureResult(
                source,
                signature_hash,
                False,
                status=warmup.status,
                error=warmup.error,
                attempts=1,
                elapsed_s=total_elapsed,
                output=output,
                normalized_source=declaration.source,
                declaration_name=declaration.name,
                binders=declaration.binders,
                proposition=declaration.proposition,
                proposition_hash=declaration.proposition_hash,
            )
        run = _run_lean(
            content,
            project_root=project_root,
            timeout_s=retry_timeout_s,
        )
        attempts = 2
        total_elapsed += run.elapsed_s
        output += run.output
    if run.timed_out:
        return LeanSignatureResult(
            source,
            signature_hash,
            False,
            status="TYPECHECK_TIMEOUT",
            error=(
                f"Lean typecheck timed out after {attempts} attempts "
                f"({timeout_s:.1f}s/{retry_timeout_s:.1f}s)"
            ),
            attempts=attempts,
            elapsed_s=total_elapsed,
            output=output,
            normalized_source=declaration.source,
            declaration_name=declaration.name,
            binders=declaration.binders,
            proposition=declaration.proposition,
            proposition_hash=declaration.proposition_hash,
        )
    if run.returncode != 0:
        return LeanSignatureResult(
            source,
            signature_hash,
            False,
            status="TYPECHECK_FAILED",
            error=f"Lean typecheck failed: {run.output[-2000:]}",
            attempts=attempts,
            elapsed_s=total_elapsed,
            output=output,
            normalized_source=declaration.source,
            declaration_name=declaration.name,
            binders=declaration.binders,
            proposition=declaration.proposition,
            proposition_hash=declaration.proposition_hash,
        )
    return LeanSignatureResult(
        source,
        signature_hash,
        True,
        status="FORMALIZED",
        attempts=attempts,
        elapsed_s=total_elapsed,
        output=output,
        normalized_source=declaration.source,
        declaration_name=declaration.name,
        binders=declaration.binders,
        proposition=declaration.proposition,
        proposition_hash=declaration.proposition_hash,
    )


def validate_lean_proof(
    source: str,
    *,
    project_root: Path,
    timeout_s: float = 45.0,
) -> LeanSignatureResult:
    """Compile one complete theorem without sorry/admit or added axioms."""
    source = source.strip()
    try:
        declaration = LEAN_SIGNATURE_CONTRACT.validate_proof(source)
    except ValueError as exc:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="CONTRACT_FAILED",
            error=str(exc),
        )
    proof_hash = hashlib.sha256(source.encode()).hexdigest()
    run = _run_lean(
        "import KakeyaLeanGate\n\nset_option autoImplicit false\n\n"
        + source
        + "\n",
        project_root=project_root,
        timeout_s=timeout_s,
    )
    if run.timed_out:
        return LeanSignatureResult(
            source,
            proof_hash,
            False,
            status="TYPECHECK_TIMEOUT",
            error=f"Lean proof timed out after {timeout_s:.1f}s",
            elapsed_s=run.elapsed_s,
            output=run.output,
            normalized_source=declaration.source,
            declaration_name=declaration.name,
            binders=declaration.binders,
            proposition=declaration.proposition,
            proposition_hash=declaration.proposition_hash,
        )
    if run.returncode != 0:
        return LeanSignatureResult(
            source,
            proof_hash,
            False,
            status="TYPECHECK_FAILED",
            error=f"Lean proof failed: {run.output[-2000:]}",
            elapsed_s=run.elapsed_s,
            output=run.output,
            normalized_source=declaration.source,
            declaration_name=declaration.name,
            binders=declaration.binders,
            proposition=declaration.proposition,
            proposition_hash=declaration.proposition_hash,
        )
    return LeanSignatureResult(
        source,
        proof_hash,
        True,
        status="PROVED",
        elapsed_s=run.elapsed_s,
        output=run.output,
        normalized_source=declaration.source,
        declaration_name=declaration.name,
        binders=declaration.binders,
        proposition=declaration.proposition,
        proposition_hash=declaration.proposition_hash,
    )
