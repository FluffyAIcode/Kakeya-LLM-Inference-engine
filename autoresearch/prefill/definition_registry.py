"""Host-owned definition choices for the active proof obligation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

from autoresearch.prefill.typed_transport import DecodedRoleFields, host_artifact


REGISTRY_VERSION = 2


@dataclass(frozen=True)
class DefinitionChoice:
    definition_id: str
    content_ref: str
    label: str
    symbol_ids: tuple[str, ...] = ()
    domain_ids: tuple[str, ...] = ()
    topology_ids: tuple[str, ...] = ()
    required_type_id: str = ""


@dataclass(frozen=True)
class DefinitionChoiceRegistry:
    target_ref: str
    symbols: Mapping[str, str]
    domains: Mapping[str, str]
    topologies: Mapping[str, str]
    definitions: Mapping[str, DefinitionChoice]
    registry_hash: str

    @property
    def registered_output_choices(self) -> dict[str, tuple[str, ...]]:
        return {
            "target_ref": (self.target_ref,),
            "symbol_id": tuple(self.symbols),
            "domain_id": tuple(self.domains),
            "topology_id": tuple(self.topologies),
            "definition_id": tuple(self.definitions),
            "missing_definition_id": tuple(self.definitions),
        }


_SYMBOLS = {
    "SYM_EPSILON": "content:symbol:epsilon",
    "SYM_GENUS_P": "content:symbol:genus_p",
    "SYM_CRITICAL_DENSITY": "content:symbol:critical_density",
    "SYM_SEQUENCE": "content:symbol:zero_sequence",
    "SYM_SEQUENCE_DENSITY": "content:symbol:sequence_density",
    "SYM_COMPLEX_VARIABLE": "content:symbol:complex_variable",
    "SYM_POLE_LOCATION": "content:symbol:pole_location",
    "SYM_POLE_MULTIPLICITY": "content:symbol:pole_multiplicity",
    "SYM_FUNCTION": "content:symbol:entire_function",
    "SYM_ZETA": "content:symbol:riemann_zeta",
    "SYM_XI": "content:symbol:completed_xi",
    "SYM_LOG_DERIVATIVE": "content:symbol:zeta_log_derivative",
    "SYM_ZERO_SPECTRUM_MAP": "content:symbol:zero_spectrum_map",
}
_DOMAINS = {
    "DOM_POSITIVE_REAL": "content:domain:positive_real",
    "DOM_NATURAL": "content:domain:natural",
    "DOM_COMPLEX": "content:domain:complex",
    "DOM_COMPLEX_SEQUENCE": "content:domain:complex_sequence",
}
_TOPOLOGIES = {
    "TOP_DELTA_NEIGHBORHOOD": "content:topology:delta_neighborhood",
    "TOP_LOCALLY_UNIFORM": "content:topology:locally_uniform",
    "TOP_POINTWISE": "content:topology:pointwise",
}
_DEFINITIONS = {
    "DEF_EPSILON": DefinitionChoice(
        "DEF_EPSILON", "content:definition:epsilon", "Epsilon",
        ("SYM_EPSILON",), ("DOM_POSITIVE_REAL",),
    ),
    "DEF_GENUS": DefinitionChoice(
        "DEF_GENUS", "content:definition:genus", "Fixed genus",
        ("SYM_GENUS_P",), ("DOM_NATURAL",),
    ),
    "DEF_CRITICAL_DENSITY": DefinitionChoice(
        "DEF_CRITICAL_DENSITY",
        "content:definition:critical_density",
        "Critical density",
        ("SYM_CRITICAL_DENSITY",), ("DOM_POSITIVE_REAL",),
        required_type_id="TYPE_DENSITY_THRESHOLD",
    ),
    "DEF_SEQUENCE_DENSITY": DefinitionChoice(
        "DEF_SEQUENCE_DENSITY",
        "content:definition:sequence_density",
        "Sequence density",
        ("SYM_SEQUENCE", "SYM_SEQUENCE_DENSITY"),
        ("DOM_COMPLEX_SEQUENCE", "DOM_POSITIVE_REAL"),
        required_type_id="TYPE_SEQUENCE_DENSITY",
    ),
    "DEF_SERIES_CONVERGENCE": DefinitionChoice(
        "DEF_SERIES_CONVERGENCE",
        "content:definition:series_convergence",
        "Series convergence",
        ("SYM_SEQUENCE", "SYM_COMPLEX_VARIABLE"),
        ("DOM_COMPLEX_SEQUENCE", "DOM_COMPLEX"),
        ("TOP_LOCALLY_UNIFORM", "TOP_POINTWISE"),
        "TYPE_CONVERGENCE_MODE",
    ),
    "DEF_POLE_NEIGHBORHOOD": DefinitionChoice(
        "DEF_POLE_NEIGHBORHOOD",
        "content:definition:pole_neighborhood",
        "Pole neighborhood",
        ("SYM_POLE_LOCATION", "SYM_POLE_MULTIPLICITY"),
        ("DOM_COMPLEX",),
        ("TOP_DELTA_NEIGHBORHOOD",),
        "TYPE_PUNCTURED_OR_FULL_NEIGHBORHOOD",
    ),
    "DEF_FUNCTION_BINDING": DefinitionChoice(
        "DEF_FUNCTION_BINDING",
        "content:definition:function_binding",
        "Function binding",
        ("SYM_FUNCTION", "SYM_SEQUENCE"),
        ("DOM_COMPLEX",),
        required_type_id="TYPE_CANONICAL_PRODUCT_BINDING",
    ),
    "DEF_GROWTH_ORDER": DefinitionChoice(
        "DEF_GROWTH_ORDER",
        "content:definition:growth_order",
        "Growth order",
        ("SYM_FUNCTION", "SYM_GENUS_P"),
        ("DOM_COMPLEX", "DOM_NATURAL"),
        required_type_id="TYPE_ENTIRE_FUNCTION_ORDER",
    ),
    "DEF_ZETA_LOG_DERIVATIVE": DefinitionChoice(
        "DEF_ZETA_LOG_DERIVATIVE",
        "content:definition:zeta_log_derivative",
        "Zeta logarithmic derivative",
        ("SYM_ZETA", "SYM_LOG_DERIVATIVE"),
        ("DOM_COMPLEX",),
        required_type_id="TYPE_MEROMORPHIC_LOG_DERIVATIVE",
    ),
    "DEF_COMPLETED_XI": DefinitionChoice(
        "DEF_COMPLETED_XI",
        "content:definition:completed_xi",
        "Completed xi function",
        ("SYM_XI",),
        ("DOM_COMPLEX",),
        required_type_id="TYPE_ENTIRE_COMPLETED_ZETA",
    ),
    "DEF_ZERO_SPECTRUM_BINDING": DefinitionChoice(
        "DEF_ZERO_SPECTRUM_BINDING",
        "content:definition:zero_spectrum_binding",
        "Non-circular zero to spectrum binding",
        ("SYM_ZETA", "SYM_XI", "SYM_ZERO_SPECTRUM_MAP"),
        ("DOM_COMPLEX",),
        required_type_id="TYPE_ZERO_SPECTRUM_BIJECTION",
    ),
}

_TARGET_TEMPLATES = {
    "DEF_EPSILON": ("epsilon", "error", "neighborhood"),
    "DEF_GENUS": ("genus",),
    "DEF_CRITICAL_DENSITY": ("critical density", "density threshold"),
    "DEF_SEQUENCE_DENSITY": ("density", "counting function"),
    "DEF_SERIES_CONVERGENCE": ("convergence", "converge", "series", "sum"),
    "DEF_POLE_NEIGHBORHOOD": ("pole", "singularity", "neighborhood"),
    "DEF_FUNCTION_BINDING": ("weierstrass", "canonical product"),
    "DEF_GROWTH_ORDER": ("growth order", "growth", "genus"),
    "DEF_ZETA_LOG_DERIVATIVE": (
        "logarithmic-derivative", "logarithmic derivative", "zeta'",
    ),
    "DEF_COMPLETED_XI": ("xi(s)", "xi zeros", "completed xi"),
    "DEF_ZERO_SPECTRUM_BINDING": (
        "zero/spectrum", "spectrum mapping", "spectral", "circular",
    ),
}


def build_definition_choice_registry(
    target_ref: str,
    target_statement: str = "",
) -> DefinitionChoiceRegistry:
    """Derive active choices only from the exact target statement.

    The catalog is reusable proof-domain metadata; it is never an active global
    default. In particular, density/genus choices cannot enter another target.
    """
    normalized = " ".join(str(target_statement).lower().split())
    definitions = {
        key: value for key, value in _DEFINITIONS.items()
        if any(marker in normalized for marker in _TARGET_TEMPLATES[key])
    }
    symbol_ids = {
        item for choice in definitions.values() for item in choice.symbol_ids
    }
    domain_ids = {
        item for choice in definitions.values() for item in choice.domain_ids
    }
    topology_ids = {
        item for choice in definitions.values() for item in choice.topology_ids
    }
    symbols = {key: value for key, value in _SYMBOLS.items() if key in symbol_ids}
    domains = {key: value for key, value in _DOMAINS.items() if key in domain_ids}
    topologies = {
        key: value for key, value in _TOPOLOGIES.items()
        if key in topology_ids
    }
    canonical = json.dumps({
        "version": REGISTRY_VERSION,
        "target_ref": target_ref,
        "target_statement_hash": hashlib.sha256(
            str(target_statement).encode()
        ).hexdigest(),
        "symbols": symbols,
        "domains": domains,
        "topologies": topologies,
        "definitions": {
            key: asdict(value) for key, value in definitions.items()
        },
    }, sort_keys=True, separators=(",", ":")).encode()
    return DefinitionChoiceRegistry(
        target_ref=target_ref,
        symbols=symbols,
        domains=domains,
        topologies=topologies,
        definitions=definitions,
        registry_hash=hashlib.sha256(canonical).hexdigest(),
    )


def serialize_definition_audit(
    decoded: DecodedRoleFields,
    registry: DefinitionChoiceRegistry,
    *,
    target_obligation_id: str,
    parent_statement_hash: str,
    root_goal_hash: str,
    producer_run_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve IDs and serialize the authoritative audit entirely on the Host."""
    values = decoded.values
    outcome = str(values["audit_outcome"])
    defined_ids = tuple(dict.fromkeys(values["definition_id"]))
    missing_ids = tuple(dict.fromkeys(values["missing_definition_id"]))

    if set(defined_ids) & set(missing_ids):
        raise ValueError("definition choice cannot be both defined and missing")
    if outcome == "COMPLETE" and missing_ids:
        raise ValueError("COMPLETE audit cannot contain missing definitions")
    if outcome == "MISSING_DEFINITION" and not missing_ids:
        raise ValueError("MISSING_DEFINITION requires a registered missing choice")
    if outcome == "REFRAME_REQUIRED" and not missing_ids:
        # No invented choice is an intentional semantic reframe, never syntax retry.
        missing_ids = tuple(registry.definitions)

    def resolved(choice_id: str, *, missing: bool) -> dict[str, object]:
        choice = registry.definitions[choice_id]
        return {
            "definition_id": choice.definition_id,
            "content_ref": choice.content_ref,
            "label": choice.label,
            "symbol_ids": list(choice.symbol_ids),
            "domain_ids": list(choice.domain_ids),
            "topology_ids": list(choice.topology_ids),
            "required_type_id": choice.required_type_id,
            **({"obligation_label": f"D{missing_ids.index(choice_id) + 1}"}
               if missing else {}),
        }

    payload: dict[str, object] = {
        "target_obligation_id": target_obligation_id,
        "parent_statement_hash": parent_statement_hash,
        "root_goal_hash": root_goal_hash,
        "producer_role": "definition_auditor",
        "producer_run_id": producer_run_id,
        "upstream_artifact_hashes": [],
        "audit_outcome": outcome,
        "definitions": [resolved(item, missing=False) for item in defined_ids],
        "missing_definitions": [resolved(item, missing=True) for item in missing_ids],
    }
    envelope = host_artifact(
        decoded,
        host_bindings={
            "artifact_kind": "DEFINITION_AUDIT",
            "target_obligation_id": target_obligation_id,
            "parent_statement_hash": parent_statement_hash,
            "root_goal_hash": root_goal_hash,
            "producer_run_id": producer_run_id,
            "registry_hash": registry.registry_hash,
            "audit_outcome": outcome,
        },
        dependencies=(),
    )
    return payload, envelope
