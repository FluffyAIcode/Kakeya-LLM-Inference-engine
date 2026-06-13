"""Mac bridge — cloud-agent access to the self-hosted Apple Silicon node.

See ``docs/design/mac-bridge-cloud-agent-access.md``. This package holds
the platform-neutral, unit-testable core of the bridge: the preset
allowlist and the request-manifest schema (:mod:`manifest`). The
executor / client CLIs in ``scripts/mac_bridge/`` are thin wrappers
around it (CLI-plumbing coverage convention, like ``scripts/serve.py``).

The package is the precursor of the ADR 0009 ``CAPABILITY_ROLE_TOOL``
plane: a preset here is a tool capability a fleet node advertises; the
manifest is its typed request message (design doc §4.1).
"""

from inference_engine.bridge.manifest import (
    BridgeRequest,
    ManifestError,
    Preset,
    PRESETS,
    build_commands,
    parse_manifest,
)

__all__ = [
    "BridgeRequest",
    "ManifestError",
    "Preset",
    "PRESETS",
    "build_commands",
    "parse_manifest",
]
