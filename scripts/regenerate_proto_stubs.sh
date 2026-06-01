#!/usr/bin/env bash
# Regenerate Python gRPC stubs from proto/kakeya/v1/*.proto.
#
# Generated files land in inference_engine/server/proto_gen/ and are
# committed to the repo. Re-run this script whenever a .proto file
# changes; CI's `proto-stub-drift` check verifies that the committed
# stubs are up-to-date by re-running the script and `git diff
# --exit-code`.
#
# This is the canonical regeneration command. Do NOT invoke
# `python -m grpc_tools.protoc` ad-hoc — keep all flags here so they
# are reproducible across contributor machines and CI.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_ROOT="$ROOT/proto"
OUT_ROOT="$ROOT/inference_engine/server/proto_gen"

# Clean only generated files (preserve hand-written package __init__.py).
find "$OUT_ROOT" -type f \( -name '*_pb2.py' -o -name '*_pb2_grpc.py' -o -name '*_pb2.pyi' \) -delete 2>/dev/null || true

mkdir -p "$OUT_ROOT"

# protoc invocation:
#   --proto_path=$PROTO_ROOT     so proto imports resolve relative to proto/
#   --python_out                 generate the message classes
#   --pyi_out                    generate type stubs (PEP 561 .pyi)
#   --grpc_python_out            generate the gRPC servicer + stub classes
#   $(find ...)                  feed every .proto under proto/ exactly once
python3 -m grpc_tools.protoc \
    --proto_path="$PROTO_ROOT" \
    --python_out="$OUT_ROOT" \
    --pyi_out="$OUT_ROOT" \
    --grpc_python_out="$OUT_ROOT" \
    $(find "$PROTO_ROOT" -name '*.proto')

# Add namespace __init__.py files so `from inference_engine.server.proto_gen.kakeya.v1 import runtime_pb2`
# works without further plumbing. Generated stubs themselves have no
# package layout marker.
mkdir -p "$OUT_ROOT/kakeya/v1"
touch "$OUT_ROOT/__init__.py"
touch "$OUT_ROOT/kakeya/__init__.py"
touch "$OUT_ROOT/kakeya/v1/__init__.py"

# protoc generates `from kakeya.v1 import runtime_pb2 as ...` style
# imports; under our package layout the stubs need to import siblings
# from the same package. The simplest fix is to rewrite the absolute
# imports to relative imports. This is a documented known issue
# (https://github.com/protocolbuffers/protobuf/issues/1491); buf-gen
# would handle it natively but we do not depend on buf for codegen
# in v0.3 (only for lint), so we patch the generated imports here.
GRPC_FILE="$OUT_ROOT/kakeya/v1/runtime_pb2_grpc.py"
if [ -f "$GRPC_FILE" ]; then
    # `from kakeya.v1 import runtime_pb2` -> `from . import runtime_pb2`
    sed -i.bak 's|^from kakeya\.v1 import runtime_pb2|from . import runtime_pb2|' "$GRPC_FILE"
    rm -f "$GRPC_FILE.bak"
fi

echo "Regenerated stubs into $OUT_ROOT"
