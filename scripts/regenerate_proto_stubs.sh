#!/usr/bin/env bash
# Regenerate gRPC stubs from proto/kakeya/v1/*.proto for every SDK.
#
# Targets:
#   * Python  -> inference_engine/server/proto_gen/   (PR-A1)
#   * TypeScript -> sdks/typescript/src/proto_gen/    (PR-B5)
#
# Generated files are committed to the repo. Re-run this script
# whenever a .proto file changes; CI's `proto-stub-drift` check
# verifies that the committed stubs are byte-identical to what
# this script produces.
#
# This is the canonical regeneration command. Do NOT invoke
# `protoc` ad-hoc — keep all flags here so they are reproducible
# across contributor machines and CI.
#
# Prerequisites:
#   * Python: grpcio-tools (provides grpc_tools.protoc)
#   * TypeScript: ts-proto (installed in sdks/typescript/node_modules)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_ROOT="$ROOT/proto"
OUT_ROOT="$ROOT/inference_engine/server/proto_gen"
TS_OUT_ROOT="$ROOT/sdks/typescript/src/proto_gen"
TS_PROTO_PLUGIN="$ROOT/sdks/typescript/node_modules/.bin/protoc-gen-ts_proto"

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

echo "Regenerated Python stubs into $OUT_ROOT"

# -----------------------------------------------------------------------------
# TypeScript stubs (PR-B5)
# -----------------------------------------------------------------------------

if [ -x "$TS_PROTO_PLUGIN" ]; then
    # Clean only generated TS files (preserve hand-written package
    # entry points — none today, but defensive).
    find "$TS_OUT_ROOT" -type f -name '*.ts' -delete 2>/dev/null || true
    mkdir -p "$TS_OUT_ROOT"

    # ts-proto generation flags worth pinning here:
    #   esModuleInterop=true      tsconfig has it; matching keeps imports clean.
    #   forceLong=string          uint64 / int64 -> string (JS number can't hold them).
    #   useExactTypes=false       ts-proto's stricter typing for oneof; we keep
    #                             the more ergonomic discriminated-union shape.
    #   outputServices=grpc-js    generate @grpc/grpc-js style client + server.
    #   useOptionals=messages     proto3 optional + message fields use TS optional.
    #   stringEnums=false         enums emit as integer constants (matches wire).
    #   removeEnumPrefix=true     strip the GenerateDone_ prefix from enum members.
    python3 -m grpc_tools.protoc \
        --plugin="protoc-gen-ts_proto=$TS_PROTO_PLUGIN" \
        --proto_path="$PROTO_ROOT" \
        --ts_proto_out="$TS_OUT_ROOT" \
        --ts_proto_opt=esModuleInterop=true,forceLong=string,outputServices=grpc-js,useOptionals=messages,stringEnums=false,removeEnumPrefix=true \
        $(find "$PROTO_ROOT" -name '*.proto')

    echo "Regenerated TypeScript stubs into $TS_OUT_ROOT"
else
    echo "WARN: ts-proto plugin not found at $TS_PROTO_PLUGIN; skipping TypeScript stub regen." >&2
    echo "      Run 'npm install' in sdks/typescript/ first." >&2
fi
