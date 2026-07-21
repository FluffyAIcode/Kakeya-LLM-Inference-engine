#!/usr/bin/env python3
"""Bridge the acceptance harness stdin protocol to the runtime control UDS."""

from __future__ import annotations

import argparse
import json
import socket
import sys

from inference_engine.backends.mlx.decode_worker import _recv_frame, _send_frame
from inference_engine.backends.mlx.decode_worker_acceptance import (
    normalize_control_socket_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    operation = str(request.get("operation", ""))
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(args.timeout_s)
    try:
        connection.connect(normalize_control_socket_path(args.socket))
        _send_frame(
            connection,
            {
                "schema_version": int(request.get("schema_version", -1)),
                "operation": operation,
                "payload": dict(request.get("payload", {})),
            },
        )
        response, _ = _recv_frame(connection)
    except (OSError, ValueError, EOFError, json.JSONDecodeError) as exc:
        print(f"acceptance control {operation!r} failed: {exc}", file=sys.stderr)
        return 1
    finally:
        connection.close()
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
