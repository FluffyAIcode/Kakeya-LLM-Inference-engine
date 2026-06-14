"""Minimal length-prefixed TCP echo server — the "remote verifier host"
network boundary for the Case-2 cross-host spec-decode test.

Run this as a SEPARATE process; the spec-decode bench
(`k3_specdecode_gpu_bench.py --socket-echo-addr`) round-trips the real
per-block proposer<->verifier payload (draft tokens + aux hidden states)
through it once per block. With `tc netem` applied to the loopback device,
the round-trip incurs REAL network latency + serialization + bandwidth —
i.e. a true two-process socket measurement rather than an injected sleep.

Protocol: 8-byte big-endian length prefix, then that many bytes; the server
writes the same framed blob straight back.
"""

from __future__ import annotations

import argparse
import socket
import struct


def _recvall(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bind", default="127.0.0.1:0")
    ap.add_argument("--port-file", default=None,
                    help="write the bound port here (for --bind :0 discovery)")
    args = ap.parse_args()
    host, port = args.bind.rsplit(":", 1)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(8)
    bound = srv.getsockname()[1]
    if args.port_file:
        with open(args.port_file, "w") as fh:
            fh.write(str(bound))
    print(f"[echo] listening on {host}:{bound}", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                hdr = _recvall(conn, 8)
                (length,) = struct.unpack(">Q", hdr)
                blob = _recvall(conn, length)
                conn.sendall(hdr + blob)
        except (ConnectionError, OSError):
            conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
