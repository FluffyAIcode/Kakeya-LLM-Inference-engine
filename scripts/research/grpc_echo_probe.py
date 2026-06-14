"""Direct-gRPC cross-host round-trip probe (no SSH tunnel).

Compares a REAL gRPC channel against the reverse-SSH-tunnel raw-socket path for
the Case-2 per-block proposer<->verifier payload. Raw-bytes unary RPC (no proto
codegen): the server echoes the payload; the client times the round-trip for a
range of payload sizes.

Roles:
  --role server --bind 0.0.0.0:PORT          (run on the reachable host, e.g. GPU)
  --role client --addr HOST:PORT --payloads 64,160000 --reps 12

Run the server on the GPU bound to a vast-mapped internal port; connect from the
cloud agent to PUBLIC_IPADDR:<mapped public port>. That round-trip traverses the
real network over HTTP/2 with gRPC flow control — no SSH encryption/relay hop.
"""

from __future__ import annotations

import argparse
import time
from concurrent import futures
from typing import List

import grpc

_IDENT = lambda b: b  # noqa: E731  (raw-bytes serializer)
_METHOD = "/echo.Echo/Echo"
_BIG = 256 * 1024 * 1024
_OPTS = [("grpc.max_send_message_length", _BIG),
         ("grpc.max_receive_message_length", _BIG)]


def _serve(bind: str) -> int:
    def handler(request: bytes, context) -> bytes:  # echo
        return request
    h = grpc.method_handlers_generic_handler(
        "echo.Echo",
        {"Echo": grpc.unary_unary_rpc_method_handler(
            handler, request_deserializer=_IDENT, response_serializer=_IDENT)})
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=_OPTS)
    server.add_generic_rpc_handlers((h,))
    server.add_insecure_port(bind)
    server.start()
    print(f"[grpc-echo] serving on {bind}", flush=True)
    server.wait_for_termination()
    return 0


def _client(addr: str, payloads: List[int], reps: int) -> int:
    channel = grpc.insecure_channel(
        addr, options=_OPTS + [("grpc.enable_http_proxy", 0)])
    echo = channel.unary_unary(_METHOD, request_serializer=_IDENT,
                               response_deserializer=_IDENT)
    grpc.channel_ready_future(channel).result(timeout=30)
    for nb in payloads:
        p = b"x" * nb
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            r = echo(p)
            ts.append((time.perf_counter() - t) * 1000.0)
            assert len(r) == nb
        ts.sort()
        med = ts[len(ts) // 2]
        print(f"[grpc-echo] payload {nb:8d} B -> median RTT {med:.1f} ms "
              f"(min {ts[0]:.1f}, max {ts[-1]:.1f})", flush=True)
    channel.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", required=True, choices=["server", "client"])
    ap.add_argument("--bind", default="0.0.0.0:72299")
    ap.add_argument("--addr", default="127.0.0.1:72299")
    ap.add_argument("--payloads", default="64,40000,80000,160000")
    ap.add_argument("--reps", type=int, default=12)
    args = ap.parse_args()
    if args.role == "server":
        return _serve(args.bind)
    return _client(args.addr, [int(x) for x in args.payloads.split(",") if x.strip()],
                   args.reps)


if __name__ == "__main__":
    raise SystemExit(main())
