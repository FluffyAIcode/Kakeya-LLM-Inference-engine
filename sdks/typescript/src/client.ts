/**
 * Kakeya TypeScript SDK — Client (PR-B5 of ADR 0008 Phase B).
 *
 * Targets Node.js 20+, Electron 30+, Bun 1.1+. Uses
 * `@grpc/grpc-js` (Node-native HTTP/2 gRPC) per ADR 0008 §3.2.
 * Browser is intentionally NOT supported in v0.3 (gRPC-Web would
 * require a separate proxy or an alternative transport).
 *
 * Public API mirrors the Python SDK at sdks/python/kakeya/client.py
 * symbol-for-symbol (modulo Promise vs sync return) so cross-
 * language users see one shape.
 *
 * @public
 */

import * as grpc from "@grpc/grpc-js";

import { wrapGrpcError } from "./errors.js";
import {
  CreateSessionRequest,
  RuntimeServiceClient as GrpcRuntimeServiceClient,
} from "./proto_gen/kakeya/v1/runtime.js";
import { Session } from "./session.js";

/**
 * Default gRPC bind address for a local Kakeya runtime. Mirrors
 * `inference_engine.server.grpc_app.DEFAULT_BIND_ADDRESS`.
 */
export const DEFAULT_ADDRESS = "localhost:50051";

/** Constructor options for {@link Client}. */
export interface ClientOptions {
  /**
   * Optional credential override. Defaults to insecure (loopback
   * only, per ADR 0008 §8 OQ-5). Pass `grpc.credentials.createSsl(...)`
   * for TLS deployments.
   */
  credentials?: grpc.ChannelCredentials;

  /** Channel options forwarded to `@grpc/grpc-js`. */
  channelOptions?: grpc.ChannelOptions;
}

/**
 * A connection to a Kakeya RuntimeService.
 *
 * Construction opens a gRPC channel; the channel is closed by
 * {@link close}. The connection is lazy — no RPC is made until a
 * method like {@link createSession} is called.
 */
export class Client {
  /** @internal — exposed for {@link Session} to share the same stub. */
  readonly grpcClient: GrpcRuntimeServiceClient;
  readonly address: string;
  private _closed: boolean = false;

  constructor(address: string = DEFAULT_ADDRESS, options: ClientOptions = {}) {
    this.address = address;
    const credentials =
      options.credentials ?? grpc.credentials.createInsecure();
    this.grpcClient = new GrpcRuntimeServiceClient(
      address,
      credentials,
      options.channelOptions ?? {},
    );
  }

  /** True once {@link close} has been called. */
  get closed(): boolean {
    return this._closed;
  }

  /**
   * Allocate a new session on the runtime.
   *
   * Returns a {@link Session} bound to this client. The session is
   * alive until `session.close()` is called or the runtime evicts
   * it; until then, `session.sessionId` is the stable handle.
   */
  async createSession(opts: {
    eosTokenIds?: number[];
    clientLabel?: string;
  } = {}): Promise<Session> {
    const request: CreateSessionRequest = {
      eosTokenIds: opts.eosTokenIds ?? [],
      clientLabel: opts.clientLabel ?? "",
    };
    return new Promise<Session>((resolve, reject) => {
      this.grpcClient.createSession(request, (err, response) => {
        if (err) {
          reject(wrapGrpcError(err));
          return;
        }
        resolve(new Session(this, response.sessionId));
      });
    });
  }

  /**
   * Close the underlying gRPC channel. Idempotent. Does NOT close
   * the runtime's sessions — call `session.close()` first if the
   * runtime should free them.
   */
  close(): void {
    if (this._closed) {
      return;
    }
    this.grpcClient.close();
    this._closed = true;
  }
}
