/**
 * Kakeya TypeScript SDK — Session (PR-B5 of ADR 0008 Phase B).
 *
 * Mirrors sdks/python/kakeya/session.py symbol-for-symbol.
 *
 * @public
 */

import * as grpc from "@grpc/grpc-js";

import {
  SessionClosedError,
  wrapGrpcError,
} from "./errors.js";
import {
  AppendTokensRequest,
  CloseSessionRequest,
  GenerateDone_StopReason,
  GenerateRequest,
  GenerateResponse,
  GetSessionInfoRequest,
} from "./proto_gen/kakeya/v1/runtime.js";
import type { Client } from "./client.js";

/**
 * Read-only snapshot of a session's server-side state. Returned by
 * {@link Session.info}.
 */
export interface SessionInfo {
  /** Number of tokens currently in the session's history. */
  historyLength: number;
  /** Live KV bytes held by this session's slab. */
  kvLiveBytes: number;
  /** INV-1 violations observed (must be 0 in healthy operation). */
  cacheInvariantInv1Violations: number;
  /** INV-2 violations observed (must be 0 in healthy operation). */
  cacheInvariantInv2Violations: number;
  /** Wall seconds since the session's last RPC interaction. */
  idleSeconds: number;
}

/** Options accepted by {@link Session.generate}. */
export interface GenerateOptions {
  /** Maximum tokens to emit in this Generate call. Required, must be >= 1. */
  maxTokens: number;
  /** Per-Generate sampling seed (greedy mode ignores it; accepted for OQ-4). */
  seed?: number | string;
  /** Sampling temperature. v0.3 supports only 0 (greedy). */
  temperature?: number;
  /** Top-p sampling. v0.3 supports only unset (greedy). */
  topP?: number;
  /** Top-k sampling. v0.3 supports only 1 (greedy). */
  topK?: number;
}

/** Metadata populated after a {@link Session.generate} stream finishes. */
export interface GenerateResult {
  stopReason: GenerateDone_StopReason | undefined;
  generatedTokenCount: number;
  prefillDurationSeconds: number;
  totalDurationSeconds: number;
  /**
   * Number of tokens dropped by the runtime's sink+window cache at
   * the start of this Generate call, or `undefined` if no
   * truncation occurred. Per the proto contract, the runtime emits
   * the truncated frame at most once per call, before any
   * `tokenId`.
   */
  historyTruncatedDropped: number | undefined;
}

/** A handle to one server-side session. */
export class Session {
  readonly sessionId: string;
  private readonly _client: Client;
  private _closed: boolean = false;
  private _lastResult: GenerateResult = {
    stopReason: undefined,
    generatedTokenCount: 0,
    prefillDurationSeconds: 0,
    totalDurationSeconds: 0,
    historyTruncatedDropped: undefined,
  };

  /** @internal — only :class:`Client.createSession` should call this. */
  constructor(client: Client, sessionId: string) {
    this._client = client;
    this.sessionId = sessionId;
  }

  /** True once {@link close} has been called locally. */
  get closed(): boolean {
    return this._closed;
  }

  /**
   * Result metadata from the most recent {@link generate} call.
   * Populated as the stream completes; reset at the start of the
   * next call.
   */
  get lastResult(): GenerateResult {
    return this._lastResult;
  }

  /**
   * Append raw token ids to the session's history. Returns the new
   * `historyLength`.
   */
  async append(tokenIds: number[]): Promise<number> {
    this._checkOpen();
    const request: AppendTokensRequest = {
      sessionId: this.sessionId,
      tokenIds,
    };
    return new Promise<number>((resolve, reject) => {
      this._client.grpcClient.appendTokens(request, (err, response) => {
        if (err) {
          reject(wrapGrpcError(err));
          return;
        }
        resolve(Number(response.historyLength));
      });
    });
  }

  /**
   * Stream generated token ids as an async iterable. Each iterator
   * yield is one committed token. After iteration completes,
   * {@link lastResult} is populated with the {@link GenerateResult}
   * metadata (stop reason, count, durations, truncation).
   *
   * v0.3 supports only greedy decoding. Setting `temperature` /
   * `topP` / `topK` to anything other than the greedy no-op default
   * raises {@link InvalidArgumentError} from the runtime.
   */
  generate(opts: GenerateOptions): AsyncIterable<number> {
    this._checkOpen();
    // Reset metadata at the start of every call so stale values
    // from a previous call don't leak into the next.
    this._lastResult = {
      stopReason: undefined,
      generatedTokenCount: 0,
      prefillDurationSeconds: 0,
      totalDurationSeconds: 0,
      historyTruncatedDropped: undefined,
    };
    // ts-proto generates `seed: string | undefined` because the
    // proto field is `optional uint64` and we set forceLong=string
    // (JS `number` can't safely represent the full uint64 range).
    // Accept number for ergonomics and stringify; callers needing
    // the full range pass a string directly.
    const request: GenerateRequest = {
      sessionId: this.sessionId,
      maxTokens: opts.maxTokens,
      seed: opts.seed === undefined ? undefined : String(opts.seed),
      temperature: opts.temperature,
      topP: opts.topP,
      topK: opts.topK,
    };
    const stream = this._client.grpcClient.generate(request);
    return this._consumeGenerateStream(stream);
  }

  private async *_consumeGenerateStream(
    stream: grpc.ClientReadableStream<GenerateResponse>,
  ): AsyncIterable<number> {
    // Bridge the event-emitter stream into an async iterator. We
    // buffer (data | error | end) events and replay them in order
    // so the caller can `for await` cleanly. Errors map to typed
    // KakeyaError subclasses before being thrown.
    type Event =
      | { kind: "data"; response: GenerateResponse }
      | { kind: "error"; err: grpc.ServiceError }
      | { kind: "end" };

    const events: Event[] = [];
    let resolveNext: (() => void) | null = null;
    const wakeup = () => {
      if (resolveNext) {
        const r = resolveNext;
        resolveNext = null;
        r();
      }
    };

    stream.on("data", (response: GenerateResponse) => {
      events.push({ kind: "data", response });
      wakeup();
    });
    stream.on("error", (err: grpc.ServiceError) => {
      events.push({ kind: "error", err });
      wakeup();
    });
    stream.on("end", () => {
      events.push({ kind: "end" });
      wakeup();
    });

    while (true) {
      while (events.length > 0) {
        const ev = events.shift()!;
        if (ev.kind === "error") {
          throw wrapGrpcError(ev.err);
        }
        if (ev.kind === "end") {
          return;
        }
        const r = ev.response;
        if (r.tokenId !== undefined) {
          yield r.tokenId;
        } else if (r.truncated !== undefined) {
          this._lastResult.historyTruncatedDropped = Number(
            r.truncated.droppedTokenCount,
          );
        } else if (r.done !== undefined) {
          this._lastResult.stopReason = r.done.stopReason;
          this._lastResult.generatedTokenCount = r.done.generatedTokenCount;
          this._lastResult.prefillDurationSeconds =
            r.done.prefillDurationSeconds;
          this._lastResult.totalDurationSeconds = r.done.totalDurationSeconds;
        }
      }
      // Wait for the next event.
      await new Promise<void>((resolve) => {
        resolveNext = resolve;
      });
    }
  }

  /** Return a snapshot of the session's server-side state. */
  async info(): Promise<SessionInfo> {
    const request: GetSessionInfoRequest = { sessionId: this.sessionId };
    return new Promise<SessionInfo>((resolve, reject) => {
      this._client.grpcClient.getSessionInfo(request, (err, response) => {
        if (err) {
          reject(wrapGrpcError(err));
          return;
        }
        resolve({
          historyLength: Number(response.historyLength),
          kvLiveBytes: Number(response.kvLiveBytes),
          cacheInvariantInv1Violations: Number(
            response.cacheInvariantInv1Violations,
          ),
          cacheInvariantInv2Violations: Number(
            response.cacheInvariantInv2Violations,
          ),
          idleSeconds: response.idleSeconds,
        });
      });
    });
  }

  /**
   * Close the session on the runtime. Returns the final history
   * length. Idempotent at the SDK level: a second call returns 0
   * without contacting the runtime.
   */
  async close(): Promise<number> {
    if (this._closed) {
      return 0;
    }
    const request: CloseSessionRequest = { sessionId: this.sessionId };
    return new Promise<number>((resolve, reject) => {
      this._client.grpcClient.closeSession(request, (err, response) => {
        if (err) {
          this._closed = true;
          reject(wrapGrpcError(err));
          return;
        }
        this._closed = true;
        resolve(Number(response.finalHistoryLength));
      });
    });
  }

  private _checkOpen(): void {
    if (this._closed) {
      throw new SessionClosedError(
        `session '${this.sessionId}' has been closed locally; ` +
          "create a new session to continue work",
      );
    }
  }
}
