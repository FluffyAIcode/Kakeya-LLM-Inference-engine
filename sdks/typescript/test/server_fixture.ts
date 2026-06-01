/**
 * In-Node gRPC test server for SDK tests.
 *
 * Spins up a real `@grpc/grpc-js` server bound to `127.0.0.1:0`
 * (random free port), implementing the RuntimeService with
 * deterministic canned-response logic. The SDK under test uses
 * the real generated stub + real HTTP/2 channel to talk to it —
 * no mocks of the SUT — but the server-side logic is a test
 * fixture, not the production runtime (which is Python).
 *
 * The fixture is parameterized so individual tests can configure
 * specific behaviors (failure paths, streaming events, etc.)
 * without re-implementing the listen/bind/teardown dance per test.
 */

import * as grpc from "@grpc/grpc-js";

import {
  AppendTokensRequest,
  AppendTokensResponse,
  CloseSessionRequest,
  CloseSessionResponse,
  CreateSessionRequest,
  CreateSessionResponse,
  GenerateRequest,
  GenerateResponse,
  GetSessionInfoRequest,
  GetSessionInfoResponse,
  RuntimeServiceService,
} from "../src/proto_gen/kakeya/v1/runtime.js";

/**
 * Per-test handlers. Each is optional; an unspecified handler
 * defaults to a benign canned response.
 */
export interface FixtureHandlers {
  createSession?: (
    req: CreateSessionRequest,
  ) => CreateSessionResponse | { error: { code: grpc.status; details: string } };
  appendTokens?: (
    req: AppendTokensRequest,
  ) => AppendTokensResponse | { error: { code: grpc.status; details: string } };
  closeSession?: (
    req: CloseSessionRequest,
  ) =>
    | CloseSessionResponse
    | { error: { code: grpc.status; details: string } };
  getSessionInfo?: (
    req: GetSessionInfoRequest,
  ) =>
    | GetSessionInfoResponse
    | { error: { code: grpc.status; details: string } };
  /** Returns the sequence of frames the Generate stream should emit. */
  generate?: (
    req: GenerateRequest,
  ) => GenerateResponse[] | { error: { code: grpc.status; details: string } };
}

export interface RunningServer {
  address: string;
  shutdown(): Promise<void>;
}

/**
 * Start a gRPC server in this process with the supplied handlers
 * and return its `host:port` plus a shutdown helper.
 */
export async function startTestServer(
  handlers: FixtureHandlers = {},
): Promise<RunningServer> {
  const server = new grpc.Server();

  // Helper: convert a handler return value into a grpc callback call.
  function unary<TReq, TResp>(
    impl: ((req: TReq) => TResp | { error: { code: grpc.status; details: string } }) | undefined,
    fallback: (req: TReq) => TResp,
  ): grpc.handleUnaryCall<TReq, TResp> {
    return (call, callback) => {
      const result = impl ? impl(call.request) : fallback(call.request);
      if (result && typeof result === "object" && "error" in result) {
        const e = (result as { error: { code: grpc.status; details: string } })
          .error;
        callback({
          code: e.code,
          details: e.details,
          metadata: new grpc.Metadata(),
          // grpc.ServiceError requires `name` + `message`; populate
          // both so client-side error.message is meaningful.
          name: "ServerError",
          message: e.details,
        });
        return;
      }
      callback(null, result as TResp);
    };
  }

  function streaming<TReq, TResp>(
    impl: ((req: TReq) => TResp[] | { error: { code: grpc.status; details: string } }) | undefined,
    fallback: (req: TReq) => TResp[],
  ): grpc.handleServerStreamingCall<TReq, TResp> {
    return (call) => {
      const result = impl ? impl(call.request) : fallback(call.request);
      if (result && !Array.isArray(result) && "error" in result) {
        const e = (result as { error: { code: grpc.status; details: string } })
          .error;
        call.emit("error", {
          code: e.code,
          details: e.details,
          metadata: new grpc.Metadata(),
          name: "ServerError",
          message: e.details,
        });
        return;
      }
      for (const frame of result as TResp[]) {
        call.write(frame);
      }
      call.end();
    };
  }

  server.addService(RuntimeServiceService, {
    createSession: unary<CreateSessionRequest, CreateSessionResponse>(
      handlers.createSession,
      (_req) => ({ sessionId: "sess-fixture-default" }),
    ),
    appendTokens: unary<AppendTokensRequest, AppendTokensResponse>(
      handlers.appendTokens,
      (req) => ({ historyLength: req.tokenIds.length.toString() }),
    ),
    generate: streaming<GenerateRequest, GenerateResponse>(
      handlers.generate,
      (_req) => [
        { tokenId: 42 },
        {
          done: {
            stopReason: 1, // MAX_TOKENS
            generatedTokenCount: 1,
            prefillDurationSeconds: 0,
            totalDurationSeconds: 0,
          },
        },
      ],
    ),
    closeSession: unary<CloseSessionRequest, CloseSessionResponse>(
      handlers.closeSession,
      (_req) => ({ finalHistoryLength: "0" }),
    ),
    getSessionInfo: unary<GetSessionInfoRequest, GetSessionInfoResponse>(
      handlers.getSessionInfo,
      (_req) => ({
        historyLength: "0",
        kvLiveBytes: "0",
        cacheInvariantInv1Violations: "0",
        cacheInvariantInv2Violations: "0",
        idleSeconds: 0,
      }),
    ),
  });

  const port: number = await new Promise<number>((resolve, reject) => {
    server.bindAsync(
      "127.0.0.1:0",
      grpc.ServerCredentials.createInsecure(),
      (err, p) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(p);
      },
    );
  });

  return {
    address: `127.0.0.1:${port}`,
    shutdown: () =>
      new Promise<void>((resolve) => {
        server.tryShutdown((_err?: Error) => resolve());
      }),
  };
}
