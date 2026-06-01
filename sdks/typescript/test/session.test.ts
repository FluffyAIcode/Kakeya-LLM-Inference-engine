import { describe, expect, it } from "vitest";
import * as grpc from "@grpc/grpc-js";

import {
  Client,
  InvalidArgumentError,
  InvariantViolationError,
  SessionClosedError,
  SessionNotFoundError,
  StopReason,
} from "../src/index.js";
import { startTestServer } from "./server_fixture.js";

describe("Session", () => {
  describe("properties", () => {
    it("session_id is what the server returned", async () => {
      const server = await startTestServer({
        createSession: () => ({ sessionId: "sess-abc-123" }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        expect(session.sessionId).toBe("sess-abc-123");
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("closed defaults to false", async () => {
      const server = await startTestServer();
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        expect(session.closed).toBe(false);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("lastResult starts in initial state", async () => {
      const server = await startTestServer();
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        expect(session.lastResult.stopReason).toBeUndefined();
        expect(session.lastResult.generatedTokenCount).toBe(0);
        expect(session.lastResult.prefillDurationSeconds).toBe(0);
        expect(session.lastResult.totalDurationSeconds).toBe(0);
        expect(session.lastResult.historyTruncatedDropped).toBeUndefined();
      } finally {
        client.close();
        await server.shutdown();
      }
    });
  });

  describe("append", () => {
    it("returns new history length", async () => {
      const server = await startTestServer({
        appendTokens: (req) => ({
          historyLength: req.tokenIds.length.toString(),
        }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        const len = await session.append([10, 20, 30]);
        expect(len).toBe(3);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("forwards token_ids verbatim", async () => {
      let captured: number[] | undefined;
      const server = await startTestServer({
        appendTokens: (req) => {
          captured = req.tokenIds;
          return { historyLength: "0" };
        },
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await session.append([42, 43, 44]);
        expect(captured).toEqual([42, 43, 44]);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("after local close raises SessionClosedError", async () => {
      const server = await startTestServer();
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await session.close();
        await expect(session.append([1, 2, 3])).rejects.toBeInstanceOf(
          SessionClosedError,
        );
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("NOT_FOUND from runtime maps to SessionNotFoundError", async () => {
      const server = await startTestServer({
        appendTokens: () => ({
          error: {
            code: grpc.status.NOT_FOUND,
            details: "session_id 'sess-x' not found",
          },
        }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await expect(session.append([1])).rejects.toBeInstanceOf(
          SessionNotFoundError,
        );
      } finally {
        client.close();
        await server.shutdown();
      }
    });
  });

  describe("generate", () => {
    it("yields token ids in order", async () => {
      const server = await startTestServer({
        generate: () => [
          { tokenId: 100 },
          { tokenId: 101 },
          { tokenId: 102 },
          {
            done: {
              stopReason: StopReason.MAX_TOKENS,
              generatedTokenCount: 3,
              prefillDurationSeconds: 0,
              totalDurationSeconds: 0.012,
            },
          },
        ],
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        const tokens: number[] = [];
        for await (const tid of session.generate({ maxTokens: 3 })) {
          tokens.push(tid);
        }
        expect(tokens).toEqual([100, 101, 102]);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("populates lastResult metadata after the stream completes", async () => {
      const server = await startTestServer({
        generate: () => [
          { tokenId: 7 },
          {
            done: {
              stopReason: StopReason.EOS,
              generatedTokenCount: 1,
              prefillDurationSeconds: 0,
              totalDurationSeconds: 0.5,
            },
          },
        ],
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        for await (const _t of session.generate({ maxTokens: 5 })) { /* drain */ }
        expect(session.lastResult.stopReason).toBe(StopReason.EOS);
        expect(session.lastResult.generatedTokenCount).toBe(1);
        expect(session.lastResult.totalDurationSeconds).toBe(0.5);
        expect(session.lastResult.historyTruncatedDropped).toBeUndefined();
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("captures HistoryTruncated frame into lastResult", async () => {
      const server = await startTestServer({
        generate: () => [
          { truncated: { droppedTokenCount: "2" } },
          { tokenId: 9 },
          {
            done: {
              stopReason: StopReason.MAX_TOKENS,
              generatedTokenCount: 1,
              prefillDurationSeconds: 0,
              totalDurationSeconds: 0,
            },
          },
        ],
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        const tokens: number[] = [];
        for await (const tid of session.generate({ maxTokens: 1 })) {
          tokens.push(tid);
        }
        expect(tokens).toEqual([9]);
        expect(session.lastResult.historyTruncatedDropped).toBe(2);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("forwards seed / temperature / topP / topK to the request", async () => {
      let captured: {
        seed?: number | string;
        temperature?: number;
        topP?: number;
        topK?: number;
      } = {};
      const server = await startTestServer({
        generate: (req) => {
          captured = {
            seed: req.seed,
            temperature: req.temperature,
            topP: req.topP,
            topK: req.topK,
          };
          return [
            {
              done: {
                stopReason: StopReason.MAX_TOKENS,
                generatedTokenCount: 0,
                prefillDurationSeconds: 0,
                totalDurationSeconds: 0,
              },
            },
          ];
        },
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        for await (const _t of session.generate({
          maxTokens: 1,
          seed: 42,
          temperature: 0,
          topK: 1,
        })) { /* drain */ }
        expect(captured.temperature).toBe(0);
        expect(captured.topK).toBe(1);
        expect(captured.seed).toBeDefined();
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("INVALID_ARGUMENT mid-stream maps to InvalidArgumentError", async () => {
      const server = await startTestServer({
        generate: () => ({
          error: {
            code: grpc.status.INVALID_ARGUMENT,
            details: "max_tokens must be >= 1",
          },
        }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await expect(async () => {
          for await (const _t of session.generate({ maxTokens: 0 })) { /* drain */ }
        }).rejects.toBeInstanceOf(InvalidArgumentError);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("FAILED_PRECONDITION mid-stream maps to InvariantViolationError", async () => {
      const server = await startTestServer({
        generate: () => ({
          error: {
            code: grpc.status.FAILED_PRECONDITION,
            details: "INV-1 violation",
          },
        }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await expect(async () => {
          for await (const _t of session.generate({ maxTokens: 1 })) { /* drain */ }
        }).rejects.toBeInstanceOf(InvariantViolationError);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("after local close raises SessionClosedError synchronously", async () => {
      const server = await startTestServer();
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await session.close();
        expect(() => session.generate({ maxTokens: 1 })).toThrow(
          SessionClosedError,
        );
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("resets lastResult between calls", async () => {
      const server = await startTestServer({
        generate: () => [
          { truncated: { droppedTokenCount: "5" } },
          { tokenId: 1 },
          {
            done: {
              stopReason: StopReason.MAX_TOKENS,
              generatedTokenCount: 1,
              prefillDurationSeconds: 0,
              totalDurationSeconds: 0,
            },
          },
        ],
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        for await (const _t of session.generate({ maxTokens: 1 })) { /* drain */ }
        expect(session.lastResult.historyTruncatedDropped).toBe(5);
        // Now make the second call return NO truncated frame.
        await server.shutdown();
        const server2 = await startTestServer({
          createSession: () => ({ sessionId: session.sessionId }),
          generate: () => [
            { tokenId: 2 },
            {
              done: {
                stopReason: StopReason.MAX_TOKENS,
                generatedTokenCount: 1,
                prefillDurationSeconds: 0,
                totalDurationSeconds: 0,
              },
            },
          ],
        });
        try {
          const client2 = new Client(server2.address);
          try {
            const s2 = await client2.createSession();
            for await (const _t of s2.generate({ maxTokens: 1 })) { /* drain */ }
            expect(s2.lastResult.historyTruncatedDropped).toBeUndefined();
          } finally {
            client2.close();
          }
        } finally {
          await server2.shutdown();
        }
      } finally {
        client.close();
      }
    });
  });

  describe("info", () => {
    it("returns SessionInfo with mapped fields", async () => {
      const server = await startTestServer({
        getSessionInfo: () => ({
          historyLength: "5",
          kvLiveBytes: "12345",
          cacheInvariantInv1Violations: "0",
          cacheInvariantInv2Violations: "0",
          idleSeconds: 1.5,
        }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        const info = await session.info();
        expect(info).toEqual({
          historyLength: 5,
          kvLiveBytes: 12345,
          cacheInvariantInv1Violations: 0,
          cacheInvariantInv2Violations: 0,
          idleSeconds: 1.5,
        });
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("NOT_FOUND maps to SessionNotFoundError", async () => {
      const server = await startTestServer({
        getSessionInfo: () => ({
          error: {
            code: grpc.status.NOT_FOUND,
            details: "session_id 'sess-x' not found",
          },
        }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await expect(session.info()).rejects.toBeInstanceOf(
          SessionNotFoundError,
        );
      } finally {
        client.close();
        await server.shutdown();
      }
    });
  });

  describe("close", () => {
    it("returns final history length on success", async () => {
      const server = await startTestServer({
        closeSession: () => ({ finalHistoryLength: "12" }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        const final = await session.close();
        expect(final).toBe(12);
        expect(session.closed).toBe(true);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("is idempotent at the SDK level", async () => {
      const server = await startTestServer({
        closeSession: () => ({ finalHistoryLength: "3" }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await session.close();
        // Second close: no RPC, returns 0.
        expect(await session.close()).toBe(0);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("RPC error on close still flips closed flag", async () => {
      const server = await startTestServer({
        closeSession: () => ({
          error: {
            code: grpc.status.NOT_FOUND,
            details: "session_id 'sess-phantom' not found",
          },
        }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        await expect(session.close()).rejects.toBeInstanceOf(
          SessionNotFoundError,
        );
        expect(session.closed).toBe(true);
        // Subsequent close is a no-op (returns 0).
        expect(await session.close()).toBe(0);
      } finally {
        client.close();
        await server.shutdown();
      }
    });
  });
});
