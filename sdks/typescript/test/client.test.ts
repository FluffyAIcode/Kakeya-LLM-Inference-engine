import { describe, expect, it } from "vitest";
import * as grpc from "@grpc/grpc-js";

import {
  Client,
  DEFAULT_ADDRESS,
  ResourceExhaustedError,
  Session,
  SessionNotFoundError,
} from "../src/index.js";
import { startTestServer } from "./server_fixture.js";

describe("Client", () => {
  describe("construction", () => {
    it("default address constant matches Python SDK", () => {
      expect(DEFAULT_ADDRESS).toBe("localhost:50051");
    });

    it("address property reflects constructor arg", () => {
      const c = new Client("127.0.0.1:99999");
      expect(c.address).toBe("127.0.0.1:99999");
      c.close();
    });

    it("default constructor uses DEFAULT_ADDRESS", () => {
      const c = new Client();
      expect(c.address).toBe(DEFAULT_ADDRESS);
      c.close();
    });

    it("closed defaults to false", () => {
      const c = new Client("127.0.0.1:99999");
      expect(c.closed).toBe(false);
      c.close();
    });

    it("accepts custom credentials option", () => {
      const c = new Client("127.0.0.1:99999", {
        credentials: grpc.credentials.createInsecure(),
      });
      c.close();
    });

    it("accepts channel options", () => {
      const c = new Client("127.0.0.1:99999", {
        channelOptions: { "grpc.enable_retries": 0 },
      });
      c.close();
    });
  });

  describe("close", () => {
    it("flips closed flag", () => {
      const c = new Client("127.0.0.1:99999");
      c.close();
      expect(c.closed).toBe(true);
    });

    it("is idempotent", () => {
      const c = new Client("127.0.0.1:99999");
      c.close();
      c.close(); // must not throw
      expect(c.closed).toBe(true);
    });
  });

  describe("createSession", () => {
    it("returns a Session with the server-issued id", async () => {
      const server = await startTestServer({
        createSession: (_req) => ({ sessionId: "sess-test-12345" }),
      });
      const client = new Client(server.address);
      try {
        const session = await client.createSession();
        expect(session).toBeInstanceOf(Session);
        expect(session.sessionId).toBe("sess-test-12345");
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("forwards eosTokenIds to the request", async () => {
      let captured: number[] | undefined;
      const server = await startTestServer({
        createSession: (req) => {
          captured = req.eosTokenIds;
          return { sessionId: "sess-test" };
        },
      });
      const client = new Client(server.address);
      try {
        await client.createSession({ eosTokenIds: [7, 11, 13] });
        expect(captured).toEqual([7, 11, 13]);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("forwards clientLabel to the request", async () => {
      let captured: string | undefined;
      const server = await startTestServer({
        createSession: (req) => {
          captured = req.clientLabel;
          return { sessionId: "sess-test" };
        },
      });
      const client = new Client(server.address);
      try {
        await client.createSession({ clientLabel: "demo-app-1" });
        expect(captured).toBe("demo-app-1");
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("defaults eosTokenIds to empty list when not provided", async () => {
      let captured: number[] | undefined;
      const server = await startTestServer({
        createSession: (req) => {
          captured = req.eosTokenIds;
          return { sessionId: "sess-test" };
        },
      });
      const client = new Client(server.address);
      try {
        await client.createSession();
        expect(captured).toEqual([]);
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("maps RESOURCE_EXHAUSTED gRPC status to ResourceExhaustedError", async () => {
      const server = await startTestServer({
        createSession: () => ({
          error: {
            code: grpc.status.RESOURCE_EXHAUSTED,
            details: "slab pool exhausted: all 1 slabs in use",
          },
        }),
      });
      const client = new Client(server.address);
      try {
        await expect(client.createSession()).rejects.toBeInstanceOf(
          ResourceExhaustedError,
        );
        await expect(client.createSession()).rejects.toMatchObject({
          rpcCode: grpc.status.RESOURCE_EXHAUSTED,
        });
      } finally {
        client.close();
        await server.shutdown();
      }
    });

    it("maps NOT_FOUND back through the same wrapper", async () => {
      const server = await startTestServer({
        createSession: () => ({
          error: {
            code: grpc.status.NOT_FOUND,
            details: "synthetic not-found",
          },
        }),
      });
      const client = new Client(server.address);
      try {
        await expect(client.createSession()).rejects.toBeInstanceOf(
          SessionNotFoundError,
        );
      } finally {
        client.close();
        await server.shutdown();
      }
    });
  });
});
