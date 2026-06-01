import { describe, expect, it } from "vitest";
import * as grpc from "@grpc/grpc-js";

import {
  InvalidArgumentError,
  InvariantViolationError,
  KakeyaError,
  ResourceExhaustedError,
  RpcCancelledError,
  SessionClosedError,
  SessionNotFoundError,
  UnimplementedError,
} from "../src/index.js";
import { wrapGrpcError } from "../src/errors.js";

/**
 * Synthesize a minimal grpc.ServiceError. The real shape is an
 * Error subclass with `code` and `details` plus a few other
 * properties; the wrapper only reads `code` and `details`.
 */
function fakeServiceError(
  code: grpc.status,
  details: string,
): grpc.ServiceError {
  const e = new Error(details) as unknown as grpc.ServiceError;
  (e as unknown as { code: grpc.status }).code = code;
  (e as unknown as { details: string }).details = details;
  (e as unknown as { metadata: grpc.Metadata }).metadata = new grpc.Metadata();
  return e;
}

describe("KakeyaError base", () => {
  it("carries message", () => {
    const e = new KakeyaError("boom");
    expect(e.message).toBe("boom");
  });

  it("carries rpcCode when provided", () => {
    const e = new KakeyaError("boom", { rpcCode: grpc.status.NOT_FOUND });
    expect(e.rpcCode).toBe(grpc.status.NOT_FOUND);
  });

  it("rpcCode defaults to undefined", () => {
    const e = new KakeyaError("boom");
    expect(e.rpcCode).toBeUndefined();
  });

  it("name is set on subclasses", () => {
    expect(new SessionNotFoundError("x").name).toBe("SessionNotFoundError");
    expect(new InvalidArgumentError("x").name).toBe("InvalidArgumentError");
    expect(new InvariantViolationError("x").name).toBe(
      "InvariantViolationError",
    );
    expect(new ResourceExhaustedError("x").name).toBe("ResourceExhaustedError");
    expect(new UnimplementedError("x").name).toBe("UnimplementedError");
    expect(new RpcCancelledError("x").name).toBe("RpcCancelledError");
    expect(new SessionClosedError("x").name).toBe("SessionClosedError");
  });
});

describe("subclass hierarchy", () => {
  it.each([
    SessionNotFoundError,
    InvalidArgumentError,
    InvariantViolationError,
    ResourceExhaustedError,
    UnimplementedError,
    RpcCancelledError,
    SessionClosedError,
  ])("%p subclasses KakeyaError", (cls) => {
    const e = new cls("synthetic");
    expect(e).toBeInstanceOf(KakeyaError);
    expect(e).toBeInstanceOf(Error);
  });
});

describe("wrapGrpcError", () => {
  it.each([
    [grpc.status.NOT_FOUND, SessionNotFoundError],
    [grpc.status.INVALID_ARGUMENT, InvalidArgumentError],
    [grpc.status.FAILED_PRECONDITION, InvariantViolationError],
    [grpc.status.RESOURCE_EXHAUSTED, ResourceExhaustedError],
    [grpc.status.UNIMPLEMENTED, UnimplementedError],
    [grpc.status.CANCELLED, RpcCancelledError],
  ] as const)("maps %s to %p", (code, expected) => {
    const wrapped = wrapGrpcError(fakeServiceError(code, "from server"));
    expect(wrapped).toBeInstanceOf(expected);
    expect(wrapped.rpcCode).toBe(code);
    expect(wrapped.message).toBe("from server");
  });

  it("falls back to KakeyaError for unknown status", () => {
    const wrapped = wrapGrpcError(
      fakeServiceError(grpc.status.INTERNAL, "server exploded"),
    );
    // Falls to base class — exactly KakeyaError, not a subclass.
    expect(wrapped.constructor).toBe(KakeyaError);
    expect(wrapped.rpcCode).toBe(grpc.status.INTERNAL);
    expect(wrapped.message).toBe("server exploded");
  });

  it("uses err.message when details is empty", () => {
    const e = fakeServiceError(grpc.status.NOT_FOUND, "");
    // Replace details with undefined to simulate older grpcjs versions
    // that may produce errors lacking the details field.
    (e as unknown as { details: string | undefined }).details = undefined;
    (e as unknown as { message: string }).message = "fallback message";
    const wrapped = wrapGrpcError(e);
    expect(wrapped.message).toBe("fallback message");
  });

  it("falls back to empty string when both details and message are nullish", () => {
    // ?? only triggers on null / undefined (not empty string), so
    // we have to actually omit both fields to exercise the final
    // empty-string fallback in wrapGrpcError.
    const e = new Error() as unknown as grpc.ServiceError;
    (e as unknown as { code: grpc.status }).code = grpc.status.NOT_FOUND;
    (e as unknown as { details: string | undefined }).details = undefined;
    (e as unknown as { message: string | undefined }).message = undefined;
    const wrapped = wrapGrpcError(e);
    expect(wrapped.message).toBe("");
  });
});
