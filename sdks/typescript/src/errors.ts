/**
 * Typed errors for the Kakeya TypeScript SDK (PR-B5 of ADR 0008).
 *
 * Mirrors the Python SDK's hierarchy at sdks/python/kakeya/errors.py
 * exactly — same gRPC StatusCode -> SDK class mapping, same names
 * (modulo TypeScript naming convention: PascalCase classes, camelCase
 * fields). Cross-language symmetry makes the wire contract auditable
 * by simple grep.
 *
 * @public
 */

import * as grpc from "@grpc/grpc-js";

/**
 * Base class for every typed error the SDK raises. Catch this if
 * you want a single handler for "anything from the Kakeya runtime".
 */
export class KakeyaError extends Error {
  /**
   * The underlying gRPC status code, if the error came from the
   * wire. `undefined` for SDK-side errors (e.g.,
   * {@link SessionClosedError}).
   */
  readonly rpcCode: grpc.status | undefined;

  constructor(message: string, opts?: { rpcCode?: grpc.status }) {
    super(message);
    this.name = "KakeyaError";
    this.rpcCode = opts?.rpcCode;
    // Preserve the prototype chain across transpilation targets.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** gRPC NOT_FOUND. */
export class SessionNotFoundError extends KakeyaError {
  constructor(message: string, opts?: { rpcCode?: grpc.status }) {
    super(message, opts);
    this.name = "SessionNotFoundError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** gRPC INVALID_ARGUMENT. */
export class InvalidArgumentError extends KakeyaError {
  constructor(message: string, opts?: { rpcCode?: grpc.status }) {
    super(message, opts);
    this.name = "InvalidArgumentError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** gRPC FAILED_PRECONDITION (INV-1 / INV-2 violation). */
export class InvariantViolationError extends KakeyaError {
  constructor(message: string, opts?: { rpcCode?: grpc.status }) {
    super(message, opts);
    this.name = "InvariantViolationError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** gRPC RESOURCE_EXHAUSTED (slab pool full). */
export class ResourceExhaustedError extends KakeyaError {
  constructor(message: string, opts?: { rpcCode?: grpc.status }) {
    super(message, opts);
    this.name = "ResourceExhaustedError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** gRPC UNIMPLEMENTED. */
export class UnimplementedError extends KakeyaError {
  constructor(message: string, opts?: { rpcCode?: grpc.status }) {
    super(message, opts);
    this.name = "UnimplementedError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** gRPC CANCELLED. */
export class RpcCancelledError extends KakeyaError {
  constructor(message: string, opts?: { rpcCode?: grpc.status }) {
    super(message, opts);
    this.name = "RpcCancelledError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Raised by the SDK itself — never crosses the wire — when a method
 * is called on a {@link Session} whose `close()` has already been
 * invoked. Distinct from {@link SessionNotFoundError} which means
 * the runtime lost the session.
 */
export class SessionClosedError extends KakeyaError {
  constructor(message: string) {
    super(message);
    this.name = "SessionClosedError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Translate a gRPC ServiceError into a typed {@link KakeyaError}
 * subclass. Used internally by the SDK; not part of the public
 * surface.
 *
 * @internal
 */
export function wrapGrpcError(err: grpc.ServiceError): KakeyaError {
  const code = err.code;
  const details = err.details ?? err.message ?? "";
  switch (code) {
    case grpc.status.NOT_FOUND:
      return new SessionNotFoundError(details, { rpcCode: code });
    case grpc.status.INVALID_ARGUMENT:
      return new InvalidArgumentError(details, { rpcCode: code });
    case grpc.status.FAILED_PRECONDITION:
      return new InvariantViolationError(details, { rpcCode: code });
    case grpc.status.RESOURCE_EXHAUSTED:
      return new ResourceExhaustedError(details, { rpcCode: code });
    case grpc.status.UNIMPLEMENTED:
      return new UnimplementedError(details, { rpcCode: code });
    case grpc.status.CANCELLED:
      return new RpcCancelledError(details, { rpcCode: code });
    default:
      return new KakeyaError(details, { rpcCode: code });
  }
}
