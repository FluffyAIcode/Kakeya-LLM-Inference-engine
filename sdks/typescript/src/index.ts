/**
 * Kakeya TypeScript SDK — public API entry (PR-B5 of ADR 0008).
 *
 * Two top-level types power the entire surface:
 *
 *   * {@link Client} — connection to a Kakeya RuntimeService.
 *   * {@link Session} — handle to one server-side session.
 *
 * Plus a typed error hierarchy rooted at {@link KakeyaError} that
 * maps every gRPC status code from the runtime to an SDK class.
 *
 * Targets Node.js 20+, Electron 30+, Bun 1.1+. Browser is NOT
 * supported (gRPC-Web / WebSocket transports are out of scope for
 * v0.3 — see ADR 0008 §8 OQ-1).
 *
 * Usage:
 *
 * ```ts
 * import { Client } from "@kakeya/runtime";
 *
 * const client = new Client("localhost:50051");
 * try {
 *   const session = await client.createSession({ eosTokenIds: [151645] });
 *   try {
 *     await session.append([10, 20, 30]);
 *     for await (const tokenId of session.generate({ maxTokens: 64 })) {
 *       console.log(tokenId);
 *     }
 *   } finally {
 *     await session.close();
 *   }
 * } finally {
 *   client.close();
 * }
 * ```
 *
 * Tokenization is intentionally NOT part of the SDK core — per
 * ADR 0008 §2.4 / §3.4, the runtime treats token ids as opaque
 * integers; rendering messages to tokens is the application's
 * responsibility.
 *
 * @public
 */

export { Client, DEFAULT_ADDRESS } from "./client.js";
export type { ClientOptions } from "./client.js";
export {
  InvalidArgumentError,
  InvariantViolationError,
  KakeyaError,
  ResourceExhaustedError,
  RpcCancelledError,
  SessionClosedError,
  SessionNotFoundError,
  UnimplementedError,
} from "./errors.js";
export { Session } from "./session.js";
export type {
  GenerateOptions,
  GenerateResult,
  SessionInfo,
} from "./session.js";
// Re-export the StopReason enum so callers can compare against
// `lastResult.stopReason` without reaching into proto_gen.
export { GenerateDone_StopReason as StopReason } from "./proto_gen/kakeya/v1/runtime.js";
