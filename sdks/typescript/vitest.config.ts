import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "json-summary"],
      include: ["src/**/*.ts"],
      // Generated protobuf stubs are protoc's artifact, not our
      // surface; coverage on them is neither meaningful nor stable
      // across protoc versions. Same exclusion the Python SDK
      // applies via .coveragerc.
      exclude: ["src/proto_gen/**"],
      // 100% gate on the hand-written SDK code.
      thresholds: {
        lines: 100,
        functions: 100,
        branches: 100,
        statements: 100,
      },
    },
    testTimeout: 15_000,
  },
});
