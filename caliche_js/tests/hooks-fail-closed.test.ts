import { afterEach, describe, expect, mock, spyOn, test } from "bun:test";
import { native } from "../src/native/module";
import { StreamIterator } from "../src/streaming/stream-iterator";

const nativeIter = { id: "mock-stream" };

describe("StreamIterator fail-closed tool requirement policy", () => {
  afterEach(() => {
    mock.restore();
  });

  test("emits policy_violation before done when required tool was not used", async () => {
    let call = 0;
    spyOn(native, "streamNext").mockImplementation(async () => {
      call += 1;
      if (call === 1) {
        return { type: "done" };
      }
      return null;
    });
    spyOn(native, "streamCancel").mockImplementation(() => {});

    const stream = new StreamIterator(nativeIter, false, {
      message: "latest weather in Tokyo",
      toolRequirementPolicies: [
        {
          name: "require_search",
          toolName: "perplexityResearch",
          predicate: ({ message }) => message.includes("weather"),
        },
      ],
    });

    const first = await stream.next();
    expect(first?.type).toBe("policy_violation");
    if (first?.type === "policy_violation") {
      expect(first.required_tool).toBe("perplexityResearch");
      expect(first.recoverable).toBe(true);
    }

    const second = await stream.next();
    expect(second?.type).toBe("done");
  });

  test("does not emit policy_violation when required tool was called", async () => {
    let call = 0;
    spyOn(native, "streamNext").mockImplementation(async () => {
      call += 1;
      if (call === 1) {
        return {
          type: "tool_call_start",
          id: "call-1",
          name: "perplexityResearch",
          arguments: { query: "weather" },
        };
      }
      if (call === 2) {
        return { type: "done" };
      }
      return null;
    });
    spyOn(native, "streamCancel").mockImplementation(() => {});

    const stream = new StreamIterator(nativeIter, false, {
      message: "latest weather in Tokyo",
      toolRequirementPolicies: [
        {
          toolName: "perplexityResearch",
          predicate: ({ message }) => message.includes("weather"),
        },
      ],
    });

    const first = await stream.next();
    expect(first?.type).toBe("tool_call_start");

    const second = await stream.next();
    expect(second?.type).toBe("done");
  });
});
