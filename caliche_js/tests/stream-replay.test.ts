import { afterEach, describe, expect, it, mock, spyOn } from "bun:test";
import { native } from "../src/native/module";
import { StreamIterator, type StreamReplaySource } from "../src/streaming/stream-iterator";

const nativeIter = { id: "mock-replay-stream" };

describe("StreamIterator.replayFrom", () => {
  afterEach(() => {
    mock.restore();
  });

  it("replays from the in-memory buffer when resumeFrom is set", async () => {
    let call = 0;
    spyOn(native, "streamNext").mockImplementation(async () => {
      call += 1;
      if (call === 1) {
        return { type: "text_chunk", content: "hello" };
      }
      if (call === 2) {
        return { type: "text_chunk", content: " world" };
      }
      if (call === 3) {
        return { type: "done" };
      }
      return null;
    });

    const stream = new StreamIterator(nativeIter, false, {
      options: { resumeFrom: "0" },
    });

    const live: string[] = [];
    for await (const event of stream) {
      if (event.type === "text_chunk") {
        live.push(event.content);
      }
    }
    expect(live).toEqual(["hello", " world"]);

    const firstId = (await stream.replayFrom("0")[Symbol.asyncIterator]().next()).value?.event_id;
    expect(firstId).toBeTruthy();

    const replayed: string[] = [];
    for await (const event of stream.replayFrom("evt-1")) {
      if (event.type === "text_chunk") {
        replayed.push(event.content);
      }
    }
    expect(replayed).toEqual([" world"]);
  });

  it("uses an injected replay source when provided", async () => {
    spyOn(native, "streamNext").mockImplementation(async () => null);

    const replaySource: StreamReplaySource = {
      replayFrom: async function* (eventId: string) {
        yield {
          type: "text_chunk" as const,
          content: `replayed-after-${eventId}`,
          event_id: "r1",
        };
        yield { type: "done" as const, event_id: "r2" };
      },
    };

    const stream = new StreamIterator(nativeIter, false, { replaySource });
    const events = [];
    for await (const event of stream.replayFrom("cursor-9")) {
      events.push(event);
    }
    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      type: "text_chunk",
      content: "replayed-after-cursor-9",
    });
    expect(events[1]?.type).toBe("done");
  });

  it("does not throw when no durable backend is configured", async () => {
    spyOn(native, "streamNext").mockImplementation(async () => null);
    const stream = new StreamIterator(nativeIter);
    const events = [];
    for await (const event of stream.replayFrom("missing")) {
      events.push(event);
    }
    expect(events).toEqual([]);
  });

  it("replays empty sequence when replay source has no events after cursor", async () => {
    spyOn(native, "streamNext").mockImplementation(async () => null);
    const replaySource: StreamReplaySource = {
      replayFrom: async function* (_eventId: string) {
        // Unknown / exhausted cursor: durable backends yield nothing.
      },
    };
    const stream = new StreamIterator(nativeIter, false, { replaySource });
    const events = [];
    for await (const event of stream.replayFrom("unknown-cursor")) {
      events.push(event);
    }
    expect(events).toEqual([]);
  });
});
