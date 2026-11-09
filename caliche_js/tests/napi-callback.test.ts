import { describe, expect, test } from "bun:test";
import { napiToolPayload, unwrapNapiCallbackArgs } from "../src/napi-callback";

describe("unwrapNapiCallbackArgs", () => {
  test("rejects single-arg payloads", () => {
    expect(() => unwrapNapiCallbackArgs([{ x: 1 }])).toThrow("error-first");
  });

  test("error-first null err", () => {
    expect(unwrapNapiCallbackArgs([null, { x: 6 }])).toEqual({
      err: null,
      payload: { x: 6 },
    });
  });

  test("error-first Error", () => {
    const err = new Error("boom");
    const out = unwrapNapiCallbackArgs([err, { x: 1 }]);
    expect(out.err).toBe(err);
    expect(out.payload).toEqual({ x: 1 });
  });
});

describe("napiToolPayload", () => {
  test("reads payload from error-first shape", () => {
    expect(napiToolPayload([null, { x: 6, y: 7 }])).toEqual({ x: 6, y: 7 });
  });

  test("defaults null payload to empty object", () => {
    expect(napiToolPayload([null, null])).toEqual({});
  });

  test("throws when err is set", () => {
    expect(() => napiToolPayload([new Error("nope"), { x: 1 }])).toThrow("nope");
  });
});
