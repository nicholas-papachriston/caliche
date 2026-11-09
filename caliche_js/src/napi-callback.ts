/**
 * Helpers for NAPI-RS ThreadsafeFunction callbacks.
 *
 * CalleeHandled TSFNs invoke JS as error-first: `(err, payload) => ...`.
 */

export type NapiCallbackArgs = {
  err: Error | null;
  payload: unknown;
};

/**
 * Normalize error-first `(err, payload)` ThreadsafeFunction callback args.
 */
export function unwrapNapiCallbackArgs(args: readonly unknown[]): NapiCallbackArgs {
  if (args.length < 2) {
    throw new Error(
      "NAPI callback expected error-first arguments (err, payload); received fewer than 2 args"
    );
  }
  const rawErr = args[0];
  if (rawErr == null) {
    return { err: null, payload: args[1] };
  }
  if (rawErr instanceof Error) {
    return { err: rawErr, payload: args[1] };
  }
  return { err: new Error(String(rawErr)), payload: args[1] };
}

/**
 * Return a plain object payload for tool executors, defaulting to `{}`.
 */
export function napiToolPayload(args: readonly unknown[]): Record<string, unknown> {
  const { err, payload } = unwrapNapiCallbackArgs(args);
  if (err) {
    throw err;
  }
  if (payload == null) {
    return {};
  }
  if (typeof payload === "object" && !Array.isArray(payload)) {
    return payload as Record<string, unknown>;
  }
  return { value: payload };
}
