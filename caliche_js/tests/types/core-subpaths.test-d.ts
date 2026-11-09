import {
  type EnvProvider,
  resolveApiKeyFromEnv,
  resolveModelFromEnv,
  resolveProviderFromEnv,
} from "@caliche/core/runtime/env-resolve";
import { collectText } from "@caliche/core/streaming/collectors";
import {
  createInitialAgentStreamState,
  type StreamEvent,
} from "@caliche/core/streaming/events";
import { decodeSseChunk } from "@caliche/core/transport/sse";
import { expectType } from "tsd";

expectType<EnvProvider>(resolveProviderFromEnv({ provider: "openai" }));
expectType<string>(resolveModelFromEnv("openai"));
expectType<string | undefined>(resolveApiKeyFromEnv("openai"));
expectType<Promise<string>>(
  collectText((async function* stream(): AsyncGenerator<StreamEvent> {})())
);
expectType<ReturnType<typeof createInitialAgentStreamState>>(createInitialAgentStreamState());
expectType<ReturnType<typeof decodeSseChunk>>(decodeSseChunk("data: {}\n\n"));
