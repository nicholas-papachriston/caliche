import Module, { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const packageModules = [
  "@caliche/contracts",
  "@caliche/runtime-node",
  "@caliche/core",
  "@caliche/mcp",
  "@caliche/hooks",
  "@caliche/adapters-ai-sdk",
  "@caliche/adapters-langchain",
];

const coreSubpaths = [
  "@caliche/core/streaming/events",
  "@caliche/core/streaming/collectors",
  "@caliche/core/streaming/normalization",
  "@caliche/core/transport/sse",
  "@caliche/core/runtime/env-resolve",
  "@caliche/core/runtime/env",
  "@caliche/core/types",
  "@caliche/core/hooks",
  "@caliche/core/mcp",
  "@caliche/core/session",
  "@caliche/core/agent",
  "@caliche/core/llm",
  "@caliche/core/chains",
  "@caliche/core/workflow",
  "@caliche/core/rag",
  "@caliche/core/adapters/ai-sdk",
  "@caliche/core/adapters/langchain",
];

const pureModules = [
  "@caliche/contracts",
  "@caliche/core/types",
  "@caliche/core/streaming/events",
  "@caliche/core/streaming/collectors",
  "@caliche/core/transport/sse",
  "@caliche/core/runtime/env-resolve",
];

async function assertNoRuntimeNodeLoad(label, load) {
  const originalLoad = Module._load;
  Module._load = function patchedLoad(request, parent, isMain) {
    if (request.startsWith("@caliche/runtime-node")) {
      throw new Error(`${label} unexpectedly loaded @caliche/runtime-node`);
    }
    return originalLoad.apply(this, [request, parent, isMain]);
  };
  try {
    await load();
  } finally {
    Module._load = originalLoad;
  }
}

for (const moduleName of pureModules) {
  await assertNoRuntimeNodeLoad(`ESM import of ${moduleName}`, () => import(moduleName));
}

for (const moduleName of pureModules) {
  await assertNoRuntimeNodeLoad(`CJS require of ${moduleName}`, () => require(moduleName));
}

const modulesToLoad = [...packageModules, ...coreSubpaths];

for (const moduleName of modulesToLoad) {
  await import(moduleName);
}

for (const moduleName of modulesToLoad) {
  require(moduleName);
}

console.log(`Smoke import/require check passed for ${modulesToLoad.length} modules`);
