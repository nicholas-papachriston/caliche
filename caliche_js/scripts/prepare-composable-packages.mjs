import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const packagesDir = path.join(rootDir, "packages");
const coreDir = path.join(packagesDir, "core");
const runtimeNodeDir = path.join(packagesDir, "runtime-node");
const contractsDir = path.join(packagesDir, "contracts");
const mcpDir = path.join(packagesDir, "mcp");
const hooksDir = path.join(packagesDir, "hooks");
const aiSdkAdapterDir = path.join(packagesDir, "adapters-ai-sdk");
const langChainAdapterDir = path.join(packagesDir, "adapters-langchain");

const rootDistDir = path.join(rootDir, "dist");
const rootNativeLoaderJs = path.join(rootDir, "index.js");
const rootNativeLoaderDts = path.join(rootDir, "index.d.ts");

function ensureDir(dir) {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

function resetDir(dir) {
  rmSync(dir, { force: true, recursive: true });
  mkdirSync(dir, { recursive: true });
}

function walkFiles(dir, predicate = () => true) {
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath, predicate));
    } else if (predicate(fullPath)) {
      files.push(fullPath);
    }
  }
  return files.sort();
}

function toPosix(value) {
  return value.split(path.sep).join("/");
}

function relativeImportSpecifier(fromFile, toFile) {
  let specifier = toPosix(path.relative(path.dirname(fromFile), toFile));
  if (!specifier.startsWith(".")) {
    specifier = `./${specifier}`;
  }
  return specifier;
}

function isIdentifier(name) {
  return /^[A-Za-z_$][\w$]*$/.test(name);
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

function extractRuntimeExports(dtsPath) {
  if (!existsSync(dtsPath)) {
    return [];
  }

  const text = readFileSync(dtsPath, "utf8");
  const names = [];

  for (const match of text.matchAll(
    /\bexport\s+(?:declare\s+)?(?:abstract\s+)?(?:class|function|const|let|var|enum)\s+([A-Za-z_$][\w$]*)/g
  )) {
    names.push(match[1]);
  }

  for (const match of text.matchAll(
    /\bexport\s*\{([\s\S]*?)\}\s*(?:from\s+["'][^"']+["'])?\s*;/g
  )) {
    for (const rawPart of match[1].split(",")) {
      const part = rawPart.trim();
      if (!part || part.startsWith("type ")) {
        continue;
      }
      const aliasMatch = part.match(/\bas\s+([A-Za-z_$][\w$]*)$/);
      const name = aliasMatch ? aliasMatch[1] : part.split(/\s+/)[0];
      if (name && name !== "default" && isIdentifier(name)) {
        names.push(name);
      }
    }
  }

  return uniqueSorted(names.filter((name) => name && name !== "default" && isIdentifier(name)));
}

function writeCjsInteropEntrypoint(targetPath, cjsSpecifier, exportNames, bindingName = "mod") {
  const lines = [
    'import { createRequire } from "node:module";',
    "",
    "const require = createRequire(import.meta.url);",
    `const ${bindingName} = require(${JSON.stringify(cjsSpecifier)});`,
    "",
    `export default ${bindingName};`,
  ];

  for (const name of uniqueSorted(exportNames)) {
    lines.push(`export const ${name} = ${bindingName}.${name};`);
  }

  writeFileSync(targetPath, `${lines.join("\n")}\n`, "utf8");
}

function originalJsRelSet(distDir) {
  return new Set(
    walkFiles(distDir, (file) => file.endsWith(".js")).map((file) =>
      toPosix(path.relative(distDir, file))
    )
  );
}

function originalDtsRelSet(distDir) {
  return new Set(
    walkFiles(distDir, (file) => file.endsWith(".d.ts")).map((file) =>
      toPosix(path.relative(distDir, file))
    )
  );
}

function rewriteRelativeCjsRequires(source, fromFile, distDir, jsRelSet) {
  return source.replace(/require\((["'])(\.{1,2}\/[^"']+)\1\)/g, (match, quote, specifier) => {
    if (path.extname(specifier)) {
      return match;
    }

    const absoluteTarget = path.resolve(path.dirname(fromFile), specifier);
    const fileCandidate = toPosix(path.relative(distDir, `${absoluteTarget}.js`));
    if (jsRelSet.has(fileCandidate)) {
      return `require(${quote}${specifier}.cjs${quote})`;
    }

    const indexCandidate = toPosix(path.relative(distDir, path.join(absoluteTarget, "index.js")));
    if (jsRelSet.has(indexCandidate)) {
      return `require(${quote}${specifier}/index.cjs${quote})`;
    }

    return match;
  });
}

function rewriteSourceMapComment(source, cjsBasename) {
  return source.replace(
    /\/\/# sourceMappingURL=([^\n]+)\.js\.map/g,
    `//# sourceMappingURL=${cjsBasename}.map`
  );
}

function convertDistJsToCjs(distDir) {
  const jsRelSet = originalJsRelSet(distDir);
  const jsFiles = walkFiles(distDir, (file) => file.endsWith(".js"));

  for (const jsFile of jsFiles) {
    const cjsFile = jsFile.replace(/\.js$/, ".cjs");
    const cjsBasename = path.basename(cjsFile);
    const source = readFileSync(jsFile, "utf8");
    const rewritten = rewriteSourceMapComment(
      rewriteRelativeCjsRequires(source, jsFile, distDir, jsRelSet),
      cjsBasename
    );
    writeFileSync(cjsFile, rewritten, "utf8");
    rmSync(jsFile);
  }

  for (const mapFile of walkFiles(distDir, (file) => file.endsWith(".js.map"))) {
    const cjsMapFile = mapFile.replace(/\.js\.map$/, ".cjs.map");
    const mapSource = readFileSync(mapFile, "utf8").replace(
      /"file":"([^"]+)\.js"/,
      '"file":"$1.cjs"'
    );
    writeFileSync(cjsMapFile, mapSource, "utf8");
    rmSync(mapFile);
  }
}

function resolveEsmDtsSpecifier(specifier, fromFile, distDir, dtsRelSet) {
  if (path.extname(specifier)) {
    return specifier;
  }

  const absoluteTarget = path.resolve(path.dirname(fromFile), specifier);
  const fileCandidate = toPosix(path.relative(distDir, `${absoluteTarget}.d.ts`));
  if (dtsRelSet.has(fileCandidate)) {
    return `${specifier}.js`;
  }

  const indexCandidate = toPosix(path.relative(distDir, path.join(absoluteTarget, "index.d.ts")));
  if (dtsRelSet.has(indexCandidate)) {
    return `${specifier}/index.js`;
  }

  return specifier;
}

function rewriteDtsRelativeSpecifiers(distDir) {
  const dtsRelSet = originalDtsRelSet(distDir);
  for (const dtsFile of walkFiles(distDir, (file) => file.endsWith(".d.ts"))) {
    const source = readFileSync(dtsFile, "utf8");
    const rewritten = source.replace(
      /\b(from|import)\s*\(\s*(["'])(\.{1,2}\/[^"']+)\2\s*\)|\b(from)\s+(["'])(\.{1,2}\/[^"']+)\5/g,
      (
        match,
        dynamicKeyword,
        dynamicQuote,
        dynamicSpecifier,
        staticKeyword,
        staticQuote,
        staticSpecifier
      ) => {
        if (dynamicKeyword) {
          const nextSpecifier = resolveEsmDtsSpecifier(
            dynamicSpecifier,
            dtsFile,
            distDir,
            dtsRelSet
          );
          return `${dynamicKeyword}(${dynamicQuote}${nextSpecifier}${dynamicQuote})`;
        }
        const nextSpecifier = resolveEsmDtsSpecifier(staticSpecifier, dtsFile, distDir, dtsRelSet);
        return `${staticKeyword} ${staticQuote}${nextSpecifier}${staticQuote}`;
      }
    );
    if (rewritten !== source) {
      writeFileSync(dtsFile, rewritten, "utf8");
    }
  }
}

function writeDistEsmWrappers(distDir) {
  const cjsFiles = walkFiles(distDir, (file) => file.endsWith(".cjs"));
  for (const cjsFile of cjsFiles) {
    const wrapperPath = cjsFile.replace(/\.cjs$/, ".js");
    const dtsPath = cjsFile.replace(/\.cjs$/, ".d.ts");
    writeCjsInteropEntrypoint(
      wrapperPath,
      relativeImportSpecifier(wrapperPath, cjsFile),
      extractRuntimeExports(dtsPath),
      "cjs"
    );
  }
}

function copyCoreArtifacts() {
  if (!existsSync(rootDistDir) || !statSync(rootDistDir).isDirectory()) {
    throw new Error("Missing dist directory. Run `bun run build:ts` first.");
  }
  const targetDist = path.join(coreDir, "dist");
  resetDir(targetDist);
  cpSync(rootDistDir, targetDist, { recursive: true });
  rewriteDtsRelativeSpecifiers(targetDist);
  convertDistJsToCjs(targetDist);
  writeDistEsmWrappers(targetDist);
}

function copyRuntimeArtifacts() {
  if (!existsSync(rootNativeLoaderJs)) {
    throw new Error("Missing root native loader `index.js`.");
  }
  if (!existsSync(rootNativeLoaderDts)) {
    throw new Error("Missing root native type declarations `index.d.ts`.");
  }

  cpSync(rootNativeLoaderJs, path.join(runtimeNodeDir, "native-loader.cjs"));
  cpSync(rootNativeLoaderDts, path.join(runtimeNodeDir, "native-loader.d.ts"));
  rmSync(path.join(runtimeNodeDir, "native-loader.js"), { force: true });

  const rootFiles = readdirSync(rootDir);
  const nativeBins = rootFiles.filter((name) => name.endsWith(".node"));
  for (const bin of nativeBins) {
    const src = path.join(rootDir, bin);
    const dest = path.join(runtimeNodeDir, bin);

    // Avoid cpSync self-copy errors when a root .node symlink already points at runtime-node.
    const srcReal = existsSync(src) ? realpathSync(src) : src;
    const destReal = existsSync(dest) ? realpathSync(dest) : dest;
    if (srcReal === destReal) {
      continue;
    }
    cpSync(src, dest);
  }
}

function writeRuntimeEntrypoints() {
  const cjsEntrypoint = '"use strict";\n\nmodule.exports = require("./native-loader.cjs");\n';
  const dtsEntrypoint = 'export * from "./native-loader.js";\n';
  writeFileSync(path.join(runtimeNodeDir, "index.cjs"), cjsEntrypoint, "utf8");
  writeFileSync(path.join(runtimeNodeDir, "index.d.ts"), dtsEntrypoint, "utf8");
  writeCjsInteropEntrypoint(
    path.join(runtimeNodeDir, "index.js"),
    "./native-loader.cjs",
    extractRuntimeExports(path.join(runtimeNodeDir, "native-loader.d.ts")),
    "runtime"
  );
}

function writeCoreEntrypoints() {
  const cjsEntrypoint = '"use strict";\n\nmodule.exports = require("./dist/index.cjs");\n';
  const dtsEntrypoint = 'export * from "./dist/index.js";\n';
  writeFileSync(path.join(coreDir, "index.cjs"), cjsEntrypoint, "utf8");
  writeFileSync(path.join(coreDir, "index.d.ts"), dtsEntrypoint, "utf8");
  writeCjsInteropEntrypoint(
    path.join(coreDir, "index.js"),
    "./dist/index.cjs",
    extractRuntimeExports(path.join(coreDir, "dist/index.d.ts")),
    "core"
  );
}

function ensureContractsEntrypoints() {
  writeCjsInteropEntrypoint(
    path.join(contractsDir, "index.js"),
    "./index.cjs",
    extractRuntimeExports(path.join(contractsDir, "index.d.ts")),
    "contracts"
  );
}

function writeFeatureEntrypoints() {
  writeFileSync(
    path.join(mcpDir, "index.cjs"),
    `"use strict";

const mcp = require("@caliche/core/mcp");
const sse = require("@caliche/core/transport/sse");

module.exports = {
  LATEST_PROTOCOL_VERSION: mcp.LATEST_PROTOCOL_VERSION,
  SUPPORTED_PROTOCOL_VERSIONS: mcp.SUPPORTED_PROTOCOL_VERSIONS,
  McpServerBuilder: mcp.McpServerBuilder,
  McpServer: mcp.McpServer,
  InProcessMcpServerBuilder: mcp.InProcessMcpServerBuilder,
  InProcessMcpServer: mcp.InProcessMcpServer,
  encodeSseEvent: sse.encodeSseEvent,
  decodeSseChunk: sse.decodeSseChunk,
  decodeSseStream: sse.decodeSseStream,
};
`,
    "utf8"
  );
  writeFileSync(
    path.join(mcpDir, "index.js"),
    `export {
  InProcessMcpServer,
  InProcessMcpServerBuilder,
  LATEST_PROTOCOL_VERSION,
  McpServer,
  McpServerBuilder,
  SUPPORTED_PROTOCOL_VERSIONS,
} from "@caliche/core/mcp";
export { decodeSseChunk, decodeSseStream, encodeSseEvent } from "@caliche/core/transport/sse";
`,
    "utf8"
  );

  writeFileSync(
    path.join(hooksDir, "index.cjs"),
    `"use strict";

const builtinTools = require("@caliche/core/builtin-tools");
const hooks = require("@caliche/core/hooks");
const jsonlSession = require("@caliche/core/jsonl-session");
const mutationStyle = require("@caliche/core/mutation-style");
const session = require("@caliche/core/session");
const toolInspection = require("@caliche/core/tool-inspection");
const toolRegistry = require("@caliche/core/tool-registry");

module.exports = {
  AgentSession: session.AgentSession,
  HookMatcher: hooks.HookMatcher,
  PermissionMode: hooks.PermissionMode,
  MutationStyle: mutationStyle.MutationStyle,
  InspectionAction: toolInspection.InspectionAction,
  InspectionHookAdapter: toolInspection.InspectionHookAdapter,
  PathSecurityInspector: toolInspection.PathSecurityInspector,
  PermissionModeInspector: toolInspection.PermissionModeInspector,
  ToolInspectionManager: toolInspection.ToolInspectionManager,
  ToolRegistry: toolRegistry.ToolRegistry,
  JsonlSessionHeader: jsonlSession.JsonlSessionHeader,
  JsonlSessionEntry: jsonlSession.JsonlSessionEntry,
  JsonlSessionTree: jsonlSession.JsonlSessionTree,
  JsonlSessionStore: jsonlSession.JsonlSessionStore,
  defaultInspectionManager: toolInspection.defaultInspectionManager,
  parseJsonlSession: jsonlSession.parseJsonlSession,
  newEntryId: jsonlSession.newEntryId,
  createApplyPatchTool: mutationStyle.createApplyPatchTool,
  createCalculatorTool: builtinTools.createCalculatorTool,
  createFsTool: mutationStyle.createFsTool,
};
`,
    "utf8"
  );
  writeFileSync(
    path.join(hooksDir, "index.js"),
    `export { createCalculatorTool } from "@caliche/core/builtin-tools";
export { HookMatcher, PermissionMode } from "@caliche/core/hooks";
export {
  JsonlSessionEntry,
  JsonlSessionHeader,
  JsonlSessionStore,
  JsonlSessionTree,
  newEntryId,
  parseJsonlSession,
} from "@caliche/core/jsonl-session";
export { createApplyPatchTool, createFsTool, MutationStyle } from "@caliche/core/mutation-style";
export { AgentSession } from "@caliche/core/session";
export {
  defaultInspectionManager,
  InspectionAction,
  InspectionHookAdapter,
  PathSecurityInspector,
  PermissionModeInspector,
  ToolInspectionManager,
} from "@caliche/core/tool-inspection";
export { ToolRegistry } from "@caliche/core/tool-registry";
`,
    "utf8"
  );

  writeFileSync(
    path.join(aiSdkAdapterDir, "index.cjs"),
    `"use strict";

const adapter = require("@caliche/core/adapters/ai-sdk");

module.exports = {
  CalicheLanguageModel: adapter.CalicheLanguageModel,
  CalicheMcpBridge: adapter.CalicheMcpBridge,
  caliche: adapter.caliche,
};
`,
    "utf8"
  );
  writeFileSync(
    path.join(aiSdkAdapterDir, "index.js"),
    'export { caliche, CalicheLanguageModel, CalicheMcpBridge } from "@caliche/core/adapters/ai-sdk";\n',
    "utf8"
  );

  writeFileSync(
    path.join(langChainAdapterDir, "index.cjs"),
    `"use strict";

const adapter = require("@caliche/core/adapters/langchain");

module.exports = {
  LangChainAdapter: adapter.LangChainAdapter,
  LangChainChainAdapter: adapter.LangChainChainAdapter,
  LangChainEmbeddingProvider: adapter.LangChainEmbeddingProvider,
  LangChainLLMProvider: adapter.LangChainLLMProvider,
  LangChainMemoryAdapter: adapter.LangChainMemoryAdapter,
  LangChainRetrieverAdapter: adapter.LangChainRetrieverAdapter,
};
`,
    "utf8"
  );
  writeFileSync(
    path.join(langChainAdapterDir, "index.js"),
    `export {
  LangChainAdapter,
  LangChainChainAdapter,
  LangChainEmbeddingProvider,
  LangChainLLMProvider,
  LangChainMemoryAdapter,
  LangChainRetrieverAdapter,
} from "@caliche/core/adapters/langchain";
`,
    "utf8"
  );
}

function main() {
  ensureDir(coreDir);
  ensureDir(runtimeNodeDir);
  copyCoreArtifacts();
  copyRuntimeArtifacts();
  writeCoreEntrypoints();
  writeRuntimeEntrypoints();
  ensureContractsEntrypoints();
  writeFeatureEntrypoints();
}

main();
