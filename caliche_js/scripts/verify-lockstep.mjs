import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const workspaceRoot = resolve(import.meta.dirname, "..");
const packageJsonPaths = [
  "packages/contracts/package.json",
  "packages/core/package.json",
  "packages/runtime-node/package.json",
  "packages/mcp/package.json",
  "packages/hooks/package.json",
  "packages/adapters-ai-sdk/package.json",
  "packages/adapters-langchain/package.json",
];

const rootPackage = JSON.parse(readFileSync(resolve(workspaceRoot, "package.json"), "utf8"));
const expectedVersion = rootPackage.version;
if (!expectedVersion) {
  throw new Error("Missing version in caliche_js/package.json");
}

const packageName = rootPackage.napi?.packageName;
if (packageName !== "@caliche/runtime-node") {
  throw new Error(
    `Expected napi.packageName "@caliche/runtime-node", found ${String(packageName)}`
  );
}

const seenNames = new Set();

function exportTargetPath(packageJsonPath, target) {
  return resolve(packageJsonPath, "..", target);
}

function verifyTarget(packageName, packageJsonPath, exportPath, condition, target) {
  if (!existsSync(exportTargetPath(packageJsonPath, target))) {
    throw new Error(
      `Missing export target for ${packageName} ${exportPath} ${condition}: ${target}`
    );
  }
}

function verifyExportEntry(packageName, packageJsonPath, exportPath, exportEntry) {
  if (typeof exportEntry === "string") {
    verifyTarget(packageName, packageJsonPath, exportPath, "default", exportEntry);
    return;
  }
  if (!exportEntry || typeof exportEntry !== "object" || Array.isArray(exportEntry)) {
    throw new Error(`Invalid export entry for ${packageName} ${exportPath}`);
  }

  const targets = {
    types: exportEntry.types,
    import: exportEntry.import,
    require: exportEntry.require,
    default: exportEntry.default,
  };

  if (exportPath !== "./package.json") {
    for (const condition of ["import", "require"]) {
      if (!targets[condition]) {
        throw new Error(`Missing ${condition} export for ${packageName} ${exportPath}`);
      }
    }
  }

  for (const [condition, target] of Object.entries(targets)) {
    if (typeof target === "string") {
      verifyTarget(packageName, packageJsonPath, exportPath, condition, target);
    }
  }
}

for (const relativePath of packageJsonPaths) {
  const fullPath = resolve(workspaceRoot, relativePath);
  const parsed = JSON.parse(readFileSync(fullPath, "utf8"));
  if (seenNames.has(parsed.name)) {
    throw new Error(`Duplicate package name: ${parsed.name}`);
  }
  seenNames.add(parsed.name);
  if (parsed.version !== expectedVersion) {
    throw new Error(
      `Lockstep version mismatch for ${parsed.name}: expected ${expectedVersion} but found ${parsed.version}`
    );
  }
  if (!parsed.engines?.node) {
    throw new Error(`Missing engines.node for ${parsed.name}`);
  }
  if (parsed.type !== "module") {
    throw new Error(`Expected ${parsed.name} to be ESM-first with type=module`);
  }
  for (const [exportPath, exportEntry] of Object.entries(parsed.exports ?? {})) {
    verifyExportEntry(parsed.name, fullPath, exportPath, exportEntry);
  }
}

const loader = readFileSync(resolve(workspaceRoot, "index.js"), "utf8");
if (!loader.includes("@caliche/runtime-node-")) {
  throw new Error("Runtime loader should reference @caliche/runtime-node-<platform> packages");
}

const versionMatches = loader.match(/bindingPackageVersion !== '([^']+)'/g) ?? [];
if (versionMatches.length === 0) {
  throw new Error("Unable to validate runtime-node binding version check in index.js");
}

for (const match of versionMatches) {
  const literalVersion = match.split("'")[1];
  if (literalVersion !== expectedVersion) {
    throw new Error(
      `Runtime loader version mismatch: expected ${expectedVersion} but found ${literalVersion}`
    );
  }
}

console.log(`Lockstep check passed for ${packageJsonPaths.length} packages at ${expectedVersion}`);
