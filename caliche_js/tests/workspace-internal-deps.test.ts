import { describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

// Every `@caliche/*` dependency must resolve inside this workspace. A version
// specifier would let the package manager fetch that name from the public
// registry, where anyone can publish it.
const ROOT = join(import.meta.dir, "..");
const DEPENDENCY_FIELDS = [
  "dependencies",
  "optionalDependencies",
  "peerDependencies",
  "devDependencies",
] as const;

type Manifest = { name?: string; workspaces?: string[] } & Partial<
  Record<(typeof DEPENDENCY_FIELDS)[number], Record<string, string>>
>;

function readManifest(dir: string): Manifest {
  return JSON.parse(readFileSync(join(dir, "package.json"), "utf8")) as Manifest;
}

function workspaceDirs(): string[] {
  const dirs = [ROOT];
  for (const pattern of readManifest(ROOT).workspaces ?? []) {
    const parent = join(ROOT, pattern.replace(/\/\*$/, ""));
    for (const entry of readdirSync(parent, { withFileTypes: true })) {
      const dir = join(parent, entry.name);
      if (entry.isDirectory() && existsSync(join(dir, "package.json"))) {
        dirs.push(dir);
      }
    }
  }
  return dirs;
}

describe("internal @caliche packages", () => {
  test("every @caliche dependency uses a workspace: specifier", () => {
    const violations: string[] = [];
    for (const dir of workspaceDirs()) {
      const manifest = readManifest(dir);
      for (const field of DEPENDENCY_FIELDS) {
        for (const [name, spec] of Object.entries(manifest[field] ?? {})) {
          if (name.startsWith("@caliche/") && !spec.startsWith("workspace:")) {
            violations.push(`${manifest.name} ${field} ${name}@${spec}`);
          }
        }
      }
    }
    expect(violations).toEqual([]);
  });

  test("every platform package in npm/ is a workspace", () => {
    const workspaces = readManifest(ROOT).workspaces ?? [];
    expect(workspaces).toContain("npm/*");
  });
});
