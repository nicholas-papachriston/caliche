import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  JsonlSessionEntry,
  JsonlSessionHeader,
  JsonlSessionStore,
  JsonlSessionTree,
  newEntryId,
  parseJsonlSession,
} from "../src/index";

describe("jsonl session", () => {
  test("jsonl session branch restore", () => {
    const dir = mkdtempSync(join(tmpdir(), "caliche-js-jsonl-"));
    try {
      const path = join(dir, "chat.jsonl");
      const store = new JsonlSessionStore(path);
      store.create(new JsonlSessionHeader("sess", "/proj"));
      store.appendEntry(JsonlSessionEntry.message("u1", { role: "user", content: "one" }, null));
      store.appendEntry(
        JsonlSessionEntry.message("a1", { role: "assistant", content: "two" }, "u1")
      );

      const tree = store.load();
      const branch = tree.branchFrom("u1", "br", "retry");
      const entries = tree.entries();
      store.appendEntry(entries[entries.length - 1]!);
      store.appendEntry(
        JsonlSessionEntry.message("a2", { role: "assistant", content: "alt" }, branch)
      );

      const reloaded = store.load();
      const main = reloaded.restoreBranchMessages("a1");
      const alt = reloaded.restoreBranchMessages("a2");
      expect(main.length).toBe(2);
      expect(alt.length).toBe(2);
      expect(main[1]?.content).toBe("two");
      expect(alt[1]?.content).toBe("alt");
      expect(branch).toBe("br");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("parse jsonl session and new entry id", () => {
    const entryId = newEntryId();
    expect(typeof entryId).toBe("string");
    expect(entryId.length).toBeGreaterThanOrEqual(4);

    const text = [
      '{"type":"session","version":3,"id":"s1","timestamp":"2024-01-01T00:00:00Z"}',
      '{"type":"message","id":"m1","parentId":null,"timestamp":"2024-01-01T00:00:01Z","message":{"role":"user","content":"hi"}}',
      "",
    ].join("\n");
    const tree = parseJsonlSession(text);
    expect(tree.header.id).toBe("s1");
    expect(tree.entryCount()).toBe(1);
    expect(tree.leaves()).toEqual(["m1"]);
  });

  test("jsonl session tree rejects unknown parent", () => {
    const tree = new JsonlSessionTree(new JsonlSessionHeader("s"));
    expect(() =>
      tree.append(JsonlSessionEntry.message("x", { role: "user", content: "z" }, "missing"))
    ).toThrow(/unknown parentId/);
  });

  test("jsonl entry factories cover compaction and model change", () => {
    const header = new JsonlSessionHeader("s");
    header.setParentSession("parent-sess");
    expect(header.parentSession).toBe("parent-sess");

    const tree = new JsonlSessionTree(header);
    tree.append(JsonlSessionEntry.message("m1", { role: "user", content: "hi" }, null));
    tree.append(JsonlSessionEntry.compactionSummary("c1", "summarized", "m1", 120));
    tree.append(JsonlSessionEntry.modelChange("mc1", "openai", "gpt-4o", "c1"));
    expect(tree.entryCount()).toBe(3);
    expect(tree.leaves()).toEqual(["mc1"]);
  });
});
