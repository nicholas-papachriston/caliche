# Binary formats

Caliche stores its current on-disk and cache formats as rkyv encodings, version 1, each starting with a magic prefix that identifies the format. Session snapshot files use the `CLXSSN03` prefix. `decode_session_snapshot_v1` also reads the earlier `CLXSSN02` layout, which has no diff state, and fills that state with empty values.

| Magic prefix | Format | Code |
| --- | --- | --- |
| `CLXSSN03` | Session snapshot files (`.cls1`) | `agent::session::codec` |
| `STRMV001` | Durable memory-mapped stream files (`durable-streams-mmap` feature) | `streams` |
| `CL\x01\x01` | Model response cache values | `llm::cache::codec` |
| `CL\x02\x01` | Redis `SessionData` values | `data::persistence::redis_codec` |
| `CL\x03\x01` | Redis `TrackedSession` registry values | `data::persistence::redis_codec` |
| `CL\x04\x01` | Distributed session affinity values in Redis or Valkey | `distributed::session_manager` |
| `DRIPST01` | Dripstone memory-mapped vector slab, aligned | `dripstone` |
| `DRIPST02` | Dripstone memory-mapped vector slab, int8 quantized | `dripstone` |

The shared helpers and constants for the 4-byte prefixes are in `data::persistence::rkyv_envelope`. Register any new 4-byte prefix there.

## Session snapshots

`BinaryFileSessionStore`, `AgentServiceConfig::with_session_autosave_dir`, and `AgentService::enable_session_autosave` write `.cls1` files.

To inspect a snapshot file from the CLI:

```bash
caliche session inspect-file <path>
```

## Dripstone memory-mapped index

The Dripstone memory-mapped backend stores vectors in a `DRIPST01` slab (aligned) or a `DRIPST02` slab (int8 quantized). See [Dripstone indexing](../../dripstone/docs/indexing.md).
