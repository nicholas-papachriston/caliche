#!/usr/bin/env python3
"""Local inference catalog fit example.

Probes host RAM/VRAM and prints which built-in catalog models are safe to load.
Use fit APIs to choose a local model that fits available resources.

Usage:
    uv run examples/local_inference_fit.py
"""

from __future__ import annotations

import sys

import caliche


def main() -> int:
    probe = caliche.probe_local_inference_host
    fit_catalog = caliche.fit_local_inference_catalog
    safe_models = caliche.safe_local_inference_models
    if probe is None or fit_catalog is None or safe_models is None:
        print(
            "Local inference fit APIs unavailable. "
            "Rebuild with the local-inference feature.",
            file=sys.stderr,
        )
        return 1

    host = probe()
    print("Host capabilities:")
    print(
        f"  RAM available: {host['available_ram_bytes']} / {host['total_ram_bytes']}"
    )
    if host.get("available_vram_bytes") is not None:
        print(f"  VRAM free: {host['available_vram_bytes']}")
    print(f"  Recommended device: {host['recommended_device']}")
    for note in host.get("notes") or []:
        print(f"  Note: {note}")

    print("\nCatalog fit:")
    for fit in fit_catalog():
        print(
            f"  [{fit['status']}] {fit['model_id']}  "
            f"n_ctx={fit['recommended_n_ctx']} max_safe={fit['max_safe_n_ctx']}"
        )

    safe = safe_models()
    print(
        f"\nSafe models ({len(safe)}): "
        + ", ".join(item["model_id"] for item in safe)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
