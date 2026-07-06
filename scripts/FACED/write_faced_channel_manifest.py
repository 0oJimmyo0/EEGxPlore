#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import lmdb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from utils.faced_channel_manifest import build_faced_channel_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill FACED channel manifest into LMDB and sidecar JSON.")
    ap.add_argument("--dataset-root", required=True, help="Path to the FACED LMDB directory.")
    ap.add_argument(
        "--sidecar-path",
        default="",
        help="Optional explicit JSON sidecar path. Defaults to <dataset-root>/channel_manifest.json",
    )
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    sidecar_path = args.sidecar_path or os.path.join(dataset_root, "channel_manifest.json")
    manifest = build_faced_channel_manifest()

    os.makedirs(dataset_root, exist_ok=True)

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    env = lmdb.open(dataset_root, map_size=64 * 1024 * 1024, subdir=True, lock=True)
    with env.begin(write=True) as txn:
        txn.put(b"__channel_manifest__", json.dumps(manifest, ensure_ascii=True).encode("utf-8"))
        txn.put(b"__channel_names__", pickle.dumps(manifest["labram_channel_names"]))
        txn.put(b"channel_names", pickle.dumps(manifest["labram_channel_names"]))
        txn.put(b"ch_names", pickle.dumps(manifest["labram_channel_names"]))
    env.sync()
    env.close()

    print(f"[FACED manifest] wrote sidecar: {sidecar_path}")
    print(f"[FACED manifest] wrote LMDB keys into: {dataset_root}")
    print(f"[FACED manifest] slots={len(manifest['stored_channel_names'])} normalized={len(manifest['labram_channel_names'])}")
    print(f"[FACED manifest] tail stored={manifest['stored_channel_names'][-4:]} tail labram={manifest['labram_channel_names'][-4:]}")


if __name__ == "__main__":
    main()
