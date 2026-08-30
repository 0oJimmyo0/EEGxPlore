#!/usr/bin/env python3
"""Verify the legacy CBraMod data contract used by ICASSP revision runs.

This is intentionally a launch-time structural check, not a replacement for
the full offline dataset audits in ``experiments/icassp2027/manifests``.  It
checks the artifact that a job will actually open, its split source and
representative tensor schema, and that the active EEGxPlore loader applies the
historical single ``/100`` scaling exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LMDB_CONTRACTS = {
    "SEED-V": {
        "shape": (62, 1, 200),
        "dtype": "float64",
        "classes": 5,
        "split_counts": {"train": 34432, "val": 42960, "test": 40352},
        "split_key_sha256": {
            "train": "9b9a40b936651a7b4ffb42effdca690683c5a2c7379dd1138a68e1cf4ec0fe3f",
            "val": "1765d5569a0d37dd80a968b0fdc049c4d569ff24c8e088070f59b48a8a24b8e4",
            "test": "c6b9b37c851de1a7b668b5a950dd18e6233962ed6254e6ea9777f88524acebe1",
        },
        "split_source": "LMDB __keys__ (within-subject trial 0-4/5-9/10-14)",
    },
    "FACED": {
        "shape": (32, 10, 200),
        "dtype": "float64",
        "classes": 9,
        "split_counts": {"train": 6720, "val": 1680, "test": 1932},
        "split_key_sha256": {
            "train": "a5bad0a15d286d8de7bfb64a01bd9366d6ab274635c4c8bc0b6e0532c680e47f",
            "val": "cd7e2d3049ebe7a47622503d5268d66057e5b462223a24bff56c26ab0900eb3b",
            "test": "937f0c3f7cf2f869c10f324da7cb883914e8b870ed1cd5ad9a83a16ccefc3cbd",
        },
        "split_source": "LMDB __keys__ (rejected-paper artifact split)",
    },
    "PhysioNet-MI": {
        "shape": (64, 4, 200),
        "dtype": "float64",
        "classes": 4,
        "split_counts": {"train": 6843, "val": 1464, "test": 1530},
        "split_key_sha256": {
            "train": "db96697ae1a38ea7cf8f6f5d10cee603ba6240749432b4c74d8d8e571f2316fe",
            "val": "d45a6bed1cf076dd87900cb6e6b5820d0acdd0d93c01f64963dbb5a142863365",
            "test": "14a23eef4ff903850711647b71ffe09cdaf26cd9ae1d067554ce1e6cc09c128c",
        },
        "split_source": "repository-frozen subject-disjoint split manifest",
        "split_manifest_sha256": "71344f5bf12edfafedee53da7247ad10f7f8f7b678abbe084c86b8f531133601",
    },
}


def _jsonable_shape(value):
    return [int(x) for x in value]


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_sha256(keys) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in keys:
        encoded = key.encode() if isinstance(key, str) else bytes(key)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _check_lmdb(dataset: str, data_dir: str, split_manifest_path: str = "") -> dict:
    contract = LMDB_CONTRACTS[dataset]
    db = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        manifest_sha256 = ""
        if split_manifest_path:
            manifest_path = Path(split_manifest_path).resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"{dataset} split manifest does not exist: {manifest_path}")
            manifest_sha256 = _sha256_file(str(manifest_path))
            expected_manifest_sha256 = contract.get("split_manifest_sha256")
            if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
                raise RuntimeError(
                    f"{dataset} split manifest hash differs from the locked contract: "
                    f"expected={expected_manifest_sha256} got={manifest_sha256}"
                )
            try:
                split_index = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{dataset} split manifest is not valid JSON: {manifest_path}") from exc
            if not isinstance(split_index, dict):
                raise RuntimeError(f"{dataset} split manifest must contain a JSON object")
        else:
            with db.begin(write=False) as txn:
                raw_index = txn.get(b"__keys__")
                if raw_index is None:
                    raise RuntimeError(f"{dataset} LMDB is missing __keys__: {data_dir}")
                split_index = pickle.loads(raw_index)

        split_counts = {
            split: len(split_index.get(split, []))
            for split in ("train", "val", "test")
        }
        if split_counts != contract["split_counts"]:
            raise RuntimeError(
                f"{dataset} split counts differ from the locked artifact: "
                f"expected={contract['split_counts']} got={split_counts}"
            )
        split_key_sha256 = {
            split: _key_sha256(split_index[split])
            for split in ("train", "val", "test")
        }
        if split_key_sha256 != contract["split_key_sha256"]:
            raise RuntimeError(
                f"{dataset} split key hashes differ from the locked artifact: "
                f"expected={contract['split_key_sha256']} got={split_key_sha256}"
            )

        with db.begin(write=False) as txn:
            split_counts = {
                split: len(split_index.get(split, []))
                for split in ("train", "val", "test")
            }
            representative = {}
            for split in ("train", "val", "test"):
                key = split_index[split][0]
                encoded = key.encode() if isinstance(key, str) else bytes(key)
                raw_record = txn.get(encoded)
                if raw_record is None:
                    raise RuntimeError(f"{dataset} missing representative key {key!r}")
                record = pickle.loads(raw_record)
                if not isinstance(record, dict) or "sample" not in record or "label" not in record:
                    raise RuntimeError(f"{dataset} record {key!r} lacks sample/label fields")
                sample = np.asarray(record["sample"])
                label = int(np.asarray(record["label"]).reshape(-1)[0])
                if tuple(sample.shape) != contract["shape"]:
                    raise RuntimeError(
                        f"{dataset} representative shape mismatch: "
                        f"expected={contract['shape']} got={tuple(sample.shape)}"
                    )
                if str(sample.dtype) != contract["dtype"]:
                    raise RuntimeError(
                        f"{dataset} representative dtype mismatch: "
                        f"expected={contract['dtype']} got={sample.dtype}"
                    )
                if not np.isfinite(sample).all():
                    raise RuntimeError(f"{dataset} representative sample contains NaN/Inf")
                if not 0 <= label < contract["classes"]:
                    raise RuntimeError(f"{dataset} representative label outside class range: {label}")
                representative[split] = {
                    "key": key.decode() if isinstance(key, bytes) else str(key),
                    "shape": _jsonable_shape(sample.shape),
                    "dtype": str(sample.dtype),
                    "label": label,
                }
    finally:
        db.close()

    return {
        "storage": "lmdb",
        "split_source": contract["split_source"],
        "split_manifest_sha256": manifest_sha256,
        "split_counts": split_counts,
        "split_key_sha256": split_key_sha256,
        "representative_records": representative,
    }


def _check_loader_scale(dataset: str, data_dir: str, split_manifest_path: str = "") -> dict:
    """Prove the active EEGxPlore loader returns exactly raw_sample / 100."""
    if dataset == "SEED-V":
        from datasets.seedv_dataset import CustomDataset

        loader_dataset = CustomDataset(data_dir, mode="train")
    elif dataset == "FACED":
        from datasets.faced_dataset import CustomDataset

        loader_dataset = CustomDataset(data_dir, mode="train", input_scale_divisor=100.0)
    elif dataset == "PhysioNet-MI":
        from datasets.physio_dataset import CustomDataset

        loader_dataset = CustomDataset(
            data_dir, mode="train", split_manifest_path=split_manifest_path
        )
    else:
        raise ValueError(f"No LMDB loader scale check for {dataset}")

    key = loader_dataset.keys[0]
    encoded = key.encode() if isinstance(key, str) else bytes(key)
    db = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with db.begin(write=False) as txn:
            record = pickle.loads(txn.get(encoded))
            raw_sample = np.asarray(record["sample"])
    finally:
        db.close()
    loaded_sample = np.asarray(loader_dataset[0][0])
    exact = bool(np.array_equal(loaded_sample, raw_sample / 100.0))
    if not exact:
        raise RuntimeError(
            f"{dataset} loader scaling is not exactly raw_sample / 100; "
            "check for missing or duplicated normalization."
        )
    return {
        "stored_scale": "raw serialized values",
        "loader_scale": "/100",
        "verified_exact_raw_div100": exact,
        "loaded_shape": _jsonable_shape(loaded_sample.shape),
    }


def _check_isruc(data_dir: str) -> dict:
    root = Path(data_dir)
    seq_root = root / "seq"
    labels_root = root / "labels"
    if not seq_root.is_dir() or not labels_root.is_dir():
        raise RuntimeError(f"ISRUC requires seq/ and labels/ under {data_dir}")

    sequence_count = 0
    split_sequences = {"train": 0, "val": 0, "test": 0}
    first_pair = None
    for subject in range(1, 101):
        seq_dir = seq_root / f"ISRUC-group1-{subject}"
        label_dir = labels_root / f"ISRUC-group1-{subject}"
        if not seq_dir.is_dir() or not label_dir.is_dir():
            raise RuntimeError(f"ISRUC missing subject pair for subject {subject}")
        seq_files = sorted(seq_dir.glob("*.npy"))
        label_files = sorted(label_dir.glob("*.npy"))
        if len(seq_files) != len(label_files):
            raise RuntimeError(
                f"ISRUC subject {subject} sequence/label count mismatch: "
                f"{len(seq_files)} vs {len(label_files)}"
            )
        split = "train" if subject <= 80 else ("val" if subject <= 90 else "test")
        split_sequences[split] += len(seq_files)
        sequence_count += len(seq_files)
        for seq_path, label_path in zip(seq_files, label_files):
            if seq_path.stem != label_path.stem:
                raise RuntimeError(f"ISRUC sequence/label basename mismatch: {seq_path} vs {label_path}")
        if first_pair is None and seq_files:
            first_pair = (seq_files[0], label_files[0])

    if sequence_count != 4462 or split_sequences != {"train": 3559, "val": 468, "test": 435}:
        raise RuntimeError(
            f"ISRUC sequence counts differ from the rejected-paper artifact: "
            f"total={sequence_count} splits={split_sequences}"
        )
    if first_pair is None:
        raise RuntimeError("ISRUC contains no sequence files")

    raw_seq = np.load(first_pair[0])
    raw_label = np.load(first_pair[1])
    if tuple(raw_seq.shape) != (20, 6, 6000) or tuple(raw_label.shape) != (20,):
        raise RuntimeError(
            f"ISRUC representative schema mismatch: signal={raw_seq.shape} label={raw_label.shape}"
        )
    if not np.isfinite(raw_seq).all() or not np.isfinite(raw_label).all():
        raise RuntimeError("ISRUC representative sample contains NaN/Inf")

    from datasets.isruc_dataset import CustomDataset

    loader_dataset = CustomDataset([(str(first_pair[0]), str(first_pair[1]))])
    loaded_seq = np.asarray(loader_dataset[0][0])
    exact = bool(np.array_equal(loaded_seq, raw_seq / 100.0))
    if not exact:
        raise RuntimeError("ISRUC loader scaling is not exactly raw_sequence / 100")
    return {
        "storage": "seq/labels numpy pairs",
        "split_source": "ordered subjects 1-80 / 81-90 / 91-100",
        "sequence_count": sequence_count,
        "split_sequence_counts": split_sequences,
        "representative_signal_shape": _jsonable_shape(raw_seq.shape),
        "representative_signal_dtype": str(raw_seq.dtype),
        "representative_label_shape": _jsonable_shape(raw_label.shape),
        "loader_scale": "/100",
        "verified_exact_raw_div100": exact,
    }


def verify(dataset: str, data_dir: str, split_manifest_path: str = "") -> dict:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    if dataset == "ISRUC":
        details = _check_isruc(data_dir)
    else:
        details = _check_lmdb(dataset, data_dir, split_manifest_path)
        details["scaling"] = _check_loader_scale(dataset, data_dir, split_manifest_path)
    return {
        "contract": (
            "icassp_physionet_mi_v1"
            if dataset == "PhysioNet-MI"
            else "rejected_paper_cbramod_primary"
        ),
        "dataset": dataset,
        "data_dir": os.path.realpath(data_dir),
        "input_scale_divisor": 100.0,
        "status": "pass",
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["SEED-V", "FACED", "ISRUC", "PhysioNet-MI"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = verify(args.dataset, args.data_dir, args.split_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
