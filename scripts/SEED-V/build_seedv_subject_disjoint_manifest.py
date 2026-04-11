#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random
from collections import defaultdict

import lmdb


def parse_subject(key):
    k = key.decode() if isinstance(key, bytes) else str(key)
    return k.split('_')[0] if '_' in k else ''


def main():
    ap = argparse.ArgumentParser(description='Build a subject-disjoint SEED-V split manifest from an LMDB.')
    ap.add_argument('--lmdb_path', type=str, required=True)
    ap.add_argument('--output_json', type=str, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train_subject_ratio', type=float, default=0.7)
    ap.add_argument('--val_subject_ratio', type=float, default=0.15)
    args = ap.parse_args()

    if not os.path.isdir(args.lmdb_path):
        raise FileNotFoundError(f'LMDB path not found: {args.lmdb_path}')

    if args.train_subject_ratio <= 0 or args.val_subject_ratio <= 0:
        raise ValueError('Ratios must be positive.')
    if args.train_subject_ratio + args.val_subject_ratio >= 1.0:
        raise ValueError('train_subject_ratio + val_subject_ratio must be < 1.0.')

    db = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        keys_blob = txn.get(b'__keys__')
        if keys_blob is None:
            raise KeyError('LMDB missing __keys__ index')
        split_keys = pickle.loads(keys_blob)

    all_keys = []
    for split_name in ['train', 'val', 'test']:
        all_keys.extend(split_keys.get(split_name, []))

    by_subject = defaultdict(list)
    for key in all_keys:
        k = key.decode() if isinstance(key, bytes) else str(key)
        sid = parse_subject(k)
        by_subject[sid].append(k)

    subjects = sorted([s for s in by_subject.keys() if s])
    if len(subjects) < 3:
        raise ValueError(f'Need at least 3 subjects, found {len(subjects)}')

    rng = random.Random(args.seed)
    rng.shuffle(subjects)

    n_sub = len(subjects)
    n_train = max(1, int(round(n_sub * args.train_subject_ratio)))
    n_val = max(1, int(round(n_sub * args.val_subject_ratio)))
    if n_train + n_val >= n_sub:
        n_val = max(1, n_sub - n_train - 1)
    n_test = n_sub - n_train - n_val
    if n_test <= 0:
        raise ValueError('Invalid split sizes; adjust ratios.')

    train_sub = set(subjects[:n_train])
    val_sub = set(subjects[n_train:n_train + n_val])
    test_sub = set(subjects[n_train + n_val:])

    manifest = {'train': [], 'val': [], 'test': []}
    for sid, keys in by_subject.items():
        if sid in train_sub:
            manifest['train'].extend(keys)
        elif sid in val_sub:
            manifest['val'].extend(keys)
        elif sid in test_sub:
            manifest['test'].extend(keys)

    for split in ['train', 'val', 'test']:
        manifest[split] = sorted(manifest[split])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    print(f'[manifest] wrote {args.output_json}')
    print(f"[manifest] subject counts train={len(train_sub)} val={len(val_sub)} test={len(test_sub)}")
    print(f"[manifest] sample counts train={len(manifest['train'])} val={len(manifest['val'])} test={len(manifest['test'])}")


if __name__ == '__main__':
    main()
