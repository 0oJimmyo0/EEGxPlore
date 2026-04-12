#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict

import lmdb


def parse_subject_from_key(key: str):
    return key.split('_')[0] if '_' in key else ''


def load_rows(meta_jsonl: str, lmdb_path: str):
    rows = []
    if meta_jsonl and os.path.isfile(meta_jsonl):
        with open(meta_jsonl, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                obj = json.loads(ln)
                rows.append(
                    {
                        'key': str(obj['key']),
                        'subject': str(obj.get('subject', parse_subject_from_key(str(obj['key'])))),
                        'label': int(obj.get('label', -1)),
                    }
                )
        return rows

    db = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        keys_blob = txn.get(b'__keys__')
        if keys_blob is None:
            raise KeyError('LMDB missing __keys__ index')
        split_keys = pickle.loads(keys_blob)

    all_keys = []
    for split in ['train', 'val', 'test']:
        all_keys.extend(split_keys.get(split, []))

    with db.begin(write=False) as txn:
        for key in all_keys:
            enc_key = key.encode() if isinstance(key, str) else key
            raw = txn.get(enc_key)
            if raw is None:
                continue
            pair = pickle.loads(raw)
            k = key.decode() if isinstance(key, bytes) else str(key)
            rows.append(
                {
                    'key': k,
                    'subject': str(pair.get('subject', parse_subject_from_key(k))),
                    'label': int(pair.get('label', -1)),
                }
            )

    return rows


def split_subjects(subjects, seed, train_ratio, val_ratio):
    rng = random.Random(seed)
    s = sorted(subjects)
    rng.shuffle(s)

    n = len(s)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError('Invalid subject split sizes; adjust train_ratio/val_ratio.')

    train = set(s[:n_train])
    val = set(s[n_train:n_train + n_val])
    test = set(s[n_train + n_val:])
    return train, val, test


def split_subjects_label_balanced(rows, seed, train_ratio, val_ratio):
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r['subject']].append(r)

    subjects = sorted([s for s in by_subject if s])
    if len(subjects) < 3:
        raise ValueError(f'Need at least 3 subjects, found {len(subjects)}.')

    subj_label = {}
    subj_total = {}
    all_labels = set()
    for sid in subjects:
        c = Counter(int(x['label']) for x in by_subject[sid])
        subj_label[sid] = c
        subj_total[sid] = sum(c.values())
        all_labels.update(c.keys())
    all_labels = sorted(all_labels)

    total_segments = sum(subj_total.values())
    total_labels = Counter()
    for sid in subjects:
        total_labels.update(subj_label[sid])

    target_ratio = {
        'train': train_ratio,
        'val': val_ratio,
        'test': 1.0 - train_ratio - val_ratio,
    }
    target_seg = {k: target_ratio[k] * total_segments for k in target_ratio}
    target_label = {
        k: {lb: target_ratio[k] * total_labels[lb] for lb in all_labels}
        for k in target_ratio
    }

    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    order = sorted(shuffled, key=lambda s: subj_total[s], reverse=True)

    split_subjects = {'train': [], 'val': [], 'test': []}
    split_seg = {'train': 0, 'val': 0, 'test': 0}
    split_label = {
        'train': Counter(),
        'val': Counter(),
        'test': Counter(),
    }

    # Seed each split with one subject first for stability.
    for split_name, sid in zip(['train', 'val', 'test'], order[:3]):
        split_subjects[split_name].append(sid)
        split_seg[split_name] += subj_total[sid]
        split_label[split_name].update(subj_label[sid])

    def score(split_name, sid):
        seg_after = split_seg[split_name] + subj_total[sid]
        seg_term = ((seg_after - target_seg[split_name]) / max(1.0, target_seg[split_name])) ** 2
        label_term = 0.0
        for lb in all_labels:
            after = split_label[split_name][lb] + subj_label[sid][lb]
            tgt = max(1.0, target_label[split_name][lb])
            label_term += ((after - tgt) / tgt) ** 2
        # Emphasize label balance over split size.
        return 0.7 * label_term + 0.3 * seg_term

    for sid in order[3:]:
        best_split = min(['train', 'val', 'test'], key=lambda sp: score(sp, sid))
        split_subjects[best_split].append(sid)
        split_seg[best_split] += subj_total[sid]
        split_label[best_split].update(subj_label[sid])

    return {
        'train': set(split_subjects['train']),
        'val': set(split_subjects['val']),
        'test': set(split_subjects['test']),
    }


def summarize_split(name, keys, rows_by_key, split_subjects_set):
    label_counter = Counter()
    for k in keys:
        label_counter[rows_by_key[k]['label']] += 1
    print(f'[seedv-manifest] {name} subjects={sorted(split_subjects_set)}')
    print(f'[seedv-manifest] {name} segments={len(keys)} labels={dict(sorted(label_counter.items()))}')


def main():
    ap = argparse.ArgumentParser(description='Build deterministic subject-disjoint SEED-V split manifest.')
    ap.add_argument('--lmdb_path', type=str, required=True)
    ap.add_argument('--meta_jsonl', type=str, default='')
    ap.add_argument('--output_json', type=str, default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train_ratio', type=float, default=0.7)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--balance_labels', action='store_true', default=True,
                    help='Use deterministic subject-level label-aware split assignment (default: on).')
    ap.add_argument('--no-balance_labels', dest='balance_labels', action='store_false')
    args = ap.parse_args()

    if not os.path.isdir(args.lmdb_path):
        raise FileNotFoundError(f'LMDB path not found: {args.lmdb_path}')

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1.0:
        raise ValueError('Require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.0.')

    meta_jsonl = args.meta_jsonl or (args.lmdb_path.rstrip('/\\') + '_sample_meta.jsonl')
    output_json = args.output_json or os.path.join(args.lmdb_path, 'subject_disjoint_manifest.json')

    rows = load_rows(meta_jsonl, args.lmdb_path)
    if not rows:
        raise ValueError('No samples found to build manifest.')

    rows_by_key = {r['key']: r for r in rows}
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r['subject']].append(r['key'])

    subjects = sorted([s for s in by_subject if s])
    if len(subjects) < 3:
        raise ValueError(f'Need at least 3 subjects, found {len(subjects)}.')

    if args.balance_labels:
        split_sub = split_subjects_label_balanced(rows, args.seed, args.train_ratio, args.val_ratio)
        train_sub, val_sub, test_sub = split_sub['train'], split_sub['val'], split_sub['test']
    else:
        train_sub, val_sub, test_sub = split_subjects(subjects, args.seed, args.train_ratio, args.val_ratio)

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

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    tv = len(train_sub & val_sub)
    tt = len(train_sub & test_sub)
    vt = len(val_sub & test_sub)

    print(f'[seedv-manifest] wrote {output_json}')
    print(f'[seedv-manifest] balance_labels={args.balance_labels}')
    summarize_split('train', manifest['train'], rows_by_key, train_sub)
    summarize_split('val', manifest['val'], rows_by_key, val_sub)
    summarize_split('test', manifest['test'], rows_by_key, test_sub)
    print(f'[seedv-manifest] overlap_check train/val={tv} train/test={tt} val/test={vt}')


if __name__ == '__main__':
    main()
