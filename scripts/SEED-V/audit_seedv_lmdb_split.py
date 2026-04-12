#!/usr/bin/env python3
import argparse
import collections
import os
import pickle
import re

import lmdb


def parse_key(key):
    k = key.decode() if isinstance(key, bytes) else str(key)

    # New format: <subject>_<session>_tXX_gXXXXX
    m = re.match(r'^([^_]+)_([^_]+)_t(\d+)_g(\d+)$', k)
    if m:
        return {
            'key': k,
            'subject': m.group(1),
            'session': m.group(2),
            'trial_id': int(m.group(3)),
            'segment_id': int(m.group(4)),
        }

    # Legacy fallback: <subject>_<session>-<trial>-<segment>
    parts = k.rsplit('-', 2)
    prefix = parts[0] if len(parts) == 3 else k
    trial_id = int(parts[1]) if len(parts) == 3 and str(parts[1]).isdigit() else -1
    segment_id = int(parts[2]) if len(parts) == 3 and str(parts[2]).isdigit() else -1

    fparts = prefix.split('_')
    subject = fparts[0] if len(fparts) >= 1 else ''
    session = fparts[1] if len(fparts) >= 2 else ''
    return {
        'key': k,
        'subject': subject,
        'session': session,
        'trial_id': trial_id,
        'segment_id': segment_id,
    }


def main():
    ap = argparse.ArgumentParser(description='Inspect SEED-V LMDB split protocol and schema.')
    ap.add_argument('--lmdb_path', type=str, required=True)
    ap.add_argument('--strict_subject_disjoint', action='store_true',
                    help='Exit with code 2 if train/val/test share any subject ids.')
    args = ap.parse_args()

    if not os.path.isdir(args.lmdb_path):
        raise FileNotFoundError(f'LMDB path not found: {args.lmdb_path}')

    db = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        keys_blob = txn.get(b'__keys__')
        if keys_blob is None:
            raise KeyError('LMDB missing __keys__ index')
        split_keys = pickle.loads(keys_blob)

    required = ['train', 'val', 'test']
    missing = [s for s in required if s not in split_keys]
    if missing:
        raise KeyError(f'Missing split(s) in __keys__: {missing}')

    print(f'[audit] lmdb_path={args.lmdb_path}')
    print(f'[audit] splits={list(split_keys.keys())}')

    split_subjects = {}
    for split in required:
        keys = split_keys[split]
        subjects = set()
        sessions = collections.Counter()
        trials = collections.Counter()
        for key in keys:
            p = parse_key(key)
            if p['subject']:
                subjects.add(p['subject'])
            if p['session']:
                sessions[p['session']] += 1
            if p['trial_id'] != -1:
                trials[p['trial_id']] += 1

        split_subjects[split] = subjects
        print(
            f'[audit] split={split} n={len(keys)} '
            f'unique_subjects={len(subjects)} '
            f'sessions={dict(sorted(sessions.items()))} '
            f'trial_ids={sorted(trials.keys())}'
        )

        if keys:
            k = keys[0]
            kb = k.encode() if isinstance(k, str) else k
            with db.begin(write=False) as txn:
                obj = pickle.loads(txn.get(kb))
            shape = getattr(obj.get('sample', None), 'shape', None)
            dtype = getattr(obj.get('sample', None), 'dtype', None)
            print(f"[audit] example[{split}] key={parse_key(k)['key']} shape={shape} dtype={dtype} label={obj.get('label')}")

    inter_tv = split_subjects['train'] & split_subjects['val']
    inter_tt = split_subjects['train'] & split_subjects['test']
    inter_vt = split_subjects['val'] & split_subjects['test']

    print(f'[audit] subject_overlap train/val={len(inter_tv)} train/test={len(inter_tt)} val/test={len(inter_vt)}')

    if args.strict_subject_disjoint and (inter_tv or inter_tt or inter_vt):
        print('[audit] FAIL: split is not subject-disjoint')
        raise SystemExit(2)

    print('[audit] done')


if __name__ == '__main__':
    main()
