#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import lmdb
import pandas as pd


N_TRIALS = 15


def extract_session_id(text: str):
    m = re.search(r'(?:session|ses)\s*[_-]?(\d)', text.lower())
    if m:
        return m.group(1)
    m = re.search(r'\b([1-3])\b', text)
    if m:
        return m.group(1)
    return ''


def parse_labels_xlsx(label_xlsx: str):
    xls = pd.read_excel(label_xlsx, sheet_name=None)
    labels = {'1': None, '2': None, '3': None}

    # First pass: tabular session/trial/label.
    for _, df in xls.items():
        col_map = {str(c).strip().lower(): c for c in df.columns}
        has_session = any(k in col_map for k in ['session', 'session_id'])
        has_trial = any(k in col_map for k in ['trial', 'trial_id', 'trial_index'])
        has_label = any(k in col_map for k in ['label', 'emotion', 'emotion_label'])
        if not (has_session and has_trial and has_label):
            continue

        sc = col_map.get('session', col_map.get('session_id'))
        tc = col_map.get('trial', col_map.get('trial_id', col_map.get('trial_index')))
        lc = col_map.get('label', col_map.get('emotion', col_map.get('emotion_label')))
        bins = {'1': [None] * N_TRIALS, '2': [None] * N_TRIALS, '3': [None] * N_TRIALS}

        for _, row in df.iterrows():
            if pd.isna(row[sc]) or pd.isna(row[tc]) or pd.isna(row[lc]):
                continue
            sid = str(int(row[sc]))
            if sid not in bins:
                continue
            tid = int(row[tc])
            tid = tid - 1 if 1 <= tid <= N_TRIALS else tid
            if 0 <= tid < N_TRIALS:
                bins[sid][tid] = int(row[lc])

        for sid in ['1', '2', '3']:
            if all(v is not None for v in bins[sid]):
                labels[sid] = bins[sid]

    if all(labels[sid] is not None for sid in ['1', '2', '3']):
        return labels

    # Fallback: official Sheet1 legend + session emotion order rows.
    emotion_to_id = {}
    for sheet_name in xls.keys():
        mat = pd.read_excel(label_xlsx, sheet_name=sheet_name, header=None)
        for row in mat.itertuples(index=False):
            vals = [v for v in row if not pd.isna(v)]
            if not vals:
                continue
            str_vals = [str(v).strip() for v in vals]
            int_vals = []
            for v in vals:
                try:
                    int_vals.append(int(v))
                except Exception:
                    continue
            if len(int_vals) == 1 and len(str_vals) >= 1:
                name = ''
                for tok in str_vals:
                    low = tok.lower()
                    if low in {'label', 'movie orders for three sessions'}:
                        continue
                    if re.fullmatch(r'\d+', tok):
                        continue
                    if 'session' in low:
                        continue
                    name = tok
                    break
                if name:
                    emotion_to_id[name.lower()] = int_vals[0]

    if emotion_to_id:
        for sheet_name in xls.keys():
            mat = pd.read_excel(label_xlsx, sheet_name=sheet_name, header=None)
            for row in mat.itertuples(index=False):
                vals = [v for v in row if not pd.isna(v)]
                if not vals:
                    continue
                row_text = ' '.join(str(v) for v in vals)
                sid = extract_session_id(row_text)
                if sid not in labels or labels[sid] is not None:
                    continue

                emotion_seq = []
                for v in vals:
                    tok = str(v).strip().lower()
                    if tok in emotion_to_id:
                        emotion_seq.append(tok)

                if len(emotion_seq) >= N_TRIALS:
                    labels[sid] = [emotion_to_id[tok] for tok in emotion_seq[:N_TRIALS]]

    if not all(labels[sid] is not None for sid in ['1', '2', '3']):
        raise ValueError(f'Failed to parse official labels from {label_xlsx}')
    return labels


def parse_key(k: str):
    # New format: <subject>_<session>_tXX_gXXXXX
    if '_t' in k and '_g' in k:
        parts = k.split('_')
        if len(parts) >= 4:
            subject = parts[0]
            session = parts[1]
            t = parts[2].replace('t', '')
            g = parts[3].replace('g', '')
            trial = int(t) if t.isdigit() else -1
            seg = int(g) if g.isdigit() else -1
            return subject, session, trial, seg

    # Legacy fallback
    p = k.rsplit('-', 2)
    prefix = p[0] if len(p) == 3 else k
    fparts = prefix.split('_')
    subject = fparts[0] if fparts else ''
    session = fparts[1] if len(fparts) > 1 else ''
    trial = int(p[1]) if len(p) == 3 and str(p[1]).isdigit() else -1
    seg = int(p[2]) if len(p) == 3 and str(p[2]).isdigit() else -1
    return subject, session, trial, seg


def load_rows_from_lmdb(lmdb_path: str, split_manifest: str):
    db = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    if split_manifest:
        if split_manifest.endswith('.json'):
            with open(split_manifest, 'r', encoding='utf-8') as f:
                split_keys = json.load(f)
        else:
            with open(split_manifest, 'rb') as f:
                split_keys = pickle.load(f)
    else:
        with db.begin(write=False) as txn:
            split_keys = pickle.loads(txn.get(b'__keys__'))

    rows = []
    missing = 0
    with db.begin(write=False) as txn:
        for split in ['train', 'val', 'test']:
            for key in split_keys.get(split, []):
                k = key.decode() if isinstance(key, bytes) else str(key)
                raw = txn.get(k.encode())
                if raw is None:
                    missing += 1
                    continue
                obj = pickle.loads(raw)
                subject, session, trial, seg = parse_key(k)
                rows.append(
                    {
                        'split': split,
                        'key': k,
                        'subject': str(obj.get('subject', subject)),
                        'session': str(obj.get('session', session)),
                        'trial': int(obj.get('trial', trial)),
                        'segment_idx': int(obj.get('segment', seg)),
                        'assigned_label': int(obj.get('label', -1)),
                    }
                )
    return rows, missing


def count_table(rows, key_fields):
    out = Counter()
    for r in rows:
        k = tuple(r[f] for f in key_fields)
        out[k] += 1
    return out


def print_label_counts_by(rows, by_field, title):
    bucket = defaultdict(Counter)
    for r in rows:
        bucket[str(r[by_field])][int(r['assigned_label'])] += 1
    print(f'[audit] {title}')
    for k in sorted(bucket.keys(), key=lambda x: (len(x), x)):
        print(f'  {k}: {dict(sorted(bucket[k].items()))}')


def verify_trial_label_consistency(rows, official_labels):
    per_session_trial = defaultdict(set)
    for r in rows:
        sid = str(r['session'])
        tid = int(r['trial'])
        if sid in official_labels and 0 <= tid < N_TRIALS:
            per_session_trial[(sid, tid)].add(int(r['assigned_label']))

    mismatches = []
    for sid in ['1', '2', '3']:
        for tid in range(N_TRIALS):
            observed = sorted(per_session_trial.get((sid, tid), set()))
            expected = int(official_labels[sid][tid])
            if not observed:
                mismatches.append((sid, tid, expected, 'MISSING'))
            elif observed != [expected]:
                mismatches.append((sid, tid, expected, observed))

    return mismatches


def build_tiny_overfit_manifest(rows, output_json, per_class=100, n_subjects=2, seed=42):
    rng = __import__('random').Random(seed)

    train_rows = [r for r in rows if r['split'] == 'train']
    by_subject = defaultdict(list)
    for r in train_rows:
        by_subject[r['subject']].append(r)

    subjects = sorted(by_subject.keys())
    rng.shuffle(subjects)
    chosen_subjects = set(subjects[:max(1, n_subjects)])

    cand = [r for r in train_rows if r['subject'] in chosen_subjects]
    by_label = defaultdict(list)
    for r in cand:
        by_label[r['assigned_label']].append(r['key'])

    tiny_train = []
    for lb in sorted(by_label.keys()):
        keys = by_label[lb][:]
        rng.shuffle(keys)
        tiny_train.extend(keys[:per_class])

    manifest = {
        'train': sorted(tiny_train),
        'val': sorted(tiny_train),
        'test': sorted(tiny_train),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    print(f'[audit] tiny overfit manifest written: {output_json}')
    print(f'[audit] tiny subjects={sorted(chosen_subjects)} train_n={len(manifest["train"])}')


def main():
    ap = argparse.ArgumentParser(description='Hard audit for SEED-V data-contract and split balance.')
    ap.add_argument('--lmdb_path', type=str, required=True)
    ap.add_argument('--label_xlsx', type=str, required=True)
    ap.add_argument('--split_manifest', type=str, default='')
    ap.add_argument('--export_csv', type=str, default='')
    ap.add_argument('--tiny_overfit_manifest', type=str, default='')
    ap.add_argument('--tiny_per_class', type=int, default=100)
    ap.add_argument('--tiny_subjects', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    official = parse_labels_xlsx(args.label_xlsx)
    rows, missing = load_rows_from_lmdb(args.lmdb_path, args.split_manifest)
    if not rows:
        raise ValueError('No rows loaded from LMDB/manifest.')

    if args.export_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_csv)), exist_ok=True)
        with open(args.export_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['subject', 'session', 'trial', 'segment_idx', 'assigned_label', 'split', 'key'])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'[audit] wrote table csv: {args.export_csv}')

    print(f'[audit] total_rows={len(rows)} missing_manifest_keys_in_lmdb={missing}')

    # Step 1 aggregates
    print_label_counts_by(rows, 'split', 'class counts by split')
    print_label_counts_by(rows, 'subject', 'class counts by subject')
    print_label_counts_by(rows, 'session', 'class counts by session')
    print_label_counts_by(rows, 'trial', 'class counts by trial')

    trial_segments = defaultdict(list)
    for r in rows:
        trial_segments[(r['subject'], r['session'], r['trial'], r['assigned_label'])].append(r['segment_idx'])

    seg_per_trial_by_label = defaultdict(list)
    for (_, _, _, lb), segs in trial_segments.items():
        seg_per_trial_by_label[int(lb)].append(len(segs))
    mean_seg = {
        lb: (sum(v) / len(v) if v else 0.0)
        for lb, v in sorted(seg_per_trial_by_label.items())
    }
    print(f'[audit] mean segments per trial by label={mean_seg}')

    # Step 2 consistency checks
    mismatches = verify_trial_label_consistency(rows, official)
    if mismatches:
        print(f'[audit] OFFICIAL_LABEL_MISMATCH count={len(mismatches)}')
        for sid, tid, exp, obs in mismatches[:20]:
            print(f'  session={sid} trial={tid} expected={exp} observed={obs}')
    else:
        print('[audit] official metadata mapping vs LMDB labels: PASS')

    # Optional Step 3 helper
    if args.tiny_overfit_manifest:
        build_tiny_overfit_manifest(
            rows,
            args.tiny_overfit_manifest,
            per_class=args.tiny_per_class,
            n_subjects=args.tiny_subjects,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
