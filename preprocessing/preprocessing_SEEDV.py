import argparse
import json
import os
import pickle
import re
from collections import Counter

import lmdb
import mne
import numpy as np
import pandas as pd

USELESS_CH = ['M1', 'M2', 'VEO', 'HEO']
N_TRIALS = 15
TARGET_FS = 200
LOW_FREQ = 1.0
HIGH_FREQ = 75.0
PATCH_LEN = 200
N_PATCH = 4
SEGMENT_LEN = PATCH_LEN * N_PATCH


def _extract_numbers(line: str):
    vals = re.findall(r'-?\d+(?:\.\d+)?', line)
    return [float(v) for v in vals]


def _extract_session_id(text: str):
    m = re.search(r'(?:session|ses)\s*[_-]?(\d)', text.lower())
    if m:
        return m.group(1)
    m = re.search(r'\b([1-3])\b', text)
    if m:
        return m.group(1)
    return ''


def parse_trial_timestamps(timestamp_txt: str):
    if not os.path.isfile(timestamp_txt):
        raise FileNotFoundError(f'trial timestamp file not found: {timestamp_txt}')

    starts = {}
    ends = {}
    paired = {}
    all_nums = []

    with open(timestamp_txt, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        nums = _extract_numbers(ln)
        if not nums:
            continue
        all_nums.extend(nums)
        sid = _extract_session_id(ln)
        low = ln.lower()

        if sid and len(nums) >= 30:
            paired[sid] = nums[:30]
            continue
        if sid and 'start' in low and len(nums) >= N_TRIALS:
            starts[sid] = nums[:N_TRIALS]
            continue
        if sid and 'end' in low and len(nums) >= N_TRIALS:
            ends[sid] = nums[:N_TRIALS]
            continue

    sessions = {}
    for sid in ['1', '2', '3']:
        if sid in paired:
            vals = paired[sid]
            sessions[sid] = {
                'start': [int(round(v)) for v in vals[:N_TRIALS]],
                'end': [int(round(v)) for v in vals[N_TRIALS:2 * N_TRIALS]],
            }
        elif sid in starts and sid in ends:
            sessions[sid] = {
                'start': [int(round(v)) for v in starts[sid]],
                'end': [int(round(v)) for v in ends[sid]],
            }

    if len(sessions) < 3 and len(all_nums) >= 3 * 2 * N_TRIALS:
        flat = [int(round(v)) for v in all_nums[:3 * 2 * N_TRIALS]]
        for i, sid in enumerate(['1', '2', '3']):
            chunk = flat[i * 30:(i + 1) * 30]
            sessions[sid] = {'start': chunk[:N_TRIALS], 'end': chunk[N_TRIALS:2 * N_TRIALS]}

    if set(sessions.keys()) != {'1', '2', '3'}:
        raise ValueError(
            f'Failed to parse all session timestamps from {timestamp_txt}. Parsed sessions={sorted(sessions.keys())}'
        )

    return sessions


def parse_labels_xlsx(label_xlsx: str):
    if not os.path.isfile(label_xlsx):
        raise FileNotFoundError(f'label file not found: {label_xlsx}')

    xls = pd.read_excel(label_xlsx, sheet_name=None)
    labels = {'1': None, '2': None, '3': None}

    for sheet_name, df in xls.items():
        col_map = {str(c).strip().lower(): c for c in df.columns}
        has_session = any(k in col_map for k in ['session', 'session_id'])
        has_trial = any(k in col_map for k in ['trial', 'trial_id', 'trial_index'])
        has_label = any(k in col_map for k in ['label', 'emotion', 'emotion_label'])
        if has_session and has_trial and has_label:
            sc = col_map.get('session', col_map.get('session_id'))
            tc = col_map.get('trial', col_map.get('trial_id', col_map.get('trial_index')))
            lc = col_map.get('label', col_map.get('emotion', col_map.get('emotion_label')))
            sess_bins = {'1': [None] * N_TRIALS, '2': [None] * N_TRIALS, '3': [None] * N_TRIALS}
            for _, row in df.iterrows():
                if pd.isna(row[sc]) or pd.isna(row[tc]) or pd.isna(row[lc]):
                    continue
                sid = str(int(row[sc]))
                if sid not in sess_bins:
                    continue
                tid = int(row[tc])
                tid = tid - 1 if 1 <= tid <= N_TRIALS else tid
                if 0 <= tid < N_TRIALS:
                    sess_bins[sid][tid] = int(row[lc])
            for sid in ['1', '2', '3']:
                if all(v is not None for v in sess_bins[sid]):
                    labels[sid] = sess_bins[sid]

    if all(labels[sid] is not None for sid in ['1', '2', '3']):
        return labels

    row_candidates = []
    for sheet_name in xls.keys():
        mat = pd.read_excel(label_xlsx, sheet_name=sheet_name, header=None)
        for row in mat.itertuples(index=False):
            nums = []
            for v in row:
                if pd.isna(v):
                    continue
                try:
                    iv = int(v)
                except Exception:
                    continue
                nums.append(iv)
            if len(nums) >= N_TRIALS:
                sid_from_sheet = _extract_session_id(str(sheet_name))
                row_candidates.append((sid_from_sheet, nums[:N_TRIALS]))

    for sid in ['1', '2', '3']:
        for sid_hint, vals in row_candidates:
            if sid_hint == sid and labels[sid] is None:
                labels[sid] = vals

    fallback_rows = [vals for _, vals in row_candidates]
    fill_sids = [sid for sid in ['1', '2', '3'] if labels[sid] is None]
    for sid, vals in zip(fill_sids, fallback_rows):
        labels[sid] = vals

    # Fallback for the official SEED-V Sheet1 layout:
    # - top legend rows define emotion -> numeric label
    # - session rows list 15 emotion names in movie order
    if not all(labels[sid] is not None for sid in ['1', '2', '3']):
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
                    # Prefer textual emotion token; ignore known headers.
                    emotion_name = ''
                    for tok in str_vals:
                        low = tok.lower()
                        if low in {'label', 'movie orders for three sessions'}:
                            continue
                        if re.fullmatch(r'\d+', tok):
                            continue
                        if 'session' in low:
                            continue
                        emotion_name = tok
                        break
                    if emotion_name:
                        emotion_to_id[emotion_name.lower()] = int_vals[0]

        if emotion_to_id:
            for sheet_name in xls.keys():
                mat = pd.read_excel(label_xlsx, sheet_name=sheet_name, header=None)
                for row in mat.itertuples(index=False):
                    vals = [v for v in row if not pd.isna(v)]
                    if not vals:
                        continue
                    row_text = ' '.join(str(v) for v in vals)
                    sid = _extract_session_id(row_text)
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
        raise ValueError(f'Failed to parse labels from {label_xlsx}')

    return labels


def parse_subject_session(file_name: str):
    stem = os.path.splitext(file_name)[0]
    parts = stem.split('_')
    subject = parts[0] if parts else stem
    session = parts[1] if len(parts) > 1 else ''
    if session not in {'1', '2', '3'}:
        sid = _extract_session_id(stem)
        session = sid if sid else session
    return subject, session


def main():
    ap = argparse.ArgumentParser(description='Preprocess SEED-V raw EEG (.cnt) into LMDB windows (62, 4, 200).')
    ap.add_argument('--raw_root', type=str, default='/data/datasets/BigDownstream/SEED-V/files')
    ap.add_argument('--timestamp_txt', type=str, default='')
    ap.add_argument('--label_xlsx', type=str, default='')
    ap.add_argument(
        '--output_lmdb',
        type=str,
        default='/data/datasets/BigDownstream/SEED-V/processed_raw_eeg_200hz_1to75_4x200_lmdb',
    )
    ap.add_argument('--map_size', type=int, default=64 * 1024 * 1024 * 1024)
    args = ap.parse_args()

    timestamp_txt = args.timestamp_txt or os.path.join(args.raw_root, 'trial_start_end_timestamp.txt')
    label_xlsx = args.label_xlsx or os.path.join(args.raw_root, 'emotion_label_and_stimuli_order.xlsx')

    print('[SEED-V preprocess] raw EEG pipeline (.cnt)')
    print(f'[SEED-V preprocess] raw_root={args.raw_root}')
    print(f'[SEED-V preprocess] timestamp_txt={timestamp_txt}')
    print(f'[SEED-V preprocess] label_xlsx={label_xlsx}')
    print(f'[SEED-V preprocess] target_fs={TARGET_FS}Hz band={LOW_FREQ}-{HIGH_FREQ}Hz patch_shape=(62,{N_PATCH},{PATCH_LEN})')

    trials = parse_trial_timestamps(timestamp_txt)
    labels = parse_labels_xlsx(label_xlsx)

    files = sorted([f for f in os.listdir(args.raw_root) if f.lower().endswith('.cnt')])
    if not files:
        raise FileNotFoundError(f'No .cnt files found under {args.raw_root}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_lmdb)), exist_ok=True)
    db = lmdb.open(args.output_lmdb, map_size=args.map_size)

    meta_jsonl = args.output_lmdb.rstrip('/\\') + '_sample_meta.jsonl'
    key_index = {'train': [], 'val': [], 'test': []}
    label_counter = Counter()
    seen_subjects = set()
    seen_sessions = set()
    seen_trials = set()
    n_segments = 0

    with open(meta_jsonl, 'w', encoding='utf-8') as mf:
        txn = db.begin(write=True)
        write_count = 0

        for file_name in files:
            subject_id, session_id = parse_subject_session(file_name)
            if session_id not in trials or session_id not in labels:
                print(f'[SEED-V preprocess][warn] skip {file_name}: unresolved session_id={session_id}')
                continue

            raw = mne.io.read_raw_cnt(os.path.join(args.raw_root, file_name), preload=True, verbose='ERROR')
            drop_ch = [ch for ch in USELESS_CH if ch in raw.ch_names]
            if drop_ch:
                raw.drop_channels(drop_ch)

            raw.resample(TARGET_FS)
            raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose='ERROR')
            data_matrix = raw.get_data(units='uV')

            if data_matrix.shape[0] != 62:
                raise ValueError(f'Expected 62 EEG channels after drop, got {data_matrix.shape[0]} for {file_name}')

            for trial_idx in range(N_TRIALS):
                start_s = int(trials[session_id]['start'][trial_idx])
                end_s = int(trials[session_id]['end'][trial_idx])
                start_i = max(0, start_s * TARGET_FS)
                end_i = min(data_matrix.shape[1], end_s * TARGET_FS)
                if end_i <= start_i:
                    continue

                trial_data = data_matrix[:, start_i:end_i]
                usable = (trial_data.shape[1] // SEGMENT_LEN) * SEGMENT_LEN
                if usable < SEGMENT_LEN:
                    continue

                segments = trial_data[:, :usable].reshape(62, -1, N_PATCH, PATCH_LEN).transpose(1, 0, 2, 3)
                split = 'train' if trial_idx < 5 else ('val' if trial_idx < 10 else 'test')
                label = int(labels[session_id][trial_idx])

                seen_subjects.add(subject_id)
                seen_sessions.add(f'{subject_id}_{session_id}')
                seen_trials.add(f'{subject_id}_{session_id}_{trial_idx}')

                for seg_idx, sample in enumerate(segments):
                    sample_key = f'{subject_id}_{session_id}_t{trial_idx:02d}_g{seg_idx:05d}'
                    rec = {
                        'sample': sample.astype(np.float32),
                        'label': label,
                        'subject': subject_id,
                        'session': session_id,
                        'trial': trial_idx,
                        'segment': seg_idx,
                        'source_file': file_name,
                    }
                    txn.put(sample_key.encode(), pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
                    key_index[split].append(sample_key)

                    meta = {
                        'key': sample_key,
                        'subject': subject_id,
                        'session': session_id,
                        'trial': trial_idx,
                        'segment': seg_idx,
                        'label': label,
                    }
                    mf.write(json.dumps(meta, ensure_ascii=True) + '\n')

                    label_counter[label] += 1
                    n_segments += 1
                    write_count += 1

                    if write_count % 2000 == 0:
                        txn.commit()
                        txn = db.begin(write=True)

        txn.put(b'__keys__', pickle.dumps(key_index, protocol=pickle.HIGHEST_PROTOCOL))
        txn.commit()

    db.close()

    print('[SEED-V preprocess] done')
    print(f'[SEED-V preprocess] output_lmdb={args.output_lmdb}')
    print(f'[SEED-V preprocess] metadata_jsonl={meta_jsonl}')
    print(f'[SEED-V preprocess][audit] subjects={len(seen_subjects)}')
    print(f'[SEED-V preprocess][audit] sessions={len(seen_sessions)}')
    print(f'[SEED-V preprocess][audit] trials={len(seen_trials)}')
    print(f'[SEED-V preprocess][audit] segments={n_segments}')
    print(f'[SEED-V preprocess][audit] sample_shape=(62, {N_PATCH}, {PATCH_LEN})')
    print(f'[SEED-V preprocess][audit] label_distribution={dict(sorted(label_counter.items()))}')


if __name__ == '__main__':
    main()
