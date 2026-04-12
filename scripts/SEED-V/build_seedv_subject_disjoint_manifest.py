#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys



def main():
    ap = argparse.ArgumentParser(
        description='Compatibility wrapper: build SEED-V manifest via canonical label-aware builder.'
    )
    ap.add_argument('--lmdb_path', type=str, required=True)
    ap.add_argument('--output_json', type=str, required=True)
    ap.add_argument('--meta_jsonl', type=str, default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train_subject_ratio', type=float, default=0.7)
    ap.add_argument('--val_subject_ratio', type=float, default=0.15)
    ap.add_argument('--balance_labels', action='store_true', default=True)
    ap.add_argument('--no-balance_labels', dest='balance_labels', action='store_false')
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    canonical_builder = os.path.join(repo_root, 'preprocessing', 'build_seedv_subject_manifest.py')

    cmd = [
        sys.executable,
        canonical_builder,
        '--lmdb_path', args.lmdb_path,
        '--output_json', args.output_json,
        '--seed', str(args.seed),
        '--train_ratio', str(args.train_subject_ratio),
        '--val_ratio', str(args.val_subject_ratio),
    ]
    if args.meta_jsonl:
        cmd.extend(['--meta_jsonl', args.meta_jsonl])
    if args.balance_labels:
        cmd.append('--balance_labels')
    else:
        cmd.append('--no-balance_labels')

    print('[manifest-wrapper] delegating to preprocessing/build_seedv_subject_manifest.py')
    print(f'[manifest-wrapper] balance_labels={args.balance_labels}')
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
