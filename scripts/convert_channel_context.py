#!/usr/bin/env python3
"""Convert channel context metadata (including FACED Electrode_Location.xlsx) to JSON."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.adapters import load_channel_context_file  # noqa: E402


def _tensor_to_jsonable(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Convert channel context file (.xlsx/.json/.pt/.pth) to normalized JSON."
    )
    parser.add_argument("--input", required=True, help="Input channel context file path.")
    parser.add_argument("--output", required=True, help="Output .json path.")
    parser.add_argument(
        "--expected_channels",
        type=int,
        default=None,
        help="Optional expected channel count (e.g., 32 for FACED).",
    )
    parser.add_argument(
        "--align_mode",
        type=str,
        default="auto",
        choices=["auto", "strict", "off"],
        help="Channel ID alignment mode used while loading input context.",
    )
    args = parser.parse_args()

    expected_ids = range(args.expected_channels) if args.expected_channels is not None else None
    context = load_channel_context_file(
        path=args.input,
        expected_channels=args.expected_channels,
        expected_channel_ids=expected_ids,
        align_mode=args.align_mode,
    )
    payload = {k: _tensor_to_jsonable(v) for k, v in context.items()}
    if not payload:
        raise ValueError("No usable channel-context fields were found in input file.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote channel context JSON: {output_path}")


if __name__ == "__main__":
    main()
