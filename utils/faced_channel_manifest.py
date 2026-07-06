"""Canonical FACED channel manifest helpers.

The current FACED LMDB stores only tensors and labels. To make LaBraM consume
the correct spatial prior, we persist an explicit 32-slot channel manifest.

Source basis:
- TorchEEG FACED_CHANNEL_LIST publishes the 30 scalp EEG channels used for FACED.
- TorchEEG's FACED dataset description notes the remaining 2 channels are the
  left/right mastoids.

We therefore define the stored 32-channel order as the 30 published scalp
channels followed by the two mastoid channels, and normalize mastoid aliases to
LaBraM's standard_1020-compatible TP9/TP10 names.
"""

from __future__ import annotations

from typing import Dict, List

FACED_SCALP_CHANNELS_30: List[str] = [
    "FP1",
    "FP2",
    "FZ",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "CZ",
    "C3",
    "C4",
    "T7",
    "T8",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "PZ",
    "P3",
    "P4",
    "P7",
    "P8",
    "PO3",
    "PO4",
    "OZ",
    "O1",
    "O2",
]

# Stored order for the current FACED tensors: 30 scalp channels plus two
# mastoids. We keep the mastoid names explicit here and normalize to TP9/TP10
# for LaBraM consumption.
FACED_STORED_CHANNEL_NAMES_32: List[str] = FACED_SCALP_CHANNELS_30 + ["A1", "A2"]

FACED_LABRAM_CHANNEL_NAMES_32: List[str] = FACED_SCALP_CHANNELS_30 + ["TP9", "TP10"]

FACED_CHANNEL_ALIASES: Dict[str, str] = {
    "A1": "TP9",
    "A2": "TP10",
    "M1": "TP9",
    "M2": "TP10",
    "LEFT_MASTOID": "TP9",
    "RIGHT_MASTOID": "TP10",
}


def normalize_faced_channel_names(channel_names: List[str]) -> List[str]:
    out: List[str] = []
    for name in channel_names:
        key = str(name).strip().upper()
        if not key:
            continue
        out.append(FACED_CHANNEL_ALIASES.get(key, key))
    return out


def build_faced_channel_manifest() -> Dict[str, object]:
    return {
        "dataset": "FACED",
        "version": 1,
        "source": (
            "TorchEEG FACED_CHANNEL_LIST (30 scalp channels) plus the two FACED "
            "mastoid channels described in the dataset docs."
        ),
        "stored_channel_names": list(FACED_STORED_CHANNEL_NAMES_32),
        "labram_channel_names": list(FACED_LABRAM_CHANNEL_NAMES_32),
        "alias_map": dict(FACED_CHANNEL_ALIASES),
        "notes": [
            "The FACED LMDB preserves tensor slot order but did not originally persist channel names.",
            "LaBraM consumes standard_1020-style channel names via input_chans.",
            "Mastoid aliases A1/A2 are normalized to TP9/TP10 for LaBraM.",
        ],
    }
