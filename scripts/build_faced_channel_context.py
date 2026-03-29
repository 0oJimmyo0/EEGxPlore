#!/usr/bin/env python3
"""Build richer FACED channel context JSON from Electrode_Location.xlsx.

Output keys:
  - channel_ids: [C]
  - channel_names: [C]
  - coords: [C,3] (approximate normalized 10-20 positions)
  - region_ids: [C]
  - montage_mask: [C] (1 if coordinate known, else 0)
  - unknown_channels: [name]
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

XLSX_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XLSX_NS = {"a": XLSX_NS_MAIN, "r": XLSX_NS_REL}


def xlsx_col_to_index(cell_ref: str) -> int:
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    if not letters:
        return -1
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def xlsx_cell_text(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", XLSX_NS))

    v = cell.find("a:v", XLSX_NS)
    if v is None:
        return ""
    text = v.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (ValueError, IndexError):
            return text
    return text


def xlsx_read_shared_strings(zf: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out: List[str] = []
    for si in root.findall("a:si", XLSX_NS):
        out.append("".join(t.text or "" for t in si.findall(".//a:t", XLSX_NS)))
    return out


def xlsx_parse_sheet_rows(zf: ZipFile, target: str, shared_strings: List[str]) -> List[List[str]]:
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    root = ET.fromstring(zf.read(target))
    rows: List[List[str]] = []
    for row in root.findall("a:sheetData/a:row", XLSX_NS):
        sparse: Dict[int, str] = {}
        for cell in row.findall("a:c", XLSX_NS):
            idx = xlsx_col_to_index(cell.attrib.get("r", ""))
            if idx < 0:
                continue
            sparse[idx] = xlsx_cell_text(cell, shared_strings)
        if not sparse:
            continue
        max_col = max(sparse.keys())
        rows.append([sparse.get(i, "") for i in range(max_col + 1)])
    return rows


def parse_positive_int(value: str):
    s = str(value).strip()
    if not s:
        return None
    try:
        n = float(s)
    except ValueError:
        return None
    if not n.is_integer():
        return None
    out = int(n)
    if out <= 0:
        return None
    return out


def parse_faced_xlsx(xlsx_path: Path) -> List[Tuple[int, str]]:
    with ZipFile(xlsx_path, "r") as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared_strings = xlsx_read_shared_strings(zf)

        id_to_name: Dict[int, str] = {}
        for sheet in wb.findall("a:sheets/a:sheet", XLSX_NS):
            rid = sheet.attrib.get(f"{{{XLSX_NS_REL}}}id")
            if not rid or rid not in rel_map:
                continue
            rows = xlsx_parse_sheet_rows(zf, rel_map[rid], shared_strings)
            for row in rows:
                for col in range(0, max(0, len(row) - 1), 2):
                    channel_id = parse_positive_int(row[col])
                    if channel_id is None:
                        continue
                    name = re.sub(r"\s+", " ", str(row[col + 1]).strip())
                    if not name or not re.search(r"[A-Za-z]", name):
                        continue
                    id_to_name[channel_id] = name

    if not id_to_name:
        raise ValueError("No [channel_id, channel_name] pairs found in xlsx.")

    return [(i, id_to_name[i]) for i in sorted(id_to_name.keys())]


# Approximate normalized cartesian coordinates for common 10-20 / 10-10 labels.
COORDS_3D: Dict[str, Tuple[float, float, float]] = {
    "FP1": (-0.30, 0.95, 0.00), "FP2": (0.30, 0.95, 0.00),
    "AF3": (-0.23, 0.88, 0.42), "AF4": (0.23, 0.88, 0.42),
    "F7": (-0.85, 0.55, 0.00), "F8": (0.85, 0.55, 0.00),
    "F5": (-0.60, 0.62, 0.35), "F6": (0.60, 0.62, 0.35),
    "F3": (-0.45, 0.68, 0.55), "F4": (0.45, 0.68, 0.55),
    "F1": (-0.20, 0.72, 0.66), "F2": (0.20, 0.72, 0.66),
    "FZ": (0.00, 0.74, 0.67),
    "FC5": (-0.65, 0.40, 0.45), "FC6": (0.65, 0.40, 0.45),
    "FC3": (-0.42, 0.44, 0.67), "FC4": (0.42, 0.44, 0.67),
    "FC1": (-0.20, 0.47, 0.77), "FC2": (0.20, 0.47, 0.77),
    "T7": (-0.98, 0.00, 0.00), "T8": (0.98, 0.00, 0.00),
    "C5": (-0.70, 0.00, 0.50), "C6": (0.70, 0.00, 0.50),
    "C3": (-0.50, 0.00, 0.78), "C4": (0.50, 0.00, 0.78),
    "C1": (-0.22, 0.00, 0.92), "C2": (0.22, 0.00, 0.92),
    "CZ": (0.00, 0.00, 1.00),
    "CP5": (-0.67, -0.35, 0.43), "CP6": (0.67, -0.35, 0.43),
    "CP3": (-0.45, -0.35, 0.63), "CP4": (0.45, -0.35, 0.63),
    "CP1": (-0.20, -0.35, 0.72), "CP2": (0.20, -0.35, 0.72),
    "P7": (-0.85, -0.55, 0.00), "P8": (0.85, -0.55, 0.00),
    "P5": (-0.62, -0.58, 0.30), "P6": (0.62, -0.58, 0.30),
    "P3": (-0.45, -0.63, 0.50), "P4": (0.45, -0.63, 0.50),
    "P1": (-0.20, -0.66, 0.60), "P2": (0.20, -0.66, 0.60),
    "PZ": (0.00, -0.68, 0.64),
    "PO7": (-0.65, -0.80, 0.00), "PO8": (0.65, -0.80, 0.00),
    "PO3": (-0.35, -0.82, 0.28), "PO4": (0.35, -0.82, 0.28),
    "O1": (-0.28, -0.95, 0.00), "O2": (0.28, -0.95, 0.00),
    "OZ": (0.00, -0.98, 0.00),
}


def normalize_name(name: str) -> str:
    n = name.strip().upper().replace(" ", "")
    # common aliases
    if n == "T3":
        return "T7"
    if n == "T4":
        return "T8"
    if n == "T5":
        return "P7"
    if n == "T6":
        return "P8"
    return n


def region_id(name: str) -> int:
    n = normalize_name(name)
    if n.endswith("Z"):
        return 9
    if n.startswith("AF") or n.startswith("FP") or n.startswith("F"):
        return 1
    if n.startswith("FC"):
        return 6
    if n.startswith("C"):
        return 2
    if n.startswith("CP"):
        return 7
    if n.startswith("P") and not n.startswith("PO"):
        return 3
    if n.startswith("PO"):
        return 8
    if n.startswith("O"):
        return 4
    if n.startswith("T"):
        return 5
    return 0


def build_context(channel_pairs: List[Tuple[int, str]], on_missing: str):
    channel_ids: List[int] = []
    channel_names: List[str] = []
    coords: List[List[float]] = []
    region_ids: List[int] = []
    montage_mask: List[int] = []
    unknown: List[str] = []

    for cid, raw_name in channel_pairs:
        name = raw_name.strip()
        key = normalize_name(name)

        channel_ids.append(int(cid))
        channel_names.append(name)
        region_ids.append(region_id(key))

        xyz = COORDS_3D.get(key)
        if xyz is None:
            coords.append([0.0, 0.0, 0.0])
            montage_mask.append(0)
            unknown.append(name)
        else:
            coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
            montage_mask.append(1)

    if unknown and on_missing == "error":
        raise ValueError("Missing coordinates for channels: " + ", ".join(sorted(set(unknown))))

    return {
        "channel_ids": channel_ids,
        "channel_names": channel_names,
        "coords": coords,
        "region_ids": region_ids,
        "montage_mask": montage_mask,
        "unknown_channels": sorted(set(unknown)),
    }


def main():
    parser = argparse.ArgumentParser(description="Build richer FACED channel context JSON")
    parser.add_argument("--xlsx", required=True, help="Path to Electrode_Location.xlsx")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument(
        "--on-missing",
        default="warn",
        choices=["warn", "error"],
        help="Behavior when channel names are not in the coordinate lookup",
    )
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = parse_faced_xlsx(xlsx_path)
    ctx = build_context(pairs, on_missing=args.on_missing)

    output_path.write_text(json.dumps(ctx, indent=2), encoding="utf-8")

    unknown_count = len(ctx["unknown_channels"])
    known_count = len(ctx["channel_ids"]) - unknown_count
    print(
        f"[build_faced_channel_context] wrote {output_path} channels={len(ctx['channel_ids'])} "
        f"known_coords={known_count} unknown_coords={unknown_count}"
    )
    if unknown_count and args.on_missing == "warn":
        print("[build_faced_channel_context][warning] Unknown channel names:", ", ".join(ctx["unknown_channels"]))


if __name__ == "__main__":
    main()
