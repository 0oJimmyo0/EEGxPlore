#!/usr/bin/env python3
"""CLI wrapper: run from CBraMod root: python scripts/analyze_facced_routing.py ..."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.faced_routing_analyze import main  # noqa: E402

if __name__ == "__main__":
    main()
