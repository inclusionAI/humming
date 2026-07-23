#!/usr/bin/env python3
"""Command-line wrapper for building Humming's native artifacts."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"

from humming.build import build_native  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build_native(args.output_dir)


if __name__ == "__main__":
    main()
