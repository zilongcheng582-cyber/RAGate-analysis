"""Run all three deterministic raw-to-processed conversion stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from prepare_dstc11 import prepare as prepare_dstc11
from prepare_dstc9 import prepare as prepare_dstc9
from prepare_ketod import prepare as prepare_ketod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ketod = prepare_ketod(args.raw_root, args.output_root)
    dstc9 = prepare_dstc9(args.raw_root, args.output_root)
    dstc11 = prepare_dstc11(args.raw_root, args.output_root)
    print({"KETOD": ketod, "DSTC9": dstc9, "DSTC11": dstc11})


if __name__ == "__main__":
    main()
