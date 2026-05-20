#!/usr/bin/env python3
"""
Merge multiple ChefNMR offline cache shard pickle files into one final cache.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ChefNMR offline cache shards")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input shard pickle files")
    parser.add_argument("--output", required=True, help="Output merged pickle file")
    parser.add_argument(
        "--allow-overwrite-keys",
        action="store_true",
        help="Allow later shards to overwrite duplicate keys",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged: dict = {}

    for input_path in args.inputs:
        path = Path(input_path)
        with open(path, "rb") as f:
            shard = pickle.load(f)
        if not isinstance(shard, dict):
            raise ValueError(f"Shard is not a dict: {path}")

        dup = set(merged).intersection(shard)
        if dup and not args.allow_overwrite_keys:
            preview = sorted(list(dup))[:10]
            raise ValueError(f"Duplicate keys found in {path}: {preview}")

        merged.update(shard)
        print(f"[merge] {path} records={len(shard)} total={len(merged)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(merged, f)
    tmp_path.replace(output_path)
    print(f"[done] wrote {len(merged)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
