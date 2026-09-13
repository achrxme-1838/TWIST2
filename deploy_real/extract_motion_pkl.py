"""Slice one motion out of a multi-motion (joblib) pkl into a single-motion pickle.

MotionLib (pose/utils/motion_lib_pkl.py) loads one motion per file with plain
``pickle``. DEX_RL_LAB's integrated datasets (PHUMA_*.pkl, GMR_*.pkl) are joblib
dumps holding many motions, so use this to pull out the one you want to play:

    python extract_motion_pkl.py <dataset.pkl> --list
    python extract_motion_pkl.py <dataset.pkl> --name <motion_key> --out ../assets/phuma/<motion_key>.pkl

Requires joblib (only needed for this helper, not for the motion server).
"""
import argparse
import os
import pickle

import joblib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="multi-motion pkl (joblib) from DEX_RL_LAB integrated_motions")
    parser.add_argument("--name", help="motion key to extract")
    parser.add_argument("--out", help="output single-motion pkl (default: ./<name>.pkl)")
    parser.add_argument("--list", action="store_true", help="print the motion keys and exit")
    parser.add_argument("--grep", default=None, help="with --list: only show keys containing this substring")
    args = parser.parse_args()

    data = joblib.load(args.dataset)
    if args.list or args.name is None:
        keys = [k for k in data.keys() if args.grep is None or args.grep in k]
        print(f"{len(keys)} motions" + (f" matching '{args.grep}'" if args.grep else "") + f" in {args.dataset}:")
        for k in keys:
            n = len(data[k]["root_trans"] if "root_trans" in data[k] else data[k]["root_pos"])
            print(f"  {k}  ({n} frames)")
        return

    if args.name not in data:
        raise SystemExit(f"'{args.name}' not found. Use --list to see available keys.")

    out = args.out or f"{args.name}.pkl"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data[args.name], f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
