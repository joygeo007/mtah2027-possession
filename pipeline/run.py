"""Regenerate the submission from the provided data with a frozen, externally tuned config.

    python -m pipeline.run --config configs/final.json --out submission/La_Culer.csv

Then:  python evaluation/validate_submission.py submission/La_Culer.csv
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.config import Config  # noqa: E402
from pipeline.data import load_competition  # noqa: E402
from pipeline.stages import run  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "final.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission", "La_Culer.csv"))
    ap.add_argument("--debug", default=None, help="also save intermediates (.npz) for the viz")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    match = load_competition()
    labels, debug = run(match, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(labels))
    if args.debug:
        np.savez_compressed(args.debug, labels=labels, **debug)
    graded = np.concatenate([labels[a:b] for a, b in match.runs])
    print("wrote {} rows -> {}".format(len(labels), args.out))
    print("graded frames: H {} A {} D {} X {}".format(*np.bincount(graded, minlength=4)))


if __name__ == "__main__":
    main()
