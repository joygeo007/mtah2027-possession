"""Local tracking visualiser: players, ball, and label strips (prediction vs reference).

    python viz/make_viz.py --match bar --start 11185 --frames 7500 --config configs/final.json
    python viz/make_viz.py --match metrica1 --start 0 --frames 15000 --config configs/final.json
    python viz/make_viz.py --match bar --start 11185 --pred submission/team.csv

Writes a self-contained HTML file under viz/ (gitignored). It embeds tracking data, so it must
stay on your machine: do not upload, publish or share it -- the match data is not redistributable.

Rules note (3 Nov 2026 clarification): inspecting the competition data to understand possession
is allowed; tuning by eye on the competition match is not. Use the external Metrica games
(--match metricaN), whose reference labels come from their own event data, for any comparison
that feeds back into the pipeline.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from pipeline.config import Config  # noqa: E402
from pipeline.data import load_competition, load_metrica  # noqa: E402
from pipeline.stages import run  # noqa: E402

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}


def _round(a, nd=2):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="bar", help="bar | metrica1 | metrica2 | metrica3")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--frames", type=int, default=7500)
    ap.add_argument("--step", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--config", default=None, help="run the pipeline with this config")
    ap.add_argument("--pred", default=None, help="or read predictions from a frame,label CSV")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.match == "bar":
        match = load_competition()
        gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        truth = pd.read_csv(gt_path)["label"].map(CODES).values.astype(np.int8)
        truth_name = "public GT (5 min only)"
    else:
        match = load_metrica(int(args.match[-1]))
        truth = match.labels
        truth_name = "event-derived labels"
    start = args.start if args.start is not None else match.runs[0][0]
    stop = min(match.n, start + args.frames)
    idx = np.arange(start, stop, args.step)

    pred = rule = dead = src = None
    if args.config:
        cfg = Config.load(args.config) if args.config != "default" else Config()
        labels, dbg = run(match, cfg)
        pred, rule, dead, src = labels, dbg["rule"], dbg["dead"], dbg["src"]
        ball = dbg["ball"]
    else:
        ball = match.ball
        src = np.where(np.isnan(ball[:, 0]), 3, 0)
    if args.pred:
        pred = pd.read_csv(args.pred)["label"].map(CODES).values.astype(np.int8)

    frames = []
    for t in idx:
        frames.append({
            "p": _round(match.players[t].reshape(-1), 1),
            "b": _round(ball[t], 2),
            "s": int(src[t]),
        })
    payload = {
        "name": match.name, "pitch": list(match.pitch), "fps": match.fps / args.step,
        "team": match.team.tolist(), "idx": idx.tolist(), "frames": frames,
        "truth": truth[idx].tolist(), "truth_name": truth_name,
        "pred": pred[idx].tolist() if pred is not None else None,
        "rule": [int(v) if v >= 0 else 3 for v in rule[idx]] if rule is not None else None,
        "dead": dead[idx].astype(int).tolist() if dead is not None else None,
    }
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as handle:
        html = handle.read().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    out = args.out or os.path.join(HERE, "{}_{}_{}.html".format(args.match, start, stop))
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(html)
    print("wrote", out, "({:.1f} MB)".format(os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
