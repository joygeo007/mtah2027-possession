"""Tune the pipeline on EXTERNAL data only: Metrica open sample games 1-3, noise-matched.

    python -m pipeline.tune --trials 80 --workers 4         # random search -> configs/tuned.json
    python -m pipeline.tune --eval configs/tuned.json       # score one config, per game

Objective: mean spellF1 over the noisy external games (the competition's ranking metric, computed
by the competition's own evaluation/possession_metrics.py), macroF1 as tie-break. The competition
match, its video and its ground truth are never read here -- only the competition ATD, to measure
noise statistics (no labels involved).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

from pipeline.config import SPACE, Config  # noqa: E402
from pipeline.data import load_competition, load_metrica  # noqa: E402
from pipeline.noise import corrupt, measure  # noqa: E402
from pipeline.stages import run  # noqa: E402

GAMES = (1, 2, 3)
SEEDS = (0, 1, 2)   # every config is judged on the mean over 3 noise draws per game
_MATCHES = None


def external_matches(seeds=SEEDS):
    stats = measure(load_competition())
    out = []
    for k in GAMES:
        m = load_metrica(k)
        for s in seeds:
            out.append(corrupt(m, stats, seed=s))
    return out


def score(match, cfg):
    pred, _ = run(match, cfg)
    runs = pm.valid_runs(match.labels)
    filled = pm.fill_abstentions(pred, runs)
    sp = pm.spell_scores(match.labels, filled, runs)
    mf = pm.macro_f1(pm.scored_confusion(pm.confusion(match.labels, pred)))
    return sp["mean_f1"], mf, sp["n_pred"], sp["n_gt"]


def evaluate(cfg, matches=None):
    matches = matches or _MATCHES
    rows = [score(m, cfg) for m in matches]
    return float(np.mean([r[0] for r in rows])), float(np.mean([r[1] for r in rows])), rows


def sample(rng, base):
    kw = {}
    for name, spec in SPACE.items():
        if spec[0] == "bool":
            kw[name] = bool(rng.random() < 0.75) if getattr(base, name) else bool(rng.random() < 0.25)
        elif spec[0] == "int":
            kw[name] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            kw[name] = float(rng.uniform(spec[1], spec[2]))
    return base.replace(**kw)


def perturb(rng, cfg, p=0.25, scale=0.15):
    kw = {}
    for name, spec in SPACE.items():
        if rng.random() > p:
            continue
        cur = getattr(cfg, name)
        if spec[0] == "bool":
            if rng.random() < 0.4:
                kw[name] = not cur
        else:
            lo, hi = spec[1], spec[2]
            v = float(np.clip(cur + rng.normal(0, scale * (hi - lo)), lo, hi))
            kw[name] = int(round(v)) if spec[0] == "int" else v
    return cfg.replace(**kw)


_SEEDS = SEEDS


def _init(seeds=None):
    global _MATCHES
    _MATCHES = external_matches(seeds or _SEEDS)


def _trial(cfg_dict):
    cfg = Config(**cfg_dict)
    sf, mf, _ = evaluate(cfg)
    return sf, mf, cfg_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=None, help="config json to start from")
    ap.add_argument("--eval", default=None, help="only score this config json")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "tuned.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                    help="noise seeds per external game")
    ap.add_argument("--refine", type=int, default=0,
                    help="rounds of local search around --base (batch = workers*2 per round)")
    args = ap.parse_args()

    base = Config.load(args.base) if args.base else Config()
    seeds = tuple(args.seeds)
    if args.refine:
        rng = np.random.default_rng(args.seed)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(seeds,)) as pool:
            best_cfg = base
            best = list(pool.map(_trial, [asdict(base)]))[0][:2]
            print("start spellF1 {:.4f} macroF1 {:.4f}".format(*best), flush=True)
            for r in range(args.refine):
                batch = [asdict(perturb(rng, best_cfg)) for _ in range(args.workers * 2)]
                for sf, mf, cfg in pool.map(_trial, batch):
                    if (sf, mf) > best:
                        best, best_cfg = (sf, mf), Config(**cfg)
                print("round {:>2}  best spellF1 {:.4f} macroF1 {:.4f}".format(r + 1, *best),
                      flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        best_cfg.save(args.out)
        print("-> {}".format(args.out))
        return
    if args.eval:
        _init(seeds)
        cfg = Config.load(args.eval) if args.eval != "default" else Config()
        sf, mf, rows = evaluate(cfg)
        for m, r in zip(_MATCHES, rows):
            print("{:<18} spellF1 {:.4f}  macroF1 {:.4f}  poss {:>4} (gt {})".format(m.name, *r))
        print("MEAN               spellF1 {:.4f}  macroF1 {:.4f}".format(sf, mf))
        return

    rng = np.random.default_rng(args.seed)
    candidates = [asdict(base)] + [asdict(sample(rng, base)) for _ in range(args.trials)]
    # second half of the budget: local perturbations around the best so far is done in rounds
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for i, (sf, mf, cfg) in enumerate(pool.map(_trial, candidates)):
            results.append((sf, mf, cfg))
            best = max(results, key=lambda r: (r[0], r[1]))
            print("[{:>3}/{}] {:6.0f}s  spellF1 {:.4f} macroF1 {:.4f} | best {:.4f}".format(
                i + 1, len(candidates), time.time() - t0, sf, mf, best[0]), flush=True)
    results.sort(key=lambda r: (r[0], r[1]), reverse=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Config(**results[0][2]).save(args.out)
    with open(args.out.replace(".json", "_trials.json"), "w") as handle:
        json.dump([{"spellF1": r[0], "macroF1": r[1], "cfg": r[2]} for r in results], handle)
    print("best spellF1 {:.4f} macroF1 {:.4f} -> {}".format(results[0][0], results[0][1], args.out))


if __name__ == "__main__":
    main()
