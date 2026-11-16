"""Home/away antisymmetry (the arzharch / Aditya03-D property, enforced here as a test).

Swapping which team is "home" -- every track slot's team, and every Smart Tagging restart team --
must swap H and A in the output and change nothing else. A pipeline with a hidden home bias
(a tie-break that prefers H, a threshold applied to one side only) fails this.

    python -m tests.test_symmetry [config.json ...]
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.config import Config  # noqa: E402
from pipeline.data import load_metrica  # noqa: E402
from pipeline.stages import run  # noqa: E402

FLIP = np.array([1, 0, 2, 3], np.int8)


def swapped(match):
    out = type(match)(**{k: v for k, v in match.__dict__.items() if not k.startswith("_")})
    out.team = (1 - match.team).astype(np.int8)
    out.restarts = [(f, (1 - t) if t >= 0 else t) for f, t in match.restarts]
    return out


def check(cfg, match):
    a, _ = run(match, cfg)
    b, _ = run(swapped(match), cfg)
    bad = int(np.sum(FLIP[a] != b))
    return bad


if __name__ == "__main__":
    paths = sys.argv[1:] or [os.path.join(ROOT, "configs", "final.json")]
    match = load_metrica(1)
    failed = False
    for p in paths:
        base = Config.load(p)
        for name, cfg in (("as-is", base),
                          ("all stages on", base.replace(rules=True, hmm=True, w_form=1.0,
                                                         dead_formation=True, gap_interp=True,
                                                         gap_follow=True, abstain=True))):
            bad = check(cfg, match)
            failed |= bad > 0
            print("{:<28} {:<14} frames not flipped: {}".format(os.path.basename(p), name, bad))
    sys.exit(1 if failed else 0)
