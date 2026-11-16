# Possession pipeline

Seven switchable stages, from raw automatic tracking to one `H`/`A`/`D`/`X` label per frame.
Every parameter lives in `pipeline/config.py`; every switch can be turned off and the pipeline
still produces a valid file.

| stage | module function | idea taken from |
|---|---|---|
| 1 ball gap filling | `stages.fill_ball` | Kim et al. 2023 (lightweight): interpolate short gaps; over long gaps the ball stays with the player nearest it at each edge of the gap |
| 2 ball in play | `stages.dead_ball` | Vidal-Codina et al. 2022; Khaustov & Mozgovoy 2020: out of bounds, whole-pitch stillness, long ball gap with slow players (dead until the ball reappears), kick-off from formation, kick-off from the centre spot, Smart Tagging set-piece restarts (fixed prior) |
| 3 primitives | `stages.primitives` | Morra et al. 2020 (layered primitives); Vidal-Codina 2022 (touch = sharp velocity change) |
| 4 individual control | `stages.control` | Link & Hoernig 2017: within `ctrl_r`, clearly nearer than any opponent, ball not flying past, held `ctrl_min` frames |
| 5 team possession | `stages.team_rules` | Link & Hoernig 2017; Khaustov 2020: possession changes only on confirmed opponent control (long enough, or two opponent controls in a row = a completed pass); contested / loose balls stay with the last team |
| 6 decoding | `stages.decode` | Viterbi over H/A/D with switch costs, then min-duration merge |
| 7 abstention | `stages.abstain` | scorer-aware `X` through long blind gaps (switch; tuner decides) |

## Tuning: external data only

Per the 3 Nov 2026 rules clarification, every tuned parameter comes from external data:
**Metrica Sports open sample games 1-3** (github.com/metrica-sports/sample-data, credited here
as their README asks). Their event data is turned into H/A/D frame labels by
`data.labels_from_events`; their tracking is degraded by `noise.corrupt` to match the noise
statistics measured automatically from the competition tracking (ball-missing burst lengths and
rate, tracked players per frame, and a slowly varying ball position offset of ~0.9 m sized so
the ball-to-nearest-player distance distribution matches: Metrica's sample ball sits on the
players' feet, the competition ball does not) -- no labels of the competition match are
involved. Without that offset the tuner picked a 0.8 m control radius that fired on 35% of
external frames but only 20% of competition frames.

`tune.py` runs a random search maximising the competition's own spellF1
(`evaluation/possession_metrics.py`) on those games. The competition video and its ground truth
are never used for training, fitting or calibration. Smart Tagging parameters (`dead_smarttag`,
`st_pre`, `restart_team_prior`) cannot be tuned externally (no Smart Tagging there), so they are
fixed, documented priors and are excluded from the search space.

## Reproduce

```bash
pip install -r requirements.txt            # numpy, pandas
git clone https://github.com/metrica-sports/sample-data ../external_metrica   # tuning only
python -m pipeline.tune --trials 300 --workers 6 --out configs/tuned_r1.json
python -m pipeline.run --config configs/final.json --out submission/La_Culer.csv
python evaluation/validate_submission.py submission/La_Culer.csv
```

`configs/final.json` is committed, so `pipeline.run` alone regenerates the submission.

## Visualiser (local only)

```bash
python viz/make_viz.py --match metrica1 --config configs/final.json   # external game
python viz/make_viz.py --match bar --start 11185 --config configs/final.json
```

Open the HTML in a browser. It embeds tracking data: keep it local, never publish it.
