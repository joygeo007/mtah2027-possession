# More than a Hack 2027 — Phase 1 submission: frame-level ball possession

A seven-stage, switchable, rule-based pipeline that labels every frame of the BAR-ATH tactical
video `H` / `A` / `D` from the automatic tracking and Smart Tagging. All parameters were tuned on
**external data only**: Metrica Sports' open sample games, noise-matched to the competition
tracking.

- **Technical document:** [TECHNICAL.md](TECHNICAL.md)
- **Pipeline overview:** [pipeline/README.md](pipeline/README.md)
- **Submission file:** [submission/](submission/)
- **Frozen config:** [configs/final.json](configs/final.json)

## Reproduce the submission

```bash
pip install -r requirements.txt                 # Python 3.8+ (generated with 3.10.9)
# unpack the competition data bundle into ./data  (see DATA.md; not included here)
python -m pipeline.run --config configs/final.json --out submission/submission.csv
python evaluation/validate_submission.py submission/submission.csv
```

The run is deterministic: it regenerates the committed file byte for byte.

## Repository layout

```
pipeline/     our code: loaders, noise matching, the 7 stages, tuner, runner
configs/      final.json (frozen) + every intermediate tuning result
logs/         tuning logs
submission/   the submitted possession file
viz/          local-only tracking viewer (generated HTML embeds tracking data: never commit it)
evaluation/   the organisers' scorer, validator and public 5-minute ground truth (unchanged)
starter/      the organisers' starter kit (unchanged)
games.py      the organisers' per-match constants (unchanged)
```

The competition rules and data documentation from the organisers' repository are kept
unchanged: [README_challenge.md](README_challenge.md), CHALLENGE.md, DATA.md,
EVALUATION.md, SUBMISSION.md, RULES.md, FAQ.md. Their code is MIT-licensed (LICENSE). **The
match data is not included and must not be added.** It is licensed for the competition only.
