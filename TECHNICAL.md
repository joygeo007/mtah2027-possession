# Frame-level ball possession from automatic tracking — technical document

More than a Hack 2027 · Phase 1 · FC Barcelona vs Athletic Club (BAR-ATH)

## 1. Summary

We label every frame `H` / `A` / `D` with a seven-stage, rule-based pipeline built on the
automatic tracking (ATD) and the Smart Tagging. Each stage implements an idea from the
literature on event and possession detection from tracking data. Each stage can be switched on
or off from a single config file. **Every tuned parameter was fitted on external data only.** We
used the three Metrica Sports open sample games, degraded so that their noise matches the
competition tracking. The competition's ground truth and video were never used for training,
fitting or calibration.

| config, scored on unseen noise draws of the external games | spellF1 | macroF1 |
|---|---|---|
| official baseline method (nearest player + ±2 s vote) | ~0.31 | — |
| our pipeline, hand-set defaults | 0.457 | 0.720 |
| **our pipeline, final (`configs/final.json`)** | **0.516** | **0.740** |

Per game, the final config scores 0.49–0.54.

On the competition match it predicts 251 possessions averaging 13.0 s, 49% dead-ball frames, and
a 66/34 split of live possession between Barcelona and Athletic.

## 2. Assets: used and dropped

| asset | use |
|---|---|
| **Tactical ATD** | Primary input: ball and player positions with team membership. Also the source of the label-free noise statistics (§5.2). |
| **Smart Tagging** | `SET PIECES` intervals only. They mark restarts (dead stretch before them) and, when the `Team` tag is set, the team restarting. Phases (Build Up, Progression, …) were not used because they are phases, not possessions. |
| Tactical homography | Dropped. It is only needed to go back to pixels, and we do not use the video. |
| Tactical video | Dropped. Not downloaded; the 3 Nov rules clarification forbids checking or calibrating on it. |
| Broadcast video, ATD, homography | Dropped. Not synchronised to the tactical video; the alignment cost was not worth it for this deadline. |
| Public 5-minute ground truth | Used only to confirm file format and frame alignment. Not used to select, tune or compare configurations. |

## 3. Training / tuning data

| dataset | source | labels: who and how | amount |
|---|---|---|---|
| Metrica Sports sample games 1, 2, 3 | github.com/metrica-sports/sample-data (anonymised; the licence asks for attribution, given here) | Metrica's event data for those games: passes, recoveries, set pieces, ball out, etc. We did not produce or edit any of it. We convert it to frame labels automatically (§5.1). | 3 matches, ≈430k frames, 869 possessions in total (276, 276 and 317) |

No hand labels of any kind, on any game, were created by us. Nothing about the competition match
other than the provided automatic data was used.

## 4. Method

All stages are in `pipeline/stages.py`; switches and parameters are in `pipeline/config.py`.

| # | stage | idea from | final setting |
|---|---|---|---|
| 1 | **Ball gap filling.** Linear interpolation over short gaps. Over long gaps, the ball is assumed to stay with the player nearest it at each edge of the gap. | Kim et al. 2023 (lightweight version: no learned model, because their model needs player identities, which the ATD lacks) | **off**: the tuner rejected it (see §6) |
| 2 | **Ball in play.** A frame is dead if: (a) the ball is outside the lines by `oob_margin`, held while the ball stays unseen after an exit, up to `oob_missing_cap`; (b) the whole pitch is still, i.e. median player speed < 1.73 m/s and median ball speed < 2.84 m/s over 26 frames, lasting ≥ 73 frames; (c) the period has not kicked off yet (ball near the centre spot, then struck); (d) it falls in the 3 s before a Smart Tagging set piece. | Vidal-Codina et al. 2022; Khaustov & Mozgovoy 2020 | on |
| 3 | **Primitives.** Nearest home and away player distance to the ball, ball speed, and touch candidates (velocity change > 3.0 m/s over 2 frames). | Morra et al. 2020 (layered primitives → complex events); Vidal-Codina 2022 (touches) | on |
| 4 | **Individual control.** A team controls the ball when its nearest player is within 1.41 m, at least 1.21 m nearer than the nearest opponent, and the ball moves slower than 8.4 m/s (unless there is a touch). If both teams are close and level, the ball is *contested*. | Link & Hoernig 2017 | on |
| 5 | **Team possession rules** (§5.3) | Link & Hoernig 2017; Khaustov & Mozgovoy 2020 | on |
| 6 | **Temporal decoding.** Optional Viterbi pass over H/A/D on rule, proximity and dead-ball evidence with switch costs, then a min-duration merge: possessions < 42 frames and dead runs < 34 frames are absorbed into a neighbour. | standard HMM smoothing | Viterbi **off**, merge on |
| 7 | **Abstention.** `X` through long blind gaps inside a possession. | scorer-aware | **off** |

## 5. Possession rules and labels

### 5.1 External labels (for tuning)

`pipeline/data.py: labels_from_events` turns Metrica events into frame labels:

- **Possession** is the team of the last *controlled* event: PASS, CARRY, SHOT, RECOVERY or SET PIECE.
- **Events that never flip possession on their own:** BALL LOST, CHALLENGE, CARD. A contested ball stays with the last team, as the challenge definition says.
- **Dead ball** runs from BALL OUT, FAULT RECEIVED or a goal until the next SET PIECE. Events logged during the stoppage are ignored.
- **Each period** is dead until its kick-off.

The result is 30–40% dead frames and mean possessions of 12–14 s, in line with the public window
(14.2 s).

### 5.2 Noise matching (no labels involved)

Metrica's sample tracking is much cleaner than the competition ATD, so thresholds tuned on it as
is do not transfer. `pipeline/noise.py` measures label-free statistics of the competition ATD:

- the ball-missing rate and the lengths of its gaps
- tracked players per frame and player gap lengths
- the distance from the ball to the nearest player

It then degrades each external game to match:

- extra ball drop-outs in realistic bursts
- player drop-outs
- a smooth (1 s) random ball offset of σ = 0.9 m

**The offset mattered most.** Metrica's sample ball sits on the players' feet (median 0.28 m to
the nearest player), while the competition ball does not (1.91 m). Without the offset, the tuner
picked a 0.8 m control radius. That radius fired on 35% of external frames but only 20% of
competition frames, and the config's score fell from 0.68 to 0.42 once the offset was added.
After retuning with the offset, control fires on 17–21% of external frames and 19.7% of
competition frames.

### 5.3 Possession rules implemented (stage 5)

- **Live play** is split into alive spans by the dead-ball mask. Every span starts with a clear possessor:
  - the Smart Tagging restart team, when tagged within the window around the span start;
  - otherwise, the first team to gain control in the span, back-dated to the span start.
- **A change of possession** needs *controlled* opponent possession. Either:
  - the opponent controls the ball for ≥ 21 frames (0.84 s), or
  - the opponent makes two control runs within 115 frames with no control by the current team in between. This is the "two consecutive opponent touches" proxy, i.e. a completed pass by the new team.

  The change is dated to the start of the opponent's first control run, not to when it was confirmed.
- **What does not change possession:** a single short touch, a deflection, a contested ball (both teams close and level) and a loose ball (nobody in control). In all of these the last team keeps possession.
- **Restarts** always pass through `D`. Dead-ball ownership is never predicted.

## 6. Tuning procedure

- **Objective:** the competition's own `spellF1` (from `evaluation/possession_metrics.py`) on noise-matched external games, with macroF1 as the tie-break.
- **Search:** 240 random configurations over the 38-dimensional space in `config.SPACE`, then 30 rounds of local perturbation search (12 candidates per round). The local search scored every candidate on two independent noise draws per game, so it could not fit one draw.
- **Holdout:** the reported scores are on two further noise draws (seeds 5 and 6) that no search saw.
- **Smart Tagging and kick-off settings are fixed, not tuned:** `dead_smarttag`, `st_pre` = 3 s, `restart_team_prior`, and the kick-off detector. The external games have no Smart Tagging, so these settings cannot be tuned legally and are fixed by hand. They are excluded from the search space.
- **Decoder structure was compared fairly.** Before the offset fix, carry-forward of confirmed control beat rules+Viterbi after equal refinement (0.679 against 0.631 on holdout). After the fix, the tuner chose the rule machine with the min-duration merge, and Viterbi off.
- **Ball-gap filling was rejected.** With a drifting ball offset, positions invented inside gaps created more false control than they recovered.

## 7. Limitations and what we would do next

- **Domain gap.** External labels come from event data, not from corrected tracking as the competition ground truth does. The two may draw the possession and dead-ball boundaries a few frames differently. External scores ran higher than real-match scores for the baseline (0.31 against 0.24), so expect our real score to be lower than 0.516 as well.
- **Noise model is approximate.**
  - One offset size cannot match both the 10th percentile and the median of the ball-to-player distance at once.
  - Ball loss in our model is random. In reality it concentrates in crowded and aerial moments.
- **Only three tuning matches,** about 870 possessions. More external data would help, for example the DFL / Bassek et al. 2025 dataset or PFF FC's World Cup 2022 tracking.
- **Throw-ins** carry no team in Smart Tagging, and out-of-bounds detection depends on the ball being tracked as it leaves the pitch.
- **No learned components.** The obvious next steps:
  - a learned ball-trajectory imputer on external data (Kim et al. 2023, adapted to team-only identities);
  - a per-frame control classifier trained on external data, feeding the Viterbi decoder.
- **No ball height.** Aerial duels and long balls are judged on 2D distance only.

## 8. Reproduction

```bash
pip install -r requirements.txt                  # numpy, pandas (Python 3.8+)
# put the competition bundle in ./data (see DATA.md)
python -m pipeline.run --config configs/final.json --out submission/<team_name>.csv
python evaluation/validate_submission.py submission/<team_name>.csv
```

`configs/final.json` is committed, and the run is deterministic: two runs produce byte-identical
files. To repeat the tuning (needs the external games):

```bash
git clone https://github.com/metrica-sports/sample-data ../external_metrica
python -m pipeline.tune --base configs/final.json --trials 240 --workers 6 --seeds 0 --seed 7 --out configs/tuned_o1.json
python -m pipeline.tune --base configs/tuned_o1.json --refine 30 --workers 6 --seeds 0 1 --seed 8 --out configs/tuned_o2.json
python -m pipeline.tune --eval configs/tuned_o2.json --seeds 5 6          # holdout noise draws
```

Code map:
- `pipeline/data.py`: loaders and external labels
- `pipeline/noise.py`: noise matching
- `pipeline/stages.py`: the seven stages
- `pipeline/tune.py`: search and evaluation
- `pipeline/run.py`: submission
- `viz/make_viz.py`: local-only viewer; its output embeds tracking data and is gitignored

## References

- Link, D. & Hoernig, M. (2017). Individual ball possession in soccer. *PLOS ONE*, 12(7), e0179953. doi:10.1371/journal.pone.0179953
- Vidal-Codina, F., Evans, N., El Fakir, B. & Billingham, J. (2022). Automatic event detection in football using tracking data. *Sports Engineering*, 25(1), 18. doi:10.1007/s12283-022-00381-6
- Morra, L., Manigrasso, F., Canto, G., Gianfrate, C., Guarino, E. & Lamberti, F. (2020). Slicing and dicing soccer: automatic detection of complex events from spatio-temporal data. In *Image Analysis and Recognition (ICIAR 2020)*, Springer. doi:10.1007/978-3-030-50347-5_11
- Khaustov, V. & Mozgovoy, M. (2020). Recognizing events in spatiotemporal soccer data. *Applied Sciences*, 10(22), 8046. doi:10.3390/app10228046
- Kim, H., Choi, H.-J., Kim, C. J., Yoon, J. & Ko, S.-K. (2023). Ball trajectory inference from multi-agent sports contexts using set transformer and hierarchical Bi-LSTM. In *Proc. 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)*, 4296–4307. doi:10.1145/3580305.3599779
- Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195. doi:10.1038/s41597-025-04505-y
- Metrica Sports sample data: github.com/metrica-sports/sample-data.
