# Frame-level ball possession from automatic tracking — technical document

**Team La Culer** · More than a Hack 2027 · Phase 1 · FC Barcelona vs Athletic Club (BAR-ATH)

## 1. Summary

We label every frame `H` / `A` / `D` with a seven-stage, rule-based pipeline built on the
automatic tracking (ATD) and the Smart Tagging. Each stage implements an idea from the
literature on event and possession detection from tracking data. Each stage can be switched on
or off from a single config file. **Every tuned parameter was fitted on external data only.** We
used the three Metrica Sports open sample games, degraded so that their noise matches the
competition tracking. The competition's ground truth and video were never used for training,
fitting or calibration.

| config, scored on 3 unseen noise draws × 3 external games (final noise model, §5.2) | spellF1 | macroF1 |
|---|---|---|
| official baseline method (nearest player + ±2 s vote) | 0.321 | — |
| our pipeline, hand-set defaults | 0.489 | 0.740 |
| our pipeline, v1 (first submitted config, `configs/final_v1_submitted.json`) | 0.543 | 0.768 |
| **our pipeline, final v2 (`configs/final.json`)** | **0.553** | **0.771** |

Per game and draw, the final config scores 0.53–0.59. Against v1 on 15 unseen draws it gains
+0.012 spellF1 on average (standard error 0.004) and is better on 12 of 15.

On the competition match it predicts 266 possessions averaging 13.2 s, 45% dead-ball frames, and
a 65/35 split of live possession between Barcelona and Athletic.

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
| 2 | **Ball in play.** Six switchable detectors; a frame is dead if any active one fires: (a) **out of bounds**, held while the ball stays unseen after an exit; (b) **whole-pitch stillness**: median player speed < 1.83 m/s and median ball speed < 1.95 m/s over 25 frames, lasting ≥ 84 frames; (c) **ball-gap stoppage**: the ball is unseen for ≥ 65 frames while mean player pace stays < 1.91 m/s, so it is dead from the first unseen frame until the ball reappears, which marks the restart (§5.2); (d) **kick-off from formation**: both teams with ≥ 90% of their visible players in their own half, then dead until the ball leaves the centre; (e) **kick-off from the centre spot** (ball near the spot, then struck); (f) the 3 s before a Smart Tagging set piece. | Vidal-Codina et al. 2022; Khaustov & Mozgovoy 2020; ball-gap cue after other teams' observation (§7) | (b), (c), (f) **on**; (a), (d), (e) **off**: the tuner found (c) covers them |
| 3 | **Primitives.** Nearest home and away player distance to the ball, ball speed, and touch candidates (velocity change > 3.3 m/s over 2 frames). | Morra et al. 2020 (layered primitives → complex events); Vidal-Codina 2022 (touches) | on |
| 4 | **Individual control.** A team controls the ball when its nearest player is within 1.41 m, at least 1.35 m nearer than the nearest opponent, and the ball moves slower than 8.2 m/s (unless there is a touch). If both teams are close and level, the ball is *contested*. | Link & Hoernig 2017 | on |
| 5 | **Team possession rules** (§5.3) | Link & Hoernig 2017; Khaustov & Mozgovoy 2020 | on |
| 6 | **Temporal decoding.** Optional Viterbi pass over H/A/D with switch costs. Its evidence: the rule labels, proximity, dead-ball flags and *formation inversion* (when the ball is unseen, the team whose block is nearer the last-seen ball is usually defending, so it counts as evidence for the other team). An exact H/A tie goes to the last team seen in control, so the decoder is exactly antisymmetric (§6). Then an optional min-duration merge. | standard HMM smoothing | Viterbi **off**, merge **off** |
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

- extra ball drop-outs, **placed by game state**: the competition tracker loses the ball 83.5% of
  the time when players are nearly still (< 1 m/s, a label-free stoppage proxy), against about
  10% when they run (> 2 m/s). Drop-outs are topped up separately in five player-pace bins to
  those rates, with gap lengths drawn per bin. The degraded external games then lose the ball
  75–86% of the time when it is truly dead and 15–21% when it is alive, as the real tracker does.
  Metrica's own raw data shows the same pattern (74–85% against 11–19%).
- player drop-outs
- a smooth (1 s) random ball offset of σ = 0.9 m

**Why game-state gaps matter.** Random drop-outs make "ball missing" look meaningless during
tuning, so the tuner learns to ignore it. In the real tracking, a long gap with slow players is
almost always a stoppage, and the ball reappears at the restart. With realistic gaps the tuner
adopted the ball-gap stoppage detector (stage 2c), and it replaced the out-of-bounds and
kick-off-spot detectors.

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
  - the opponent controls the ball for ≥ 20 frames (0.8 s), or
  - the opponent makes two control runs within 126 frames with no control by the current team in between. This is the "two consecutive opponent touches" proxy, i.e. a completed pass by the new team.

  The change is dated to the start of the opponent's first control run, not to when it was confirmed.
- **What does not change possession:** a single short touch, a deflection, a contested ball (both teams close and level) and a loose ball (nobody in control). In all of these the last team keeps possession.
- **Restarts** always pass through `D`. Dead-ball ownership is never predicted.

## 6. Tuning procedure

- **Objective:** the competition's own `spellF1` (from `evaluation/possession_metrics.py`) on noise-matched external games, with macroF1 as the tie-break.
- **Search (v2, final):** 240 random configurations over the 46-dimensional space in `config.SPACE`, then 30 rounds of local perturbation search (14 candidates per round). Every candidate was scored on the mean over **3 independent noise draws per game** (9 degraded matches), so a single lucky draw cannot win.
- **Holdout:** the reported scores are on noise draws (seeds 5–9) that no search saw. v2 replaced v1 only after beating it paired, draw by draw: +0.012 mean, standard error 0.004, better on 12 of 15.
- **Home/away antisymmetry is tested.** `tests/test_symmetry.py` swaps every slot's team and every Smart Tagging restart team, and requires H and A to swap exactly in the output. It found 1,557 frames on one external game where the Viterbi decoder broke exact ties toward home. Ties now go to the last team seen in control, and every config passes with 0 frames unflipped. (v1 and v2 have Viterbi off, so their outputs were never affected.)
- **Smart Tagging and kick-off settings are fixed, not tuned:** `dead_smarttag`, `st_pre` = 3 s, `restart_team_prior`, and the kick-off detector. The external games have no Smart Tagging, so these settings cannot be tuned legally and are fixed by hand. They are excluded from the search space.
- **Decoder structure was compared fairly.** Before the offset fix, carry-forward of confirmed control beat rules+Viterbi after equal refinement (0.679 against 0.631 on holdout). After the fix, the tuner chose the rule machine with Viterbi off, in both v1 and v2.
- **Kick-off from formation** (stage 2d) finds all 5 kick-offs in the graded halves of the competition match, each within a second or two of the Smart Tagging kick-off time, with no false detections. This is an automatic, label-free check. It did not improve the external score, because those labels begin at the kick-off, so the tuner left it off. Formation inversion (stage 6) only acts inside the Viterbi decoder, which is off.
- **Ball-gap filling was rejected.** With a drifting ball offset, positions invented inside gaps created more false control than they recovered.

## 7. Limitations and what we would do next

- **Domain gap.** External labels come from event data, not from corrected tracking as the competition ground truth does. The two may draw the possession and dead-ball boundaries a few frames differently. External scores ran higher than real-match scores for the baseline (0.32 against 0.24), so expect our real score to be lower than 0.553 as well.
- **Noise model is approximate.**
  - One offset size cannot match both the 10th percentile and the median of the ball-to-player distance at once.
  - Ball loss now follows the game state, using player pace as a proxy, but not finer causes such as crowded boxes or aerial balls.
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
python -m pipeline.run --config configs/final.json --out submission/La_Culer.csv
python evaluation/validate_submission.py submission/La_Culer.csv
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
