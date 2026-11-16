"""Every switch and every parameter of the pipeline, in one place.

Each stage reads only its own block. A switch set to False removes the stage; the pipeline still
produces a valid labelling (see pipeline/stages.py for what replaces each stage when it is off).

Parameters are tuned on EXTERNAL data only (Metrica open sample games, noise-matched), never on
the competition match, its video or its ground truth -- see pipeline/tune.py and the 3 Nov 2026
rules clarification.
"""

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # ---- Stage 1: ball gap filling (Kim et al. 2023, lightweight version) -------------------
    gap_interp: bool = True          # linear interpolation over short gaps
    interp_max: int = 25             # frames; longer gaps are not interpolated
    gap_follow: bool = True          # long gaps: ball assumed to stay with the nearest player
    follow_r: float = 2.0            # m; anchor player must be this close to the last ball seen
    follow_max: int = 75             # frames followed from each side of the gap

    # ---- Stage 2: ball in play (Vidal-Codina 2022, Khaustov & Mozgovoy 2020) ------------------
    dead_oob: bool = True            # ball outside the pitch lines -> dead
    oob_margin: float = 0.5          # m outside the line before it counts as out
    oob_missing_cap: int = 375       # frames of missing ball after an exit still counted dead
    dead_slow: bool = True           # players and ball nearly still -> stoppage
    slow_window: int = 50            # frames, centred median window
    slow_player_v: float = 1.2       # m/s median player speed below which play is stopped
    slow_ball_v: float = 1.0         # m/s
    slow_min_len: int = 50           # frames; shorter "slow" runs are ignored
    dead_prekick: bool = True        # dead from period start until the kick-off is struck
    kick_r: float = 3.0              # m from the centre spot
    kick_v: float = 2.0              # m/s ball speed that marks the kick
    kick_search: int = 4500          # frames searched for the kick-off
    dead_formation: bool = False     # both teams set up in their own halves -> kick-off pending
                                     # (off by default so older configs reproduce unchanged)
    form_frac: float = 0.9           # share of each team's visible players in its own half
    form_min: int = 25               # frames the set-up must last
    form_ext: int = 250              # frames to extend until the ball leaves the centre
    dead_gap: bool = False           # long unseen-ball gap while players are slow -> dead from the
                                     # gap's start until the ball reappears (= the restart)
    gap_dead_min: int = 125          # frames
    gap_dead_pv: float = 1.5         # m/s mean player pace over the gap
    dead_smarttag: bool = True       # Smart Tagging SET PIECES: dead stretch before a restart
    st_pre: int = 75                 # frames before the tagged restart (fixed prior, not tuned)

    # ---- Stage 3/4: touch primitives + individual control (Morra 2020; Link & Hoernig 2017) --
    ctrl_r: float = 1.5              # m; player within this of the ball can control it
    ctrl_margin: float = 0.5         # m; nearest opponent must be this much further
    ctrl_ball_v: float = 12.0        # m/s; faster balls are passing by, not controlled
    ctrl_min: int = 4                # frames of control to count as an individual possession
    ctrl_gap_tol: int = 3            # frames of interruption tolerated inside one control run
    use_touch: bool = True           # a sharp ball speed/direction change near a player = touch
    touch_dv: float = 4.0            # m/s change in ball velocity over 2 frames

    # ---- Stage 5: team possession rules (Link & Hoernig 2017; Khaustov 2020) ------------------
    rules: bool = True
    change_min: int = 8              # frames of opponent control to flip possession
    pass_chain: bool = True          # two opponent controls within chain_window also flip it
    chain_window: int = 75
    restart_team_prior: bool = True  # Smart Tagging restart team starts the possession after D

    # ---- Stage 6: temporal decoding ------------------------------------------------------------
    hmm: bool = True
    w_rule: float = 2.0              # score for agreeing with the rule-machine label
    w_prox: float = 1.0              # score from soft proximity (nearest team)
    prox_scale: float = 2.0          # m; softness of the proximity sigmoid
    prox_r: float = 5.0              # m; proximity evidence only when someone is this close
    w_dead: float = 3.0              # score for D on detected dead frames (and against D elsewhere)
    w_form: float = 0.0              # ball unseen: the team whose block is nearer the last-seen
                                     # ball is defending -> evidence for the OTHER team (Nexus)
    form_scale: float = 5.0          # m; softness of that evidence
    switch_ha: float = 40.0          # cost of an H<->A switch
    switch_d: float = 20.0           # cost of entering or leaving D
    min_merge: bool = True
    min_poss: int = 50               # frames; shorter possession runs are absorbed
    min_dead: int = 25               # frames; shorter dead runs are absorbed

    # ---- Stage 7: abstention ---------------------------------------------------------------------
    abstain: bool = False            # X through long un-filled ball gaps inside a possession
    abstain_min: int = 125

    # -----------------------------------------------------------------------------------------
    def save(self, path):
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            raw = json.load(handle)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def replace(self, **kw):
        data = asdict(self)
        data.update(kw)
        return Config(**data)


# Search space for pipeline/tune.py: name -> ("bool",) | ("int", lo, hi) | ("float", lo, hi).
SPACE = {
    "gap_interp": ("bool",), "interp_max": ("int", 5, 50),
    "gap_follow": ("bool",), "follow_r": ("float", 1.0, 3.5), "follow_max": ("int", 25, 200),
    "dead_oob": ("bool",), "oob_margin": ("float", 0.0, 2.0), "oob_missing_cap": ("int", 50, 750),
    "dead_slow": ("bool",), "slow_window": ("int", 25, 125), "slow_player_v": ("float", 0.6, 2.0),
    "slow_ball_v": ("float", 0.3, 3.0), "slow_min_len": ("int", 25, 150),
    "dead_prekick": ("bool",),
    "dead_formation": ("bool",), "form_frac": ("float", 0.75, 1.0), "form_min": ("int", 10, 125),
    "form_ext": ("int", 25, 500),
    "dead_gap": ("bool",), "gap_dead_min": ("int", 25, 500), "gap_dead_pv": ("float", 0.8, 2.5),
    "w_form": ("float", 0.0, 3.0), "form_scale": ("float", 1.0, 15.0),
    "ctrl_r": ("float", 0.8, 3.0), "ctrl_margin": ("float", 0.0, 1.5),
    "ctrl_ball_v": ("float", 5.0, 20.0), "ctrl_min": ("int", 1, 12), "ctrl_gap_tol": ("int", 0, 8),
    "use_touch": ("bool",), "touch_dv": ("float", 2.0, 8.0),
    "rules": ("bool",), "change_min": ("int", 2, 25), "pass_chain": ("bool",),
    "chain_window": ("int", 25, 150),
    "hmm": ("bool",), "w_rule": ("float", 0.0, 5.0), "w_prox": ("float", 0.0, 3.0),
    "prox_scale": ("float", 0.5, 5.0), "prox_r": ("float", 2.0, 10.0),
    "w_dead": ("float", 0.5, 6.0), "switch_ha": ("float", 5.0, 150.0),
    "switch_d": ("float", 5.0, 100.0),
    "min_merge": ("bool",), "min_poss": ("int", 0, 125), "min_dead": ("int", 0, 75),
    "abstain": ("bool",), "abstain_min": ("int", 50, 375),
}
