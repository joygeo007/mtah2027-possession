"""Loaders: the competition match and the Metrica open sample games, into one common format.

Match fields (all arrays indexed by 0-based frame):
    ball     (T, 2) float32, metres, NaN where not observed
    players  (T, K, 2) float32, metres, NaN where not observed
    team     (K,) int8, 0 = home, 1 = away
    runs     [(start, stop)] half-open spans to label (graded halves / periods)
    labels   (T,) int8 codes H=0 A=1 D=2 X=3, or None (the competition match: no labels used)
    restarts [(frame, team or -1)] Smart Tagging set-piece starts (competition match only)

Parsed arrays are cached as .npz under ./cache (gitignored: it is derived tracking data and must
not be published).
"""

import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, half_video_blocks, require  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
EXTERNAL = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(ROOT), "external_metrica", "data")
H, A, D, X = 0, 1, 2, 3


@dataclass
class Match:
    name: str
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    runs: list
    pitch: tuple
    fps: int = 25
    labels: np.ndarray = None
    restarts: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.ball)


# -- shared parsing ------------------------------------------------------------------------------

def _parse_epts_raw(path, n_players, per_player, per_ball):
    """'frame:p;p;...:ball' lines -> (frame_ids, players (T,K,per_player), ball (T,per_ball))."""
    with open(path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"]).values
    frames = table[:, 0].astype(np.int64)
    width = n_players * per_player
    players = table[:, 1:1 + width].reshape(len(table), n_players, per_player)
    ball = table[:, 1 + width:1 + width + per_ball]
    return frames, players, ball


def _parse_epts(meta_path, raw_path):
    """EPTS pair -> (frame_ids, players (T,K,2), ball (T,2), team (K,)).

    Honours every <DataFormatSpecification> (one per frame range; substitutions change which
    players a line carries), mapping each to a global slot per player id.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")][:2]
    player_team = {p.get("id"): (0 if p.get("teamId") == team_ids[0] else 1)
                   for p in root.iter("Player")}
    chan_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order, per = [], {}
        for r in refs:
            pid = chan_player[r]
            if pid not in per:
                order.append(pid)
                per[pid] = 0
            per[pid] += 1
        n_ball = len(list(spec.iter("BallChannelRef")))
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order,
                      per[order[0]], n_ball))
    all_players = []
    for _, _, order, _, _ in specs:
        all_players += [p for p in order if p not in all_players]
    slot = {p: i for i, p in enumerate(all_players)}
    team = np.array([player_team[p] for p in all_players], np.int8)

    with open(raw_path) as handle:
        lines = handle.read().splitlines()
    ids = np.array([int(l[:l.index(":")]) for l in lines])
    frames_out, P_out, B_out = [], [], []
    for start, end, order, per, n_ball in specs:
        sel = np.flatnonzero((ids >= start) & (ids <= end))
        if not len(sel):
            continue
        text = "\n".join(lines[i] for i in sel).replace(":", ",").replace(";", ",")
        table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                            na_values=["NaN"]).values
        P = np.full((len(sel), len(all_players), 2), np.nan)
        cols = table[:, 1:1 + len(order) * per].reshape(len(sel), len(order), per)[:, :, :2]
        P[:, [slot[p] for p in order]] = cols
        frames_out.append(table[:, 0].astype(np.int64))
        P_out.append(P)
        B_out.append(table[:, 1 + len(order) * per:1 + len(order) * per + 2])
    return np.concatenate(frames_out), np.concatenate(P_out), np.concatenate(B_out), team


def _team_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")][:2]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    out = build()
    np.savez_compressed(path, **out)
    return out


# -- the competition match -----------------------------------------------------------------------

def smart_tag_restarts(path, fps=25):
    """[(video_frame, team code or -1)] for every SET PIECES instance."""
    root = ET.parse(path).getroot()
    out = []
    for inst in root.iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        if code != "SET PIECES":
            continue
        start = float(inst.findtext("start"))
        team = -1
        for label in inst.iter("label"):
            if (label.findtext("group") or "").strip().lower() == "team":
                value = (label.findtext("text") or "").strip().lower()
                team = 0 if value.startswith("barcelona") else 1 if value.startswith("atl") else -1
        out.append((int(round(start * fps)), team))
    return sorted(out)


def load_competition(game="BAR-ATH"):
    meta, raw = require(["atd_meta", "atd_raw"], game)
    team = _team_slots(meta)
    L, W = GAMES[game]["pitch"]
    n = GAMES[game]["n_video_frames"]

    def build():
        frames, players, ball = _parse_epts_raw(raw, len(team), 3, 3)
        P = np.full((n, len(team), 2), np.nan, np.float32)
        B = np.full((n, 2), np.nan, np.float32)
        P[frames - 1] = players[:, :, :2]
        B[frames - 1] = ball[:, :2]
        return {"players": P, "ball": B}

    z = _cached(game + "_tactical", build)
    scale = np.array([L, W], np.float32)
    runs = [(a, b + 1) for _, a, b in half_video_blocks(game)]
    restarts = []
    try:
        st = require(["smart_tagging"], game)[0]
        restarts = smart_tag_restarts(st)
    except Exception as exc:  # noqa: BLE001 -- Smart Tagging is optional input
        print("smart tagging unavailable:", exc)
    return Match(game, z["ball"] * scale, z["players"] * scale, team, runs, (L, W),
                 restarts=restarts)


# -- Metrica open sample games (external, for tuning only) ---------------------------------------

def _metrica_csv_tracking(game_dir, k):
    def side(name):
        path = os.path.join(game_dir, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(k, name))
        df = pd.read_csv(path, skiprows=2)
        cols = list(df.columns)
        xy = []
        for i, c in enumerate(cols):
            if c.startswith("Player"):
                xy.append(df.iloc[:, [i, i + 1]].values)
        ball_i = cols.index("Ball")
        return df["Frame"].values, np.stack(xy, axis=1), df.iloc[:, [ball_i, ball_i + 1]].values

    f_h, p_h, ball = side("Home")
    f_a, p_a, _ = side("Away")
    assert (f_h == f_a).all()
    players = np.concatenate([p_h, p_a], axis=1)
    team = np.array([0] * p_h.shape[1] + [1] * p_a.shape[1], np.int8)
    return f_h, players, ball, team


def _metrica_csv_events(game_dir, k):
    df = pd.read_csv(os.path.join(game_dir, "Sample_Game_{}_RawEventsData.csv".format(k)))
    out = []
    for r in df.itertuples(index=False):
        out.append({"team": 0 if r.Team == "Home" else 1, "type": str(r.Type).upper(),
                    "sub": str(r.Subtype).upper() if isinstance(r.Subtype, str) else "",
                    "period": int(r.Period), "start": int(r[4]), "end": int(r[6])})
    return out


def _metrica_json_events(path, home_id):
    data = json.load(open(path))["data"]
    out = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub.get("name", "")
        elif isinstance(sub, list):
            sub = "-".join(s.get("name", "") for s in sub)
        else:
            sub = ""
        out.append({"team": 0 if e["team"]["id"] == home_id else 1,
                    "type": e["type"]["name"].upper(), "sub": (sub or "").upper(),
                    "period": int(e["period"]), "start": int(e["start"]["frame"]),
                    "end": int(e["end"]["frame"] or e["start"]["frame"])})
    return out


CONTROL_EVENTS = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}


def labels_from_events(events, n):
    """H/A/D/X per frame from event data, by the rule agreed for tuning:

    - possession = team of the last controlled event (PASS, CARRY, SHOT, RECOVERY, SET PIECE)
    - BALL LOST / CHALLENGE / CARD never flip possession on their own
    - D from BALL OUT, FAULT RECEIVED or a goal, until the next SET PIECE
    - each period starts dead until its kick-off; X outside the periods
    Event frames are 1-based like the tracking, so index = frame - 1.
    """
    labels = np.full(n, X, np.int8)
    runs = []
    for period in sorted({e["period"] for e in events}):
        evs = sorted([e for e in events if e["period"] == period], key=lambda e: e["start"])
        first = max(evs[0]["start"] - 1 - 50, 0)        # 2 s of pre-kick dead ball
        last = min(max(e["end"] for e in evs) - 1, n - 1)
        runs.append((first, last + 1))
        state, pos = D, first
        for e in evs:
            s = e["start"] - 1
            new = None
            if state == D and e["type"] != "SET PIECE":
                continue                 # dead until the restart, whatever is logged meanwhile
            if e["type"] in CONTROL_EVENTS:
                new = e["team"]
            elif e["type"] in ("BALL OUT", "FAULT RECEIVED"):
                new = D
            if e["type"] == "SHOT" and "GOAL" in e["sub"]:
                s, new = max(e["end"] - 1, s), D
            if new is None or new == state:
                continue
            s = min(max(s, pos), last + 1)
            labels[pos:s] = state
            state, pos = new, s
        labels[pos:last + 1] = state
    return labels, runs


def load_metrica(k):
    game_dir = os.path.join(EXTERNAL, "Sample_Game_{}".format(k))
    L, W = 105.0, 68.0

    def build():
        if k in (1, 2):
            frames, players, ball, team = _metrica_csv_tracking(game_dir, k)
            events = _metrica_csv_events(game_dir, k)
        else:
            meta = os.path.join(game_dir, "Sample_Game_3_metadata.xml")
            frames, players, ball, team = _parse_epts(
                meta, os.path.join(game_dir, "Sample_Game_3_tracking.txt"))
            home_id = [t.get("id") for t in ET.parse(meta).getroot().iter("Team")][0]
            events = _metrica_json_events(
                os.path.join(game_dir, "Sample_Game_3_events.json"), home_id)
        n = int(frames.max())
        P = np.full((n, players.shape[1], 2), np.nan, np.float32)
        B = np.full((n, 2), np.nan, np.float32)
        P[frames - 1] = players
        B[frames - 1] = ball
        labels, runs = labels_from_events(events, n)
        return {"players": P, "ball": B, "team": team, "labels": labels,
                "runs": np.array(runs, np.int64)}

    z = _cached("metrica_{}".format(k), build)
    scale = np.array([L, W], np.float32)
    return Match("metrica{}".format(k), z["ball"] * scale, z["players"] * scale,
                 z["team"].astype(np.int8), [tuple(map(int, r)) for r in z["runs"]], (L, W),
                 labels=z["labels"])
