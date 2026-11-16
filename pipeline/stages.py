"""The pipeline: seven switchable stages from raw tracking to one H/A/D/X label per frame.

    1  fill_ball        ball gap filling                   Kim et al. 2023 (lightweight)
    2  dead_ball        ball-in-play detection             Vidal-Codina 2022; Khaustov 2020
    3  primitives       distances, ball speed, touches     Morra 2020; Vidal-Codina 2022
    4  control          individual ball control runs       Link & Hoernig 2017
    5  team_rules       team possession state machine      Link & Hoernig 2017; Khaustov 2020
    6  decode           Viterbi over H/A/D + min-duration  (temporal smoothing)
    7  abstain          X through long blind gaps          (scorer-aware abstention)

Every stage is switched by its Config flag; when a stage is off the next one still gets a valid
input (documented at each stage). run() returns the labels and every intermediate array, which
the visualiser shows.
"""

import numpy as np
import pandas as pd

H, A, D, X = 0, 1, 2, 3
SRC_OBS, SRC_INTERP, SRC_FOLLOW, SRC_NONE = 0, 1, 2, 3


def _true_runs(mask, offset=0):
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip((np.flatnonzero(d == 1) + offset).tolist(),
                    (np.flatnonzero(d == -1) + offset).tolist()))


def _close(mask, gap):
    """Fill False gaps of length <= gap that lie between True runs."""
    if gap <= 0:
        return mask
    out = mask.copy()
    for s, e in _true_runs(~mask):
        if s > 0 and e < len(mask) and e - s <= gap:
            out[s:e] = True
    return out


def _speed(xy, fps):
    v = np.full(len(xy), np.nan, np.float32)
    v[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) * fps / 2.0
    return v


def _player_speed_median(match):
    cache = match.__dict__.setdefault("_cache", {})
    if "pspeed" not in cache:
        P = match.players
        sp = np.full(P.shape[:2], np.nan, np.float32)
        sp[1:-1] = np.linalg.norm(P[2:] - P[:-2], axis=2) * match.fps / 2.0
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache["pspeed"] = np.nanmedian(sp, axis=1)
    return cache["pspeed"]


# -- stage 1 ---------------------------------------------------------------------------------

def fill_ball(match, cfg):
    """Ball position on every frame we can justify, and where each came from.

    Off: the observed ball only (gaps stay NaN, src = SRC_NONE).
    """
    ball = match.ball.copy()
    src = np.where(np.isnan(ball[:, 0]), SRC_NONE, SRC_OBS).astype(np.int8)
    P = match.players
    for a, b in match.runs:
        for s, e in _true_runs(np.isnan(match.ball[a:b, 0]), a):
            left, right = s > a, e < b
            if cfg.gap_interp and left and right and e - s <= cfg.interp_max:
                w = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
                ball[s:e] = (1 - w)[:, None] * ball[s - 1] + w[:, None] * ball[e]
                src[s:e] = SRC_INTERP
                continue
            if not cfg.gap_follow:
                continue
            mid = (s + e) // 2 if (left and right) else (e if left else s)
            if left:
                d = np.linalg.norm(P[s - 1] - ball[s - 1], axis=1)
                if np.isfinite(d).any() and np.nanmin(d) < cfg.follow_r:
                    k = int(np.nanargmin(d))
                    stop = min(mid, s + cfg.follow_max)
                    seg = P[s:stop, k]
                    ok = ~np.isnan(seg[:, 0])
                    ball[s:stop][ok] = seg[ok]
                    src[s:stop][ok] = SRC_FOLLOW
            if right:
                d = np.linalg.norm(P[e] - ball[e], axis=1)
                if np.isfinite(d).any() and np.nanmin(d) < cfg.follow_r:
                    k = int(np.nanargmin(d))
                    start = max(mid, e - cfg.follow_max)
                    seg = P[start:e, k]
                    ok = ~np.isnan(seg[:, 0])
                    ball[start:e][ok] = seg[ok]
                    src[start:e][ok] = SRC_FOLLOW
    return ball, src


# -- stage 2 ---------------------------------------------------------------------------------

def dead_ball(match, cfg, ball, src, bspeed):
    """Boolean dead-ball mask. All sub-detectors off -> nothing is dead."""
    n = match.n
    dead = np.zeros(n, bool)
    L, W = match.pitch
    seen = (src == SRC_OBS) | (src == SRC_INTERP)
    for a, b in match.runs:
        if cfg.dead_oob:
            x, y = ball[a:b, 0], ball[a:b, 1]
            m = cfg.oob_margin
            with np.errstate(invalid="ignore"):
                oob = seen[a:b] & ((x < -m) | (x > L + m) | (y < -m) | (y > W + m))
            for s, e in _true_runs(oob, a):
                dead[s:e] = True
                j = e
                while j < b and j - e < cfg.oob_missing_cap and src[j] in (SRC_FOLLOW, SRC_NONE):
                    dead[j] = True
                    j += 1
        if cfg.dead_slow:
            ps = pd.Series(_player_speed_median(match)[a:b]).rolling(
                cfg.slow_window, center=True, min_periods=1).median().values
            bs = pd.Series(np.where(seen[a:b], bspeed[a:b], np.nan)).rolling(
                cfg.slow_window, center=True, min_periods=1).median().values
            with np.errstate(invalid="ignore"):
                slow = (ps < cfg.slow_player_v) & ((bs < cfg.slow_ball_v) | np.isnan(bs))
            for s, e in _true_runs(slow, a):
                if e - s >= cfg.slow_min_len:
                    dead[s:e] = True
        if cfg.dead_prekick:
            centre = np.array([L / 2.0, W / 2.0])
            stop = min(b, a + cfg.kick_search)
            near = np.linalg.norm(ball[a:stop] - centre, axis=1) < cfg.kick_r
            with np.errstate(invalid="ignore"):
                kick = np.flatnonzero(near & seen[a:stop] & (bspeed[a:stop] > cfg.kick_v))
            if len(kick):
                dead[a:a + kick[0]] = True
        if cfg.dead_smarttag:
            for f, _ in match.restarts:
                if a <= f < b:
                    dead[max(a, f - cfg.st_pre):f] = True
    return dead


# -- stages 3 and 4 --------------------------------------------------------------------------

def primitives(match, cfg, ball, src):
    """Nearest home / away player distance to the (filled) ball, ball speed, touch flags."""
    usable = src != SRC_NONE
    d = np.linalg.norm(match.players - ball[:, None, :], axis=2)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dH = np.nanmin(np.where(match.team[None, :] == 0, d, np.nan), axis=1)
            dA = np.nanmin(np.where(match.team[None, :] == 1, d, np.nan), axis=1)
    dH[~usable] = np.nan
    dA[~usable] = np.nan
    bspeed = _speed(ball, match.fps)
    touch = np.zeros(match.n, bool)
    if cfg.use_touch:
        vel = np.full_like(ball, np.nan)
        vel[1:] = (ball[1:] - ball[:-1]) * match.fps
        dv = np.full(match.n, np.nan, np.float32)
        dv[2:-2] = np.linalg.norm(vel[4:] - vel[:-4], axis=1)
        with np.errstate(invalid="ignore"):
            touch = (dv > cfg.touch_dv) & (src == SRC_OBS)
    return dH, dA, bspeed, touch


def control(match, cfg, dH, dA, bspeed, touch):
    """Per-frame control candidate (-1 none, -2 contested, 0/1 team) and control runs."""
    inf = np.float32(1e9)
    h = np.nan_to_num(dH, nan=inf)
    a = np.nan_to_num(dA, nan=inf)
    with np.errstate(invalid="ignore"):
        slow_enough = (np.nan_to_num(bspeed, nan=0.0) < cfg.ctrl_ball_v) | touch
    cand = np.full(match.n, -1, np.int8)
    cand[(h < cfg.ctrl_r) & (h + cfg.ctrl_margin < a) & slow_enough] = H
    cand[(a < cfg.ctrl_r) & (a + cfg.ctrl_margin < h) & slow_enough] = A
    cand[(h < cfg.ctrl_r) & (a < cfg.ctrl_r) & (np.abs(h - a) <= cfg.ctrl_margin)] = -2
    events = []
    for team in (H, A):
        for s, e in _true_runs(_close(cand == team, cfg.ctrl_gap_tol)):
            if e - s >= cfg.ctrl_min:
                events.append((s, e, team))
    events.sort()
    return cand, events


# -- stage 5 ---------------------------------------------------------------------------------

def team_rules(match, cfg, dead, events):
    """Team possession: changes only on confirmed opponent control; loose/contested balls stay.

    Off: returns all -1 (unknown) on alive frames, D on dead ones.
    """
    lab = np.full(match.n, -1, np.int8)
    lab[dead] = D
    if not cfg.rules:
        return lab
    starts = np.array([e[0] for e in events]) if events else np.zeros(0, int)
    restart_at = {f: t for f, t in match.restarts if t >= 0}
    restart_frames = np.array(sorted(restart_at)) if restart_at else np.zeros(0, int)
    for a, b in match.runs:
        for p, q in _true_runs(~dead[a:b], a):
            i0, i1 = np.searchsorted(starts, p), np.searchsorted(starts, q)
            span = events[i0:i1]
            cur = None
            if cfg.restart_team_prior and len(restart_frames):
                j = np.searchsorted(restart_frames, p - cfg.st_pre - 25)
                if j < len(restart_frames) and restart_frames[j] <= p + 50:
                    cur = restart_at[int(restart_frames[j])]
            pos = p
            for idx, (s, e, c) in enumerate(span):
                if cur is None:
                    cur, pos = c, p          # first control after a restart owns from the restart
                    continue
                if c == cur:
                    continue
                confirmed = (e - s) >= cfg.change_min
                if not confirmed and cfg.pass_chain:
                    for s2, e2, c2 in span[idx + 1:]:
                        if s2 - e > cfg.chain_window:
                            break
                        if c2 == cur:
                            break
                        if c2 == c:
                            confirmed = True
                            break
                if confirmed:
                    lab[pos:s] = cur
                    cur, pos = c, s
            if cur is not None:
                lab[pos:q] = cur
    return lab


# -- stage 6 ---------------------------------------------------------------------------------

def _viterbi(S, c_ha, c_d):
    n = len(S)
    cost = [[0.0, c_ha, c_d], [c_ha, 0.0, c_d], [c_d, c_d, 0.0]]
    back = np.zeros((n, 3), np.int8)
    v = [float(S[0, 0]), float(S[0, 1]), float(S[0, 2])]
    s0, s1, s2 = S[:, 0].tolist(), S[:, 1].tolist(), S[:, 2].tolist()
    for t in range(1, n):
        nv, row = [0.0, 0.0, 0.0], back[t]
        for j, sj in ((0, s0[t]), (1, s1[t]), (2, s2[t])):
            c = cost[j]
            b0, b1, b2 = v[0] - c[0], v[1] - c[1], v[2] - c[2]
            if b0 >= b1 and b0 >= b2:
                nv[j], row[j] = b0 + sj, 0
            elif b1 >= b2:
                nv[j], row[j] = b1 + sj, 1
            else:
                nv[j], row[j] = b2 + sj, 2
        v = nv
    path = np.zeros(n, np.int8)
    path[-1] = int(np.argmax(v))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _merge_short(lab, runs, code_set, min_len):
    if min_len <= 0:
        return lab
    lab = lab.copy()
    for a, b in runs:
        w = lab[a:b]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], br, [len(w)]))
        segs = [[bounds[i], bounds[i + 1], w[bounds[i]]] for i in range(len(bounds) - 1)]
        for i, (s, e, c) in enumerate(segs):
            if c not in code_set or e - s >= min_len:
                continue
            prev = segs[i - 1] if i > 0 else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            if prev and nxt:
                pick = prev if (prev[1] - prev[0]) >= (nxt[1] - nxt[0]) else nxt
                if prev[2] == nxt[2]:
                    pick = prev
            else:
                pick = prev or nxt
            if pick is not None:
                w[s:e] = pick[2]
                segs[i][2] = pick[2]
    return lab


def decode(match, cfg, rule_lab, dead, dH, dA, cand):
    """Viterbi over H/A/D on rule, proximity and dead-ball evidence, then min-duration merge.

    HMM off: the rule labels are used directly (unknowns filled forward, then with the nearest
    team). Rules and HMM both off: nearest team per frame, forward-filled, D where dead.
    """
    n = match.n
    lab = np.full(n, X, np.int8)
    if cfg.hmm:
        S = np.zeros((n, 3), np.float32)
        if cfg.rules:
            for k in (H, A, D):
                S[rule_lab == k, k] += cfg.w_rule
        if cfg.w_prox > 0:
            with np.errstate(invalid="ignore", over="ignore"):
                dmin = np.fmin(dH, dA)
                diff = np.nan_to_num(dA, nan=99.0) - np.nan_to_num(dH, nan=99.0)
                p = 1.0 / (1.0 + np.exp(-diff / cfg.prox_scale))
                near = dmin < cfg.prox_r
            S[near, H] += cfg.w_prox * (2 * p[near] - 1)
            S[near, A] += cfg.w_prox * (1 - 2 * p[near])
        S[:, D] += np.where(dead, cfg.w_dead, -cfg.w_dead)
        for a, b in match.runs:
            lab[a:b] = _viterbi(S[a:b], cfg.switch_ha, cfg.switch_d)
    else:
        base = rule_lab.copy()
        if not cfg.rules:
            with np.errstate(invalid="ignore"):
                base = np.where(dead, D, np.where(cand >= 0, cand, -1)).astype(np.int8)
        for a, b in match.runs:
            w = base[a:b].astype(np.int16)
            idx = np.where(w >= 0, np.arange(len(w)), -1)
            idx = np.maximum.accumulate(idx)
            first = np.flatnonzero(w >= 0)
            idx[idx < 0] = first[0] if len(first) else 0
            filled = w[idx] if len(first) else np.full(len(w), D)
            lab[a:b] = np.where(filled < 0, D, filled)
    if cfg.min_merge:
        lab = _merge_short(lab, match.runs, (H, A), cfg.min_poss)
        lab = _merge_short(lab, match.runs, (D,), cfg.min_dead)
    return lab


# -- stage 7 ---------------------------------------------------------------------------------

def abstain(match, cfg, lab, src):
    if not cfg.abstain:
        return lab
    lab = lab.copy()
    for a, b in match.runs:
        for s, e in _true_runs(src[a:b] == SRC_NONE, a):
            if e - s >= cfg.abstain_min and s > a and e < b and lab[s - 1] == lab[e] \
                    and lab[s - 1] in (H, A):
                lab[s:e] = X
    return lab


# -- the whole thing ---------------------------------------------------------------------------

def run(match, cfg):
    ball, src = fill_ball(match, cfg)
    dH, dA, bspeed, touch = primitives(match, cfg, ball, src)
    dead = dead_ball(match, cfg, ball, src, bspeed)
    cand, events = control(match, cfg, dH, dA, bspeed, touch)
    rule_lab = team_rules(match, cfg, dead, events)
    lab = decode(match, cfg, rule_lab, dead, dH, dA, cand)
    lab = abstain(match, cfg, lab, src)
    out = np.full(match.n, X, np.int8)
    for a, b in match.runs:
        out[a:b] = lab[a:b]
    debug = {"ball": ball, "src": src, "dead": dead, "cand": cand, "rule": rule_lab,
             "dH": dH, "dA": dA, "bspeed": bspeed}
    return out, debug
