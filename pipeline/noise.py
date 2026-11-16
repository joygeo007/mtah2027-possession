"""Make external tracking as noisy as the competition tracking, so tuned parameters transfer.

Noise statistics are measured automatically from the competition ATD alone (ball-missing run
lengths and rate, tracked players per frame, player gap lengths). No labels are involved, which
the 3 Nov 2026 clarification allows ("running automatic checks using only the provided data").
"""

import numpy as np


def _runs(mask):
    """Lengths of True runs in a boolean array."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


# Game-state ball gaps (OsamaQ05 / Aaryan): the tracker loses the ball mostly during stoppages.
# "Stoppage" is proxied label-free by how fast the players move (10 s mean of the median player
# speed). Bin edges in m/s.
SPEED_BINS = (0.0, 1.0, 1.5, 2.0, 2.5, 99.0)
STATE_GAPS = True


def player_pace(match):
    """10 s rolling mean of the per-frame median player speed (m/s), label-free."""
    import pandas as pd
    from pipeline.stages import _player_speed_median
    return pd.Series(_player_speed_median(match)).rolling(
        250, center=True, min_periods=1).mean().values


def _bin_of(pace):
    return np.clip(np.digitize(pace, SPEED_BINS) - 1, 0, len(SPEED_BINS) - 2)


def measure(match):
    pace_bin = _bin_of(player_pace(match))
    nb = len(SPEED_BINS) - 1
    bin_miss, bin_tot, bin_gaps = np.zeros(nb), np.zeros(nb), [[] for _ in range(nb)]
    for a, b in match.runs:
        miss = np.isnan(match.ball[a:b, 0])
        pb = pace_bin[a:b]
        np.add.at(bin_miss, pb, miss)
        np.add.at(bin_tot, pb, 1)
        m = np.concatenate(([False], miss, [False])).astype(np.int8)
        d = np.diff(m)
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            bin_gaps[pb[s]].append(e - s)
    ball_gaps, ball_vis, player_gaps = [], [], []
    n_miss = n_tot = 0
    present_per_frame = []
    for a, b in match.runs:
        miss = np.isnan(match.ball[a:b, 0])
        ball_gaps.append(_runs(miss))
        ball_vis.append(_runs(~miss))
        n_miss += miss.sum()
        n_tot += b - a
        pres = ~np.isnan(match.players[a:b, :, 0])
        present_per_frame.append(pres.sum(1))
        for k in range(pres.shape[1]):
            idx = np.flatnonzero(pres[:, k])
            if len(idx) > 1:
                inner = ~pres[idx[0]:idx[-1] + 1, k]
                player_gaps.append(_runs(inner))
    return {
        "ball_missing": n_miss / n_tot,
        "ball_gaps": np.concatenate(ball_gaps),
        "ball_vis": np.concatenate(ball_vis),
        "players_present": float(np.concatenate(present_per_frame).mean()),
        "player_gaps": np.concatenate(player_gaps) if player_gaps else np.array([25]),
        "bin_missing": bin_miss / np.maximum(bin_tot, 1),
        "bin_gaps": [np.array(g if g else [25]) for g in bin_gaps],
    }


def smooth_offset(rng, n, sigma, tau):
    """(n, 2) slowly varying offset: white noise, moving-average over tau frames, std sigma."""
    if sigma <= 0:
        return np.zeros((n, 2), np.float32)
    w = np.ones(tau) / tau
    out = np.stack([np.convolve(rng.normal(size=n + tau), w, "same")[:n] for _ in range(2)], 1)
    return (out / out.std(0) * sigma).astype(np.float32)


# Ball position offset relative to the players, chosen by a grid over sigma so that the
# nearest-player distance distribution of the corrupted external games matches the competition
# ATD's (p10 0.44 m / p50 1.91 m; sigma 0.75 fits p10, 1.1 fits p50, 0.9 splits them).
# Label-free statistics of the provided tracking only.
OFFSET_SIGMA = 0.9
OFFSET_TAU = 25


def corrupt(match, stats, seed=0, sigma=None, tau=None):
    """A copy of `match` with extra ball and player dropouts and a ball position offset."""
    rng = np.random.default_rng(seed)
    ball = match.ball.copy()
    players = match.players.copy()
    sigma = OFFSET_SIGMA if sigma is None else sigma
    tau = OFFSET_TAU if tau is None else tau
    ball += smooth_offset(rng, len(ball), sigma, tau)
    pace_bin = _bin_of(player_pace(match)) if STATE_GAPS else None
    for a, b in match.runs:
        n = b - a
        if STATE_GAPS:
            # ball: top up missingness separately in each player-pace bin, to that bin's rate in
            # the competition ATD, with gap lengths from that bin -> gaps land in stoppages
            pb = pace_bin[a:b]
            for k, target in enumerate(stats["bin_missing"]):
                frames = np.flatnonzero(pb == k)
                if not len(frames):
                    continue
                cur = np.isnan(ball[a + frames, 0]).mean()
                need = max(target - cur, 0.0) / max(1.0 - cur, 1e-6)
                gaps = stats["bin_gaps"][k]
                n_bursts = int(need * len(frames) / gaps.mean())
                for s in rng.choice(frames, n_bursts):
                    ln = int(rng.choice(gaps))
                    ball[a + s:a + min(s + ln, n)] = np.nan
            need = 0.0
        else:
            # ball: top up missingness with bursts drawn from the competition gap lengths
            cur = np.isnan(ball[a:b, 0]).mean()
            need = max(stats["ball_missing"] - cur, 0.0) / max(1.0 - cur, 1e-6)
        if need > 0:
            mean_gap = stats["ball_gaps"].mean()
            n_bursts = int(need * n / mean_gap)
            for s in rng.integers(0, n, n_bursts):
                ln = int(rng.choice(stats["ball_gaps"]))
                ball[a + s:a + min(s + ln, n)] = np.nan
        # players: drop per-player bursts until the mean count matches
        pres = ~np.isnan(players[a:b, :, 0])
        cur_present = pres.sum(1).mean()
        drop_frac = max(cur_present - stats["players_present"], 0.0) / max(cur_present, 1e-6)
        if drop_frac > 0:
            gaps = stats["player_gaps"]
            for k in range(players.shape[1]):
                alive = pres[:, k].sum()
                n_bursts = int(drop_frac * alive / gaps.mean())
                for s in rng.integers(0, n, n_bursts):
                    ln = int(rng.choice(gaps))
                    players[a + s:a + min(s + ln, n), k] = np.nan
    out = type(match)(**{k: v for k, v in match.__dict__.items() if not k.startswith("_")})
    out.ball, out.players = ball, players
    out.name = match.name + "_noisy{}".format(seed)
    return out
