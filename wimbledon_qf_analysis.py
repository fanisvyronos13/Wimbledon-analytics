# =============================================================================
# WIMBLEDON 2026 — MEN'S QUARTERFINAL DEEP-DIVE
# Extends Wimbledon_2026_Master.ipynb with a real-data, real-bracket analysis
# of the actual 2026 men's quarterfinals (built once the draw was known).
#
# Author: Fanis Vyronos
# Data: Tennis Abstract (Grass Elo ratings), Tennis Tonic (in-tournament
#       serve stats), ATP Tour / Tennis Majors / Olympics.com (results)
#
# WHY THIS IS SEPARATE FROM THE MASTER NOTEBOOK
# -----------------------------------------------------------------------------
# The master notebook predicts the FULL 128-player draw pre-tournament using
# a 70% XGBoost + 30% Grass Elo hybrid model. This notebook is narrower and
# built AFTER the draw was known: it uses only real Grass Elo (the single
# strongest validated predictor from the master notebook's own Section 10
# correlation analysis, r=0.510) plus real in-tournament serve stats for the
# 8 confirmed 2026 quarterfinalists. Predictions here are a transparent
# formula (documented below), not a black box, and every set-score and
# Monte Carlo output traces back to that one real input.
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go

pd.set_option("display.width", 120)

# =============================================================================
# SECTION 1 — REAL GRASS ELO RATINGS
# Source: Tennis Abstract, tennisabstract.com/reports/atp_elo_ratings.html
# Updated 2026-06-29. gElo is the primary signal used throughout.
# =============================================================================

_ELO_DATA = {
    "Jannik Sinner":            {"elo": 2319.8, "helo": 2263.2, "celo": 2215.7, "gelo": 2088.3},
    "Jan-Lennard Struff":       {"elo": 1694.2, "helo": 1634.2, "celo": 1659.3, "gelo": 1623.1},
    "Novak Djokovic":           {"elo": 2059.3, "helo": 2021.7, "celo": 1959.2, "gelo": 1924.9},
    "Felix Auger-Aliassime":    {"elo": 1957.9, "helo": 1932.2, "celo": 1863.9, "gelo": 1789.1},
    "Flavio Cobolli":           {"elo": 1896.3, "helo": 1809.6, "celo": 1862.4, "gelo": 1742.2},
    "Arthur Fery":              {"elo": 1720.6, "helo": 1684.5, "celo": 1570.4, "gelo": 1592.5},
    "Taylor Fritz":             {"elo": 1909.0, "helo": 1838.5, "celo": 1760.9, "gelo": 1856.6},
    "Alexander Zverev":         {"elo": 2099.7, "helo": 2041.5, "celo": 2051.1, "gelo": 1913.6},
}

# =============================================================================
# SECTION 2 — REAL WIMBLEDON 2026 SERVE STATS (Rounds 1-4, supplementary
# context only — NOT used in the win probability).
# Source: Tennis Tonic combined H2H pages, tennistonic.com, July 5-6 2026.
# NOTE on data quality: Cobolli, Fritz, and Zverev have clean cumulative R1-3
# stats from consistent preview pages. Fery's numbers mix a single-match
# snapshot (serve%, aces) with cumulative break-point data — flagged in
# data_quality below rather than silently presented as equally solid.
# =============================================================================

_RAW_QF_STATS = {
    "Jannik Sinner": {
        "aces": 81, "avg_aces": 20.3, "avg_double_faults": 3.5,
        "first_serve_won_pct": 85, "second_serve_won_pct": 54,
        "bp_won_pct": 40, "bp_conceded": 18, "bp_saved": 13,
        "avg_winners": 57.0, "data_quality": "cumulative_R1-4",
    },
    "Jan-Lennard Struff": {
        "aces": 100, "avg_aces": 25.0, "avg_double_faults": 5.5,
        "first_serve_won_pct": 82, "second_serve_won_pct": 46,
        "bp_won_pct": 40, "bp_conceded": 39, "bp_saved": 26,
        "avg_winners": 77.8, "data_quality": "cumulative_R1-4",
    },
    "Felix Auger-Aliassime": {
        "aces": 67, "avg_aces": 16.8, "avg_double_faults": 3.8,
        "first_serve_won_pct": 83, "second_serve_won_pct": 54,
        "bp_won_pct": 41, "bp_conceded": 11, "bp_saved": 10,
        "avg_winners": 59.3, "data_quality": "cumulative_R1-4",
    },
    "Novak Djokovic": {
        "aces": 52, "avg_aces": 13.0, "avg_double_faults": 2.0,
        "first_serve_won_pct": 76, "second_serve_won_pct": 60,
        "bp_won_pct": 52, "bp_conceded": 35, "bp_saved": 27,
        "avg_winners": 59.0, "data_quality": "cumulative_R1-4",
    },
    "Flavio Cobolli": {
        "aces": 42, "avg_aces": 14.0, "avg_double_faults": 4.0,
        "first_serve_won_pct": 75, "second_serve_won_pct": 50,
        "bp_won_pct": 46, "bp_conceded": None, "bp_saved": None, "bp_saved_pct": 68,
        "avg_winners": 55.0, "data_quality": "cumulative_R1-3_only",
    },
    "Arthur Fery": {
        "aces": 17, "avg_aces": 5.7, "avg_double_faults": 4.0,
        "first_serve_won_pct": 70, "second_serve_won_pct": 36,
        "bp_won_pct": 50, "bp_conceded": None, "bp_saved": None, "bp_saved_pct": 48,
        "avg_winners": 45.0, "data_quality": "mixed_R3_single_match_serve",
    },
    "Taylor Fritz": {
        "aces": 46, "avg_aces": 15.3, "avg_double_faults": 2.0,
        "first_serve_won_pct": 81, "second_serve_won_pct": 64,
        "bp_won_pct": 36, "bp_conceded": None, "bp_saved": None, "bp_saved_pct": 71,
        "avg_winners": 50.3, "data_quality": "cumulative_R1-3_only",
    },
    "Alexander Zverev": {
        "aces": 52, "avg_aces": 17.3, "avg_double_faults": 3.0,
        "first_serve_won_pct": 82, "second_serve_won_pct": 55,
        "bp_won_pct": 38, "bp_conceded": None, "bp_saved": None, "bp_saved_pct": 67,
        "avg_winners": 63.0, "data_quality": "cumulative_R1-3_only",
    },
}

df_qf_stats = pd.DataFrame.from_dict(_RAW_QF_STATS, orient="index")
derived_mask = df_qf_stats["bp_conceded"].notna()
df_qf_stats.loc[derived_mask, "bp_saved_pct"] = (
    df_qf_stats.loc[derived_mask, "bp_saved"] / df_qf_stats.loc[derived_mask, "bp_conceded"] * 100
).round(1)

df_elo = pd.DataFrame.from_dict(_ELO_DATA, orient="index")
df_qf_stats = df_qf_stats.join(df_elo)
df_qf_stats.index.name = "player"

# =============================================================================
# SECTION 3 — THE 4 CONFIRMED QUARTERFINALS (real 2026 draw)
# Source: ATP Tour, Tennis Majors, Olympics.com
# =============================================================================

QF_QUARTERFINALS_CONFIRMED = [
    {
        "qf_id": "QF1", "player_a": "Jannik Sinner", "seed_a": 1,
        "player_b": "Jan-Lennard Struff", "seed_b": None,
        "h2h": "Sinner leads 3-0 overall (1 win on grass, Halle 2024)",
        "path_a": "d. Kecmanovic, Borges, Brooksby, Mochizuki (R1-R4)",
        "path_b": "d. Baez (5 sets), Nakashima (5 sets), Medvedev, Hurkacz (ret. trailing 4-2 in 5th) (R1-R4)",
        "notes": "Struff, 36, is the oldest man in the Open Era to reach a maiden major QF.",
    },
    {
        "qf_id": "QF2", "player_a": "Novak Djokovic", "seed_a": 7,
        "player_b": "Felix Auger-Aliassime", "seed_b": 3,
        "h2h": "1-1 (Djokovic won Rome 2022, tour-level; Auger-Aliassime won the 2022 Laver Cup, exhibition)",
        "path_a": "d. Wu Yibing, Tsitsipas, Rinderknech, Safiullin (R1-R4) -- his 106th Wimbledon win, breaking Federer's record",
        "path_b": "d. Shevchenko, Prizmic, Zheng, Davidovich Fokina (5 sets) (R1-R4)",
        "notes": "Djokovic is into his 17th Wimbledon QF. Auger-Aliassime is into his 6th career Slam QF but first at Wimbledon.",
    },
    {
        "qf_id": "QF3", "player_a": "Flavio Cobolli", "seed_a": 9,
        "player_b": "Arthur Fery", "seed_b": None,
        "h2h": "1-0 Cobolli (first ever meeting)",
        "path_a": "d. Navone, Duckworth, Khachanov, De Minaur (R1-R4)",
        "path_b": "d. Dzumhur, Virtanen, Bergs, Dimitrov (5 sets, saved match points) (R1-R4)",
        "notes": "Fery, a British wildcard ranked No. 114 who grew up minutes from the All England Club, is the first British man to reach a Wimbledon QF since Andy Murray and only the 5th wildcard in the Open Era to do so.",
    },
    {
        "qf_id": "QF4", "player_a": "Taylor Fritz", "seed_a": 6,
        "player_b": "Alexander Zverev", "seed_b": 2,
        "h2h": "4-4 overall, Fritz leads 2-0 on grass (most recently Stuttgart 2026)",
        "path_a": "d. Lajovic, Kypson, Sonego, Bublik (R1-R4)",
        "path_b": "d. Blockx, Royer, Giron, vs Lehecka (leads 6-4, 7-5, 3-3 -- suspended by curfew, resumed July 7)",
        "notes": "QF4 opponent was not 100% confirmed at time of writing -- Zverev led Lehecka 2 sets to love and needed one more set to close it out.",
    },
]

# =============================================================================
# SECTION 4 — ELO WIN-PROBABILITY MODEL
# The ENTIRE basis for every win % in this notebook: one real input
# (Grass Elo), one documented formula.
# =============================================================================

def qf_elo_win_prob(df, player_a, player_b, elo_col="gelo"):
    """Standard Elo win-probability formula (Tennis Abstract's own formula):
        P(A beats B) = 1 / (1 + 10^(-(EloA - EloB)/400))
    Uses gelo (grass Elo) -- the strongest validated predictor (Spearman
    r=0.510) from the master notebook's own Section 10 analysis."""
    elo_a, elo_b = df.loc[player_a, elo_col], df.loc[player_b, elo_col]
    gap = elo_a - elo_b
    prob_a = 1 / (1 + 10 ** (-gap / 400))
    return {
        "player_a": player_a, "player_b": player_b,
        "elo_a": elo_a, "elo_b": elo_b, "gap": round(gap, 1),
        "win_prob_a": round(prob_a * 100, 1),
        "win_prob_b": round((1 - prob_a) * 100, 1),
    }

# =============================================================================
# SECTION 5 — FULL SCORELINE BREAKDOWN
# Derived from the SAME Elo win probability via the standard best-of-5
# formula (independent-sets assumption). Not a separate model.
# =============================================================================

def qf_implied_set_prob(match_prob_a, tol=1e-10):
    """Invert P(match) = p^3 * (1 + 3(1-p) + 6(1-p)^2) to solve for p =
    single-set win probability, via bisection."""
    def match_prob_from_set_prob(p):
        return p**3 * (1 + 3*(1-p) + 6*(1-p)**2)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if match_prob_from_set_prob(mid) < match_prob_a:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def qf_predict_sets(df, player_a, player_b):
    elo = qf_elo_win_prob(df, player_a, player_b)
    match_prob_a = elo["win_prob_a"] / 100
    p = qf_implied_set_prob(match_prob_a)
    q = 1 - p

    scorelines = {
        f"{player_a} 3-0": p**3, f"{player_a} 3-1": 3 * p**3 * q, f"{player_a} 3-2": 6 * p**3 * q**2,
        f"{player_b} 3-0": q**3, f"{player_b} 3-1": 3 * q**3 * p, f"{player_b} 3-2": 6 * q**3 * p**2,
    }
    exp_sets_a = (3 * (scorelines[f"{player_a} 3-0"] + scorelines[f"{player_a} 3-1"] + scorelines[f"{player_a} 3-2"])
                  + scorelines[f"{player_b} 3-1"] + 2 * scorelines[f"{player_b} 3-2"])
    exp_sets_b = (3 * (scorelines[f"{player_b} 3-0"] + scorelines[f"{player_b} 3-1"] + scorelines[f"{player_b} 3-2"])
                  + scorelines[f"{player_a} 3-1"] + 2 * scorelines[f"{player_a} 3-2"])
    most_likely = max(scorelines.items(), key=lambda kv: kv[1])

    return {
        "implied_set_prob_a": round(p, 3),
        "scorelines": {k: round(v * 100, 1) for k, v in scorelines.items()},
        "most_likely_score": most_likely[0], "most_likely_prob": round(most_likely[1] * 100, 1),
        "expected_sets_a": round(exp_sets_a, 2), "expected_sets_b": round(exp_sets_b, 2),
    }

# =============================================================================
# SECTION 6 — RADAR CHART (Grass Elo + serve stats, context only)
# =============================================================================

QF_RADAR_STATS = {
    "gelo":                 {"label": "Grass Elo (PRIMARY signal)", "range": (1550, 2150), "invert": False},
    "avg_aces":             {"label": "Aces / match",        "range": (0, 30),   "invert": False},
    "first_serve_won_pct":  {"label": "1st serve won %",     "range": (60, 95),  "invert": False},
    "second_serve_won_pct": {"label": "2nd serve won %",     "range": (35, 70),  "invert": False},
    "bp_won_pct":           {"label": "Break points won %",  "range": (20, 60),  "invert": False},
    "bp_saved_pct":         {"label": "Break points saved %","range": (40, 100), "invert": False},
    "avg_winners":          {"label": "Winners / match",     "range": (40, 90),  "invert": False},
    "avg_double_faults":    {"label": "Serve security*",     "range": (0, 8),    "invert": True},
}

def _normalize(value, lo, hi, invert):
    v = (value - lo) / (hi - lo) * 100
    v = max(0, min(100, v))
    return 100 - v if invert else v

def qf_radar_comparison(df, player_a, player_b, dark=False):
    row_a, row_b = df.loc[player_a], df.loc[player_b]
    labels = [v["label"] for v in QF_RADAR_STATS.values()]
    vals_a = [_normalize(row_a[s], *cfg["range"], cfg["invert"]) for s, cfg in QF_RADAR_STATS.items()]
    vals_b = [_normalize(row_b[s], *cfg["range"], cfg["invert"]) for s, cfg in QF_RADAR_STATS.items()]

    color_a, color_b = ("#5DA9E9", "#FF6B6B") if dark else ("#1f77b4", "#d62728")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_a + [vals_a[0]], theta=labels + [labels[0]],
                                   name=player_a, fill="toself", line_color=color_a, opacity=0.75))
    fig.add_trace(go.Scatterpolar(r=vals_b + [vals_b[0]], theta=labels + [labels[0]],
                                   name=player_b, fill="toself", line_color=color_b, opacity=0.75))

    layout_kwargs = dict(
        title=f"{player_a} vs {player_b} -- Wimbledon 2026<br>"
              f"<sub>Grass Elo axis = the actual driver of the win %. All other axes are "
              f"in-tournament serve stats, shown for context only.</sub>",
        height=580, width=680,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
    )
    if dark:
        layout_kwargs.update(
            paper_bgcolor="#111111", plot_bgcolor="#111111", font=dict(color="white"),
            polar=dict(bgcolor="#111111",
                       radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
                                        gridcolor="#444444", linecolor="#444444"),
                       angularaxis=dict(gridcolor="#444444", linecolor="#444444")),
        )
    else:
        layout_kwargs.update(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)))

    fig.update_layout(**layout_kwargs)
    return fig

# =============================================================================
# SECTION 7 — THE AGENT (ties Elo + scoreline + radar together)
# =============================================================================

class WimbledonQFAgent:
    def __init__(self, stats_df, quarterfinals):
        self.stats = stats_df
        self.qfs = quarterfinals

    def find_qf(self, name):
        for qf in self.qfs:
            if name.lower() in qf["player_a"].lower() or name.lower() in qf["player_b"].lower():
                return qf
        return None

    def analyze_matchup(self, player_a, player_b, show_charts=False):
        qf = self.find_qf(player_a)
        elo = qf_elo_win_prob(self.stats, player_a, player_b)
        sets = qf_predict_sets(self.stats, player_a, player_b)

        print(f"=== {player_a} vs {player_b} ===")
        print(f"Head-to-head: {qf['h2h']}")
        print(f"{player_a}'s path: {qf['path_a']}")
        print(f"{player_b}'s path: {qf['path_b']}")
        print(f"Context: {qf['notes']}\n")

        print("PRIMARY SIGNAL -- Grass Elo (Tennis Abstract, validated r=0.510 predictor):")
        print(f"  {player_a}: {elo['elo_a']}  |  {player_b}: {elo['elo_b']}  (gap: {elo['gap']:+.1f})")
        print(f"  >>> Win probability: {player_a} {elo['win_prob_a']}%  /  {player_b} {elo['win_prob_b']}% <<<\n")

        print("PREDICTED SET SCORE (derived from the same Elo probability, best-of-5 formula):")
        print(f"  Most likely score: {sets['most_likely_score']} ({sets['most_likely_prob']}%)")
        print(f"  Expected sets won: {player_a} {sets['expected_sets_a']}  |  {player_b} {sets['expected_sets_b']}")
        for score, prob in sets["scorelines"].items():
            print(f"    {score}: {prob}%")
        print()

        if show_charts:
            qf_radar_comparison(self.stats, player_a, player_b, dark=False).show()
            qf_radar_comparison(self.stats, player_a, player_b, dark=True).show()

# =============================================================================
# SECTION 8 — VECTORIZED MONTE CARLO SIMULATION
# Runs 1,000,000 simulations in well under a second (fully vectorized with
# NumPy -- an earlier pure-Python-loop version of this took minutes).
# =============================================================================

def qf_run_monte_carlo(df, qf1, qf2, qf3, qf4, n_sims=1_000_000, seed=42):
    """Bracket: SF1 = winner(qf1) vs winner(qf2); SF2 = winner(qf3) vs winner(qf4).
    Uses the real gElo win-probability formula for every simulated match."""
    players = df.index.tolist()
    idx = {p: i for i, p in enumerate(players)}
    elo = df["gelo"].values
    gap = elo[:, None] - elo[None, :]
    prob_matrix = 1 / (1 + 10 ** (-gap / 400))

    rng = np.random.default_rng(seed)

    def play(pair, n):
        a, b = idx[pair[0]], idx[pair[1]]
        p_a = prob_matrix[a, b]
        return np.where(rng.random(n) < p_a, a, b)

    def play_pairs(a_arr, b_arr):
        p_a = prob_matrix[a_arr, b_arr]
        return np.where(rng.random(len(a_arr)) < p_a, a_arr, b_arr)

    w1, w2, w3, w4 = play(qf1, n_sims), play(qf2, n_sims), play(qf3, n_sims), play(qf4, n_sims)
    sf1_w, sf2_w = play_pairs(w1, w2), play_pairs(w3, w4)
    champion = play_pairs(sf1_w, sf2_w)

    sf_counts = np.bincount(np.concatenate([w1, w2, w3, w4]), minlength=len(players))
    f_counts = np.bincount(np.concatenate([sf1_w, sf2_w]), minlength=len(players))
    w_counts = np.bincount(champion, minlength=len(players))

    rows = []
    for i, p in enumerate(players):
        rows.append({
            "player": p, "QF_pct": 100.0,
            "SF_pct": round(sf_counts[i] / n_sims * 100, 1),
            "F_pct": round(f_counts[i] / n_sims * 100, 1),
            "Win_pct": round(w_counts[i] / n_sims * 100, 1),
        })
    out = pd.DataFrame(rows).sort_values("Win_pct", ascending=False).reset_index(drop=True)

    assert abs(out["SF_pct"].sum() - 400) < 0.5
    assert abs(out["F_pct"].sum() - 200) < 0.5
    assert abs(out["Win_pct"].sum() - 100) < 0.5

    # Also track the single most likely FINAL pairing (not just individual
    # reach-final odds), used for the "most likely final" insight.
    from collections import Counter
    pairs = [tuple(sorted((players[a], players[b]))) for a, b in zip(sf1_w, sf2_w)]
    top_final_pairs = Counter(pairs).most_common(3)
    top_final_pairs = [(pair, round(count / n_sims * 100, 1)) for pair, count in top_final_pairs]

    return out, top_final_pairs

# =============================================================================
# SECTION 9 — LEADERBOARD HEATMAP (Top Contenders / Dark Horses style)
# =============================================================================

CUSTOM_SCALE = [
    [0.0, "#0B1F3A"], [0.25, "#123A5E"], [0.5, "#1B6E6E"], [0.75, "#3FAE6A"], [1.0, "#3ED17A"],
]

def qf_plot_leaderboard_heatmap(mc_results, seeds_dict, dark=True, n_top=4, save_html=None):
    players = mc_results["player"].tolist()
    rows = ["Win_pct", "F_pct", "SF_pct"]
    row_labels = ["Win Title", "Reach Final", "Reach SF"]

    z, text = [], []
    for r in rows:
        vals = mc_results.set_index("player").loc[players, r].values.astype(float)
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        z.append(norm)
        text.append([f"{v:.1f}%" for v in vals])

    bg = "#0E1526" if dark else "#FFFFFF"
    text_color = "#FFFFFF" if dark else "#0E1526"
    muted = "#9AA5C0" if dark else "#6B6B6B"

    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"col{i}" for i in range(len(players))], y=row_labels,
        text=text, texttemplate="%{text}",
        textfont={"size": 13, "color": text_color},
        colorscale=CUSTOM_SCALE, showscale=False, xgap=5, ygap=5,
    ))

    tick_labels = [f"{i+1}. {players[i]} [{seeds_dict.get(players[i], '?')}]" for i in range(len(players))]
    fig.update_xaxes(tickvals=[f"col{i}" for i in range(len(players))], ticktext=tick_labels,
                      tickangle=-25, tickfont=dict(size=10, color=text_color),
                      showgrid=False, side="bottom")
    fig.update_yaxes(tickfont=dict(size=13, color=text_color), showgrid=False, autorange="reversed")

    fig.add_annotation(x=0.5, y=1.5, xref="x", yref="paper", yanchor="bottom",
                        text="TOP CONTENDERS", showarrow=False,
                        font=dict(size=12, color="#E8B84B", family="Arial Black"), xanchor="left")
    fig.add_annotation(x=n_top + 1.5, y=1.5, xref="x", yref="paper", yanchor="bottom",
                        text="DARK HORSES & OUTSIDERS", showarrow=False,
                        font=dict(size=12, color=muted, family="Arial"), xanchor="left")
    fig.add_shape(type="line", x0=n_top - 0.5, x1=n_top - 0.5, y0=-0.5, y1=len(rows) - 0.5,
                  xref="x", yref="y", line=dict(color=muted, width=1.5, dash="dot"))

    fig.update_layout(
        title=dict(
            text="Wimbledon 2026 -- Quarterfinalist Title-Race Probabilities<br>"
                 "<sub>Monte Carlo simulation - 1,000,000 iterations - Grass Elo model - "
                 "n=8 confirmed quarterfinalists</sub>",
            font=dict(color=text_color, size=17), x=0.02, xanchor="left",
        ),
        paper_bgcolor=bg, plot_bgcolor=bg,
        height=480, width=980,
        margin=dict(l=110, t=110, b=140, r=30),
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")
        print(f"Saved: {save_html}")

    return fig


# =============================================================================
# SECTION 10 — RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    agent = WimbledonQFAgent(df_qf_stats, QF_QUARTERFINALS_CONFIRMED)

    QF1 = ("Jannik Sinner", "Jan-Lennard Struff")
    QF2 = ("Novak Djokovic", "Felix Auger-Aliassime")
    QF3 = ("Flavio Cobolli", "Arthur Fery")
    QF4 = ("Taylor Fritz", "Alexander Zverev")

    print("=" * 90)
    print("QUARTERFINAL-BY-QUARTERFINAL ANALYSIS")
    print("=" * 90)
    for a, b in (QF1, QF2, QF3, QF4):
        agent.analyze_matchup(a, b)
        print("-" * 90)

    print("\n" + "=" * 90)
    print("MONTE CARLO SIMULATION -- ROAD TO THE TITLE")
    print("=" * 90)
    qf_montecarlo_results, top_final_pairs = qf_run_monte_carlo(df_qf_stats, QF1, QF2, QF3, QF4)
    print(qf_montecarlo_results.to_string(index=False))
    print("\nMost likely FINAL pairings:")
    for pair, pct in top_final_pairs:
        print(f"  {pair[0]} vs {pair[1]}: {pct}%")

    QF_SEEDS = {
        "Jannik Sinner": "1", "Alexander Zverev": "2", "Novak Djokovic": "7",
        "Taylor Fritz": "6", "Flavio Cobolli": "9", "Felix Auger-Aliassime": "3",
        "Arthur Fery": "WC", "Jan-Lennard Struff": "-",
    }
    qf_plot_leaderboard_heatmap(qf_montecarlo_results, QF_SEEDS, dark=True,
                                 save_html="wimbledon_qf_leaderboard_dark.html")

    df_qf_stats.round(2).to_csv("wimbledon_qf_player_stats.csv")
    print("\nSaved: wimbledon_qf_player_stats.csv")
