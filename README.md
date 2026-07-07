## 🎾 Wimbledon 2026 Analytics
What actually wins Wimbledon? A data-driven investigation into the statistics that predict grass-court success, built into an XGBoost + Monte Carlo simulation model to predict the 2026 champion.

## Key Findings

Grass Elo dominates everything. A player's historical grass court performance is the single strongest predictor of Wimbledon success — stronger than any in-match statistic.
Return wins titles. Serving wins matches. Return Points Won % is the 2nd strongest predictor, ahead of every serve metric.
The "big server" myth is busted. Ace rate is negatively correlated with how far a player goes at Wimbledon.


## Methodology

Data: 80 players who reached Round 16 or better across 5 Wimbledons (2019, 2021-2024)
Match-level dataset: 75 historical R16+ matches restructured into 150 head-to-head observations
Features: serve stats, return stats, tiebreak performance, points dominance, Grass Elo, pre-Wimbledon form
Model: Hybrid prediction — 70% XGBoost + 30% Grass Elo
Validation: 5-fold cross-validated AUC = 0.788
Prediction engine: Monte Carlo simulation, 10,000 iterations, full 128-player draw

## 🏆 2026 Predictions (Top 5)

| Rank | Player | Win Probability |
|------|--------|------------------|
| 🥇 | Jannik Sinner | 27.5% |
| 🥈 | Alex De Minaur | 9.2% |
| 🥉 | Taylor Fritz | 5.2%* |
| 4 | Hubert Hurkacz | 4.0% |
| 5 | Felix Auger-Aliassime | 3.4% |

*Updated to reflect Jack Draper's withdrawal — Fritz now faces Dusan Lajovic in R1.

## Draw Luck Analysis

Most unlucky: Casper Ruud (-52.7%) — drawing Hubert Hurkacz in Round 1
Luckiest: Alexander Bublik (+12.6%) — drawing Thanasi Kokkinakis

## Quarterfinal Deep-Dive

Once the real 2026 draw was known, a narrower follow-up analysis (`quarterfinals/`)
switched from the pre-tournament hybrid model to pure real Grass Elo — the single
strongest validated predictor from this project's own Section 10 correlation
analysis — plus real in-tournament serve stats for the 8 confirmed quarterfinalists.

- **Win probability**: one formula, `P(A beats B) = 1 / (1 + 10^-(EloA-EloB)/400)`
- **Scoreline breakdowns**: derived from that same probability via the standard best-of-5 formula
- **Monte Carlo**: 1,000,000 vectorized simulations of the real bracket
- **Most likely final**: Sinner vs Zverev (31.8%)

See [`quarterfinals/wimbledon_qf_analysis.py`](quarterfinals/wimbledon_qf_analysis.py)
and the [LinkedIn carousel](Wimbledon_2026_QF_Carousel.pdf).


## Repository Structure

README.md
wimbledon_2026_master.ipynb
data/ — wimbledon_data.csv, wimbledon_matchup.csv, wimbledon_2026_predictions.csv
charts/ — wimbledon_correlations_dark.html, wimbledon_heatmap_dark.html, wimbledon_validation_dark.html, wimbledon_monte_carlo_dark.html, wimbledon_draw_luck_dark.html, wimbledon_r1_watchlist_dark.html
images/ — PNG screenshots of the charts above


## Tech Stack
Python, pandas, numpy, scikit-learn, XGBoost, scipy, Plotly

## Limitations

2026 grass form sourced from available ATP leaderboard data at time of analysis
Head-to-head rivalry effects not explicitly modelled
Pre-Wimbledon form based on Halle/Queen's Club 2026 results only


## Author
Fanis Vyronos


## License
MIT
