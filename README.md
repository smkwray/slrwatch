# SLR Watch

**[smkwray.github.io/slrwatch](https://smkwray.github.io/slrwatch/)**

A public-data research project studying how Supplementary Leverage Ratio pressure shapes large-bank Treasury holdings and balance-sheet behavior. The pipeline is built entirely from free regulatory filings and market data, covering FFIEC Call Reports, FR Y-9C, FR Y-15, NY Fed dealer statistics, and FINRA TRACE.

## Why this matters

The Supplementary Leverage Ratio requires large banks to hold Tier 1 capital against all on-balance-sheet assets without risk-weighting. Because SLR treats Treasuries the same as corporate loans in the denominator, leverage-constrained banks face a real cost to holding Treasuries. That link matters for Treasury-market intermediation, monetary-policy transmission, and capital-rule design.

When the Fed temporarily excluded Treasuries and reserves from SLR in April 2020, it created a natural experiment for measuring how leverage-ratio relief affects bank Treasury demand. SLR Watch uses that event as its causal core and extends the question with a broader constraint-regime analysis.

## Core finding: the 2020 SLR exclusion

Banks that appear more balance-sheet constrained increased Treasury holdings more after the temporary exclusion than comparison banks did, but the repo now distinguishes the descriptive universe from the causal SLR sample instead of treating them as the same object.

**Universe A: all insured-bank descriptive universe (2019Q1-2021Q4)**
- 5,488 insured-bank filers
- 61,626 bank-quarter observations

**Universe B: SLR-reporting insured banks**
- 37 entities
- 426 bank-quarter observations

**Universe C: treatment-definable SLR sample**
- 20 entities
- 233 bank-quarter observations
- Requires a usable 2019Q4 baseline for treatment assignment

**Universe D: primary causal core**
- 19 entities
- 228 balanced bank-quarter observations
- Low-headroom banks: +1.78pp Treasury inventory increase relative to controls (p = 0.016)
- Covered-bank subsidiaries: +1.62pp (p = 0.021)

**Universe F: flagship per-parent clustered inference (17 parent clusters)**
- 17 entities
- 204 bank-quarter observations
- Low headroom: +1.76pp (p = 0.181)
- Covered bank: +1.61pp (p = 0.245)

Under parent-level clustering in the flagship sample, coefficients retain sign and approximate magnitude but lose conventional significance. These results should be described as directional evidence rather than a settled causal estimate.

The repo now also writes explicit diagnostics. In Universe D, the Treasury pre-trend joint p-values are 0.103 for the low-headroom split and 0.333 for the covered-bank split. But the low-headroom split shows a nontrivial fake-date placebo in the clustered flagship sample (+2.30pp, p = 0.010 before the actual exemption window), so the covered-bank treatment is currently the cleaner regulatory treatment and the low-headroom split should be treated more cautiously.

## Mechanism evidence

Five extension reports support the core result with a consistent mechanism story:

- **Reallocation, not expansion.** Constrained banks shifted balance-sheet capacity toward Treasuries while reducing deposits and loans. Low-headroom banks: Treasury +2.47pp, deposit growth -3.52pp, loan growth -1.65pp.
- **Safe-asset composition shift.** The response was within safe assets, not just between safe and risky. Covered banks shifted their safe-asset mix toward Treasuries by +9.86pp relative to controls.
- **Family-level transmission.** Bank and parent Treasury-share changes moved in the same direction in 73.3% of linked quarter-over-quarter comparisons across 16 parent families.
- **Trading-balance-sheet tradeoff.** Constrained banks reduced trading-asset share while Treasury holdings rose, consistent with leverage-capacity reallocation.
- **Policy-regime context.** A longer quarterly panel (2019-2026) places the 2020 event within a broader safe-asset absorption story across pre-exclusion, exclusion, post-exclusion, and QT-era regimes.

## Constraint decomposition

A live decomposition module extends the project from "did SLR relief matter in 2020?" to "which balance-sheet constraint matters in which regime?" It compares leverage headroom, duration-loss pressure, and funding stress across insured-bank and parent/IHC panels through 2025Q4, using a scorecard built from SLR headroom, Treasury unrealized-loss proxies, deposit runoff, repo reliance, deposit funding gaps, liquid buffers, and HTM mix.

Key results:

- In the 2022-2023 duration-loss window, duration loss is the dominant bucket for insured banks in 65.9% of observations and for parents/IHCs in 63.0%.
- By late QT normalization, insured banks still lean duration loss at 42.1%, while parents/IHCs now lean back toward leverage at 35.4%.
- Linked parent-bank families match on the dominant constraint in 64.8% of family-quarters during the duration-loss window; both bank and parent are duration-loss dominant in 48.6% of those observations.

A first interaction-regression layer provides supporting evidence on the parent panel: higher duration pressure is associated with higher Treasury share in the 2022-2023 window (coefficient 0.027, p = 0.001) and lower Treasury share during late QT normalization (-0.034, p = 0.026). Bank-side interaction results are weaker. This layer supports the descriptive decomposition rather than constituting a separate causal claim.

## Public data sources

| Source | Use | Cadence |
|--------|-----|---------|
| FFIEC Call Reports | Main insured-bank panel | Quarterly |
| FR Y-9C (Chicago Fed / NIC) | Parent holding companies | Quarterly |
| FR Y-15 Snapshots | GSIB surcharge context | Annual |
| OFR Bank Systemic Risk Monitor | Surcharge enrichment | Mixed |
| NY Fed Primary Dealer Statistics | Market context overlay | Weekly |
| FINRA TRACE Treasury | Market context overlay | Weekly / Monthly |

## Repository structure

```
config/          Rule regime definitions, variable registry, source manifest
src/slr_watch/   Python package: rules, headroom, ingestion, panels, analytics
data/            Raw, staged, and derived data (generated by pipeline)
output/reports/  Event-study results and extension reports
tests/           Unit and integration tests
site/            Static GitHub Pages site
```

## Reproducing the outputs

```bash
# Install (core analysis — reproducible, pinned dependencies)
pip install -r requirements.lock && pip install -e ".[dev]" --no-deps

# Or install with latest compatible versions
pip install -e ".[dev]"

# Optional: browser automation for FR Y-9C downloads
pip install -e ".[browser]"
python -m playwright install chromium

# Run tests
python -m pytest -q

# Download and stage data (example quarters)
python -m slr_watch.cli download-call-reports --quarter 2020Q1
python -m slr_watch.cli stage-call-reports --quarter 2020Q1
python -m slr_watch.cli download-fry9c --quarter 2025Q4
python -m slr_watch.cli stage-fry9c --quarter 2025Q4 --input data/raw/fry9c/2025Q4/BHCF20251231.ZIP

# Build panels
python -m slr_watch.cli build-crosswalk
python -m slr_watch.cli build-fdic-institutions
python -m slr_watch.cli build-all-insured-panel
python -m slr_watch.cli build-insured-panel
python -m slr_watch.cli build-parent-panel

# Run analysis (or `make reproduce` to run all steps below at once)
python -m slr_watch.cli run-event-study
python -m slr_watch.cli run-reallocation-report
python -m slr_watch.cli run-safe-asset-absorption-report
python -m slr_watch.cli run-parent-transmission-report
python -m slr_watch.cli run-treasury-intermediation-report
python -m slr_watch.cli run-policy-regime-panel-report
python -m slr_watch.cli run-constraint-decomposition-report
python -m slr_watch.cli build-site-data
```

The event-study pipeline now writes `output/reports/event_2020/sample_manifest.csv`, `sample_ladder.csv`, `methodology_memo.md`, and `gpt_pro_next_steps_prompt.md` so the exact descriptive universe, causal sample ladder, exclusion reasons, and next-question handoff are explicit.

See `python -m slr_watch.cli --help` for the full command set.

For current FR Y-9C quarters, the NIC downloader first tries browser automation. If FFIEC blocks that flow, it automatically checks for the expected ZIP in the raw quarter folder, the current working directory, and `~/Downloads` before failing, so the practical fallback is to download the ZIP manually and rerun the same command.

## Project site

A static research microsite with interactive charts and detailed findings is available at **[smkwray.github.io/slrwatch](https://smkwray.github.io/slrwatch/)** (deployed from [`site/`](site/)).

```bash
cd site && python -m http.server 8000
```

## License

See [LICENSE](LICENSE).
