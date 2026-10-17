# SLR Watch

SLR Watch studies large-bank Treasury holdings and balance-sheet constraints
using public regulatory filings and market data. Its event study examines the
2020 temporary exclusion of Treasury securities and reserve balances from the
Supplementary Leverage Ratio (SLR). Separate descriptive modules compare leverage
headroom, duration losses, and funding stress across banks and parent companies.

## Findings and interpretation

The parent-clustered Treasury estimates do not show a reliable positive effect
of the 2020 exclusion. The sample with 16 parent clusters estimates a −0.59 percentage-point
Treasury-share difference for low-headroom banks (p = 0.690) and a −0.23
percentage-point difference for directly eligible banks (p = 0.911).
Leave-one-parent-out ranges cross zero. These results are inconclusive; they do
not support a general claim that SLR relief increases Treasury absorption.

Treatment assignment requires a usable 2019Q4 baseline. The project keeps the full descriptive universe, treatment-definable sample,
balanced panel, and parent-clustered sample distinct. Sample rosters and exclusion
reasons accompany the event study. Pretrend and placebo diagnostics do not by
themselves establish causal identification.

Reallocation, safe-asset composition, parent-bank co-movement, and constraint
scorecards provide descriptive context. Their associations and classifications
are not separately identified causal mechanisms or estimates of the Treasury
Deposit Component. The [research site](https://smkwray.github.io/slrwatch/)
presents the event study and supporting diagnostics.

## Install and validate

Python 3.11 or later is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -B -m pytest -q
ruff check .
python -m slr_watch.cli --help
```

The default suite uses synthetic inputs and committed display assets. After
rebuilding the empirical reports, `python -B -m pytest -q -m integration` checks
the public site against those reports; missing inputs fail that lane.

For the locked analysis environment, install `requirements.lock` before the
editable package. Browser-assisted FR Y-9C acquisition is optional:

```bash
python -m pip install -e '.[browser]'
python -m playwright install chromium
```

## Sources and workflow

| Source | Role |
|---|---|
| FFIEC Call Reports | Insured-bank quarterly balance sheets |
| FR Y-9C | Parent holding-company balance sheets |
| FR Y-15 and OFR | Systemic-risk and surcharge context |
| NY Fed primary dealer statistics | Intermediation context |
| FINRA TRACE Treasury aggregates | Trading context |

Download and stage the quarters required by the configured sample. For example:

```bash
python -m slr_watch.cli download-call-reports --quarter 2020Q1
python -m slr_watch.cli stage-call-reports --quarter 2020Q1
python -m slr_watch.cli download-fry9c --quarter 2025Q4
python -m slr_watch.cli stage-fry9c --quarter 2025Q4 --input data/raw/fry9c/2025Q4/BHCF20251231.ZIP
```

Call Report extraction is temporary; normalized identity and Parquet tables are
retained alongside the source ZIP. If browser acquisition is blocked, place the
required FR Y-9C ZIP in its raw quarter directory and rerun staging.

After supplying the required source quarters:

```bash
python -m slr_watch.cli build-crosswalk
python -m slr_watch.cli build-fdic-institutions
python -m slr_watch.cli build-all-insured-panel
python -m slr_watch.cli build-insured-panel
python -m slr_watch.cli build-parent-panel
python -m slr_watch.cli run-event-study
```

`make reproduce` runs the configured analysis and site-data sequence. Individual
commands expose the reallocation, safe-asset, parent-transmission, intermediation,
policy-regime, and constraint-decomposition reports; see the CLI help.

## Outputs

`output/reports/event_2020/` contains the event-study estimates, sample manifest,
sample ladder, treatment roster, and methodology memo. Other reports retain their
own samples and definitions. Raw, staged, and derived data under `data/` and
analysis outputs under `output/` are local, generated files ignored by Git,
except the two small constraint-decomposition summaries retained for public-copy
validation.

`site/` contains the curated research-site assets. Serve it locally with
`python -m http.server 8000 --directory site`.

See [LICENSE](LICENSE) for license terms.
