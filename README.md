# SLR Watch

SLR Watch analyzes large US bank Treasury holdings and balance-sheet leverage constraints from public regulatory filings and market data. It downloads and normalizes quarterly bank Call Reports from the Federal Financial Institutions Examination Council (FFIEC), the Federal Reserve's FR Y-9C consolidated financial statements for holding companies, and market overlays; builds insured-bank and parent panels; and runs an event study of the 2020 temporary exclusion of Treasury securities and reserve balances from the Supplementary Leverage Ratio (SLR) alongside six descriptive balance-sheet and market reports.

## Data

All inputs are public and require no API keys. Run `slr-watch print-sources` to inspect the full manifest in `config/source_manifest.yml`.

- Call Reports (FFIEC Central Data Repository bulk ZIPs): quarterly insured-bank balance sheets.
- FR Y-9C: quarterly bank holding company and intermediate holding company balance sheets (direct Chicago Fed CSV archive for 1986Q3–2021Q1; FFIEC National Information Center (NIC) ZIP via Playwright or `--zip-path` for 2021Q2 onward).
- Federal Deposit Insurance Corporation (FDIC) Institutions metadata API (`build-fdic-institutions`): charter and parent-hierarchy reference.
- Snapshots of the Federal Reserve's FR Y-15 systemic risk report and the Office of Financial Research (OFR) Bank Systemic Risk Monitor (`build-fry15-overlay`): annual Method 1 surcharge overlay for global systemically important banks (G-SIBs).
- NY Fed Primary Dealer Statistics API (`build-primary-dealer-overlay`, dealer positions in USD millions) and Treasury aggregates from the Financial Industry Regulatory Authority's Trade Reporting and Compliance Engine (FINRA TRACE; `build-trace-overlay`, manual CSV or XLSX input in USD billions): quarterly market overlays combined by `build-market-overlay-panel`.

## Install

Python 3.11 or later is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

To use the locked dependency set, run `python -m pip install -r requirements.lock` before installing the editable package. For browser-assisted FR Y-9C and FR Y-15 downloads:

```bash
python -m pip install -e '.[browser]'
python -m playwright install chromium
```

## Usage

Download and stage each quarter in your sample. The 2020 event study needs Call Reports for every quarter from 2019Q1 through 2021Q4; the results reported below were built from Call Reports for 2019Q1–2025Q4 and FR Y-9C filings for 2020Q1–2025Q4. For example:

```bash
slr-watch download-call-reports --quarter 2020Q1
slr-watch stage-call-reports --quarter 2020Q1
slr-watch download-fry9c --quarter 2025Q4
slr-watch stage-fry9c --quarter 2025Q4 --input data/raw/fry9c/2025Q4/BHCF20251231.ZIP
```

Call Report ZIPs are extracted to a temporary directory during staging; compact identity and normalized Parquet tables are saved under `data/staging/`. If automated NIC downloads are blocked, download the `BHCFYYYYMMDD.ZIP` archive manually and pass it to `download-fry9c --zip-path` or `stage-fry9c --input`.

Build the crosswalk, panels, and 2020 event study:

```bash
slr-watch build-crosswalk
slr-watch build-fdic-institutions
slr-watch build-all-insured-panel
slr-watch build-insured-panel
slr-watch build-parent-panel
slr-watch run-event-study
```

Run `make reproduce` to build the panels and all seven reports from the staged quarters (event study, reallocation, safe-asset absorption, parent transmission, Treasury intermediation, policy-regime panel, and constraint decomposition). `python -m slr_watch.cli` invokes the same interface; run `slr-watch --help` for all commands, including `demo-headroom`.

## Output

Raw downloads, staged filings, derived panels, and generated reports under `data/` and `output/` are local and ignored by Git, except two tracked summary CSVs under `output/reports/constraint_decomposition/`.

| Path | Description |
| --- | --- |
| `data/derived/insured_bank_panel.parquet` | Quarterly insured-bank balance-sheet, Treasury-share, and SLR-headroom panel |
| `data/derived/parent_panel.parquet` | Quarterly holding-company balance-sheet and leverage panel |
| `output/reports/event_2020/` | 2020 SLR exclusion difference-in-differences estimates, parent-clustered checks, placebos, sample rosters, and methodology memo |
| `output/reports/` | Additional report folders for reallocation, safe-asset absorption, parent transmission, Treasury intermediation, policy-regime panel, and constraint decomposition |

## Event-study results and limits

In the parent-clustered 2020 SLR exclusion event study (16 parent clusters), estimated Treasury-share differences are negative and statistically indistinguishable from zero: -0.59 percentage-point for low-headroom banks (p = 0.690) and -0.23 percentage-point for directly eligible banks (p = 0.911). Leave-one-parent-out estimates cross zero for both specifications. These results are inconclusive and do not support a claim that SLR relief increased bank Treasury absorption.

- Treatment assignment requires a usable 2019Q4 baseline and is held fixed on that baseline across samples. The event-study tables keep the full descriptive universe, treatment-definable sample, balanced panel, and parent-clustered sample distinct.
- Pretrend and placebo checks do not establish causal identification.
- The reallocation, safe-asset, parent-transmission, intermediation, policy-regime, and constraint-decomposition reports are descriptive and do not estimate causal mechanisms.

## Tests

The default suite uses synthetic inputs and runs offline:

```bash
python -B -m pytest -q
ruff check .
```

After rebuilding the empirical reports locally, run the integration test to verify the event-study figures reported in this README against `output/reports/event_2020/`:

```bash
python -B -m pytest -q -m integration
```

## License

MIT. See [LICENSE](LICENSE).
