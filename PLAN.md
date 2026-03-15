# Project plan

## Project name
`slr-watch` — a large-bank SLR / Treasury-capacity monitor.

## Goal

Build a reproducible public-data pipeline that estimates:

1. **SLR headroom** for large banks and selected foreign-bank U.S. entities.
2. **Treasury-capacity proxies** based on balance-sheet room.
3. **Balance-sheet behavior** when leverage pressure tightens or loosens.
4. A small number of **clean causal designs**, especially around the 2020 temporary exclusion.

## Core research questions

1. Which institutions appear most constrained by SLR or eSLR, quarter by quarter?
2. How much incremental balance-sheet capacity did they have, in dollars?
3. When headroom tightened, did Treasury holdings, Fed balances, reverse repos, or trading assets adjust?
4. Did the 2020 temporary exclusion produce different balance-sheet responses for low-headroom institutions?
5. After the 2025 final rule takes effect in 2026, how much measured headroom changes mechanically under the new calibration?

## Scope

### Included in v1
- U.S. GSIB parent BHCs.
- Major insured depository subsidiaries.
- Large-bank controls.
- U.S. IHCs of major foreign banking organizations.
- A working rule engine and quarterly panel logic.
- A flagship event study around the 2020 temporary exclusion.

### Deferred to v1.5 / v2
- FFIEC 002 branch/agency panel.
- Foreign-parent Pillar 3 enrichments.
- TRACE Treasury overlays.
- New York Fed primary dealer overlays.
- UI dashboard beyond a minimal prototype.

### Explicitly out of scope for v1
- Every U.S. bank.
- Named-bank daily Treasury desk attribution from free public data.
- Full global-bank coverage.
- Heavy machine learning.

## Deliverables

### v0.1 seed
- repo scaffold
- source manifest
- variable registry
- rule regime file
- headroom math
- sample data
- tests
- parser skeletons

### v0.2 public-data pipeline
- automated ingestion for the most important sources
- normalized entity-quarter tables
- first derived metrics
- quality checks

### v0.3 analysis release
- 2020 event study
- summary figures and tables
- bank cards
- first dashboard or notebook pack

## Data strategy

### 1) FR Y-9C
Use historical Chicago Fed direct files where convenient, then automate current/recent retrieval from the NIC Financial Data Download page.

Use cases:
- parent-level Tier 1 capital
- parent-level total leverage exposure where available
- Treasury holdings
- trading assets
- total assets

### 2) FFIEC Call Reports
Use FFIEC bulk schedule files as the main empirical panel for insured-bank subsidiaries.

Use cases:
- bank-level total leverage exposure
- Tier 1 capital
- Treasury holdings
- balances due from Federal Reserve Banks
- repos/reverse repos
- trading assets
- loans, deposits, funding proxies

### 3) FR Y-15
Use snapshot files for method 1 surcharge context and systemic-intensity overlays.

### 4) OFR Bank Systemic Risk Monitor
Use for enrichment and ranking context, not as the base accounting dataset.

### 5) TRACE Treasury / NY Fed primary dealer stats
Use as market context only. Do not overpromise named-bank dealer attribution.

### 6) Foreign-bank sources
Prioritize U.S. IHCs first. Add FFIEC 002 next. Treat foreign parents as a later enrichment step.

## Entity model

Each observation should be keyed by:

- `entity_id`
- `quarter_end`
- `entity_type`

Entity types:
- `bhc_parent`
- `insured_bank_sub`
- `ihc_fbo_us`
- `foreign_branch_agency_us`

Important crosswalk fields:
- RSSD ID
- FDIC cert
- LEI if available
- top parent RSSD
- legal name history
- country / home-jurisdiction marker
- FR Y-15 reporter flag

## Core variables

### Required first-pass raw items
- Tier 1 capital
- total leverage exposure
- total assets / average assets
- Treasury holdings (HTM, AFS, trading)
- Fed balances
- repos / reverse repos
- trading assets
- deposits
- loans

### Derived metrics
- `actual_slr`
- `required_slr`
- `headroom_pp`
- `headroom_dollars`
- `ust_inventory_fv`
- `ust_share_assets`
- `ust_share_headroom`

## Rule logic

### Pre-2026 baseline
- SLR minimum: 3%.
- GSIB parent eSLR buffer: +2 percentage points.
- Covered-bank subsidiary well-capitalized threshold: 6%.

### Final rule effective 2026-04-01
- GSIB parent buffer: `0.5 * method_1_surcharge`.
- Covered-bank subsidiary buffer: `min(1.0 percentage point, 0.5 * parent method_1_surcharge)`.
- Both sit on top of the 3% minimum.
- Optional early adoption begins 2026-01-01.

## Empirical design

### Main flagship design: 2020 temporary exclusion
Treatment options:
- low pre-shock headroom vs. high pre-shock headroom
- GSIB / covered-bank exposure vs. large-bank controls
- high Treasury inventory share vs. low Treasury inventory share

Outcomes:
- Treasury holdings
- Fed balances
- reverse repos
- repo financing
- trading assets
- deposit growth
- loan growth

Specifications:
- panel event study
- diff-in-diff
- local projections

## Quality plan

### Data QA
- schema checks
- duplicate key checks
- sign / range checks
- revision-awareness
- variable lineage tracking

### Analytical QA
- headroom formula unit tests
- regime-switch unit tests
- cross-source reconciliation where possible
- sensitivity tables for alternative Treasury-inventory definitions

## Acceptance criteria for a strong v1
- one reproducible insured-bank panel
- one reproducible parent monitor
- one validated rule-aware headroom engine
- one clean 2020 event study notebook / script
- one documentation set that lets a new contributor extend the repo quickly
