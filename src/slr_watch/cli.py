from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analytics.headroom_panel import enrich_with_headroom
from .analytics.constraint_decomposition import run_constraint_decomposition_report
from .analytics.market_context import run_market_context_report
from .analytics.parent_transmission import run_parent_transmission_report
from .analytics.policy_regime_panel import run_policy_regime_panel_report
from .analytics.reallocation_2020 import run_reallocation_report
from .analytics.safe_asset_absorption import run_absorption_report
from .analytics.treasury_intermediation import run_treasury_intermediation_report
from .config import data_path, derived_data_path, raw_data_path, reference_data_path, repo_root, reports_path, staging_data_path
from .ingest.call_reports import (
    download_call_report_bulk_zip,
    stage_call_report_folder,
    stage_call_report_zip,
    write_merged_call_report_bulk_folder,
)
from .ingest.fdic_institutions import build_fdic_institutions_reference
from .ingest.fry15 import build_method1_surcharge_overlay
from .ingest.fry9c import download_fry9c, historical_chicago_fed_url, stage_fry9c_file
from .ingest.market_overlays import build_market_overlay_panel
from .ingest.nyfed_primary_dealers import build_primary_dealer_overlay
from .ingest.trace_treasury import build_trace_treasury_overlay
from .insured_banks import build_all_insured_bank_panel, ensure_insured_bank_override_template
from .panels import build_crosswalk, build_insured_bank_panel, build_parent_panel
from .quarters import QuarterRef
from .source_manifest import format_sources_for_cli
from .site_data import build_site_data


def cmd_print_sources(_: argparse.Namespace) -> int:
    print(format_sources_for_cli())
    return 0


def cmd_print_plan(_: argparse.Namespace) -> int:
    print((repo_root() / "PLAN.md").read_text(encoding="utf-8"))
    return 0


def cmd_demo_headroom(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = pd.read_csv(input_path)
    enriched = enrich_with_headroom(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return 0


def cmd_fry9c_url(args: argparse.Namespace) -> int:
    quarter = QuarterRef.parse(args.quarter)
    print(historical_chicago_fed_url(quarter))
    return 0


def cmd_merge_call_folder(args: argparse.Namespace) -> int:
    root = Path(args.root)
    output = Path(args.output)
    write_merged_call_report_bulk_folder(root=root, output_path=output)
    print(f"Wrote {output}")
    return 0


def cmd_download_call_reports(args: argparse.Namespace) -> int:
    quarter = QuarterRef.parse(args.quarter)
    output = Path(args.output) if args.output else None
    path = download_call_report_bulk_zip(quarter, output_path=output)
    print(f"Wrote {path}")
    return 0


def cmd_stage_call_reports(args: argparse.Namespace) -> int:
    quarter = QuarterRef.parse(args.quarter) if args.quarter else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.root:
        if quarter is None:
            raise ValueError("--quarter is required when staging from --root")
        merged, normalized = stage_call_report_folder(Path(args.root), quarter=quarter, output_dir=output_dir)
    elif args.zip_path:
        merged, normalized = stage_call_report_zip(Path(args.zip_path), quarter=quarter, output_dir=output_dir)
    else:
        if quarter is None:
            raise ValueError("--quarter is required when staging from the default raw ZIP")
        default_zip = raw_data_path(
            "call_reports",
            quarter.label,
            f"FFIEC-CDR-Call-Bulk-All-Schedules-{quarter.report_date_mmddyyyy_compact}.zip",
        )
        merged, normalized = stage_call_report_zip(default_zip, quarter=quarter, output_dir=output_dir)
    print(f"Wrote {merged}")
    print(f"Wrote {normalized}")
    return 0


def cmd_download_fry9c(args: argparse.Namespace) -> int:
    quarter = QuarterRef.parse(args.quarter)
    zip_path = Path(args.zip_path) if args.zip_path else None
    output = Path(args.output) if args.output else None
    path = download_fry9c(quarter, zip_path=zip_path, output_path=output)
    print(f"Wrote {path}")
    return 0


def cmd_stage_fry9c(args: argparse.Namespace) -> int:
    quarter = QuarterRef.parse(args.quarter)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    raw_output, normalized_output = stage_fry9c_file(input_path, quarter=quarter, output_dir=output_dir)
    print(f"Wrote {raw_output}")
    print(f"Wrote {normalized_output}")
    return 0


def cmd_build_crosswalk(args: argparse.Namespace) -> int:
    universe = Path(args.universe)
    output = Path(args.output) if args.output else None
    path = build_crosswalk(universe, output_path=output)
    print(f"Wrote {path}")
    return 0


def cmd_build_insured_panel(args: argparse.Namespace) -> int:
    path = build_insured_bank_panel(
        Path(args.staged_path),
        Path(args.crosswalk),
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote {path}")
    return 0


def cmd_build_all_insured_panel(args: argparse.Namespace) -> int:
    override_path = Path(args.overrides) if args.overrides else None
    if override_path is not None:
        ensure_insured_bank_override_template(override_path)
    written = build_all_insured_bank_panel(
        Path(args.staged_path),
        crosswalk_path=Path(args.crosswalk) if args.crosswalk else None,
        overrides_path=override_path,
        fdic_metadata_path=Path(args.fdic_metadata) if args.fdic_metadata else None,
        output_path=Path(args.output) if args.output else None,
        universe_output_path=Path(args.universe_output) if args.universe_output else None,
        coverage_output_path=Path(args.coverage_output) if args.coverage_output else None,
        manifest_output_path=Path(args.manifest_output) if args.manifest_output else None,
    )
    for path in written.values():
        print(f"Wrote {path}")
    return 0


def cmd_build_fdic_institutions(args: argparse.Namespace) -> int:
    path = build_fdic_institutions_reference(output_path=Path(args.output) if args.output else None)
    print(f"Wrote {path}")
    return 0


def cmd_build_parent_panel(args: argparse.Namespace) -> int:
    path = build_parent_panel(
        Path(args.staged_path),
        Path(args.crosswalk),
        fry15_path=Path(args.fry15_path) if args.fry15_path else None,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote {path}")
    return 0


def cmd_build_fry15_overlay(args: argparse.Namespace) -> int:
    path = build_method1_surcharge_overlay(
        Path(args.input) if args.input else None,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote {path}")
    return 0


def cmd_build_primary_dealer_overlay(args: argparse.Namespace) -> int:
    path = build_primary_dealer_overlay(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote {path}")
    return 0


def cmd_build_trace_overlay(args: argparse.Namespace) -> int:
    path = build_trace_treasury_overlay(
        Path(args.input),
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote {path}")
    return 0


def cmd_build_market_overlay_panel(args: argparse.Namespace) -> int:
    path = build_market_overlay_panel(
        primary_dealer_path=Path(args.primary_dealer_path) if args.primary_dealer_path else None,
        trace_path=Path(args.trace_path) if args.trace_path else None,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Wrote {path}")
    return 0


def cmd_run_event_study(args: argparse.Namespace) -> int:
    from .analytics.event_2020 import run_event_2020

    output_dir = Path(args.output_dir) if args.output_dir else None
    market_panel_path = Path(args.market_panel) if args.market_panel else None
    path = run_event_2020(Path(args.panel), output_dir=output_dir, market_panel_path=market_panel_path)
    print(f"Wrote outputs under {path}")
    return 0


def cmd_run_market_report(args: argparse.Namespace) -> int:
    path = run_market_context_report(
        panel_path=Path(args.panel) if args.panel else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Wrote outputs under {path}")
    return 0


def cmd_run_reallocation_report(args: argparse.Namespace) -> int:
    path = run_reallocation_report(
        panel_path=Path(args.panel) if args.panel else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Wrote outputs under {path}")
    return 0


def cmd_run_parent_transmission_report(args: argparse.Namespace) -> int:
    path = run_parent_transmission_report(
        bank_panel_path=Path(args.bank_panel),
        parent_panel_path=Path(args.parent_panel),
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote outputs under {path}")
    return 0


def cmd_run_absorption_report(args: argparse.Namespace) -> int:
    path = run_absorption_report(
        panel_path=Path(args.panel) if args.panel else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Wrote outputs under {path}")
    return 0


def cmd_run_treasury_intermediation_report(args: argparse.Namespace) -> int:
    path = run_treasury_intermediation_report(
        panel_path=Path(args.panel) if args.panel else None,
        market_panel_path=Path(args.market_panel) if args.market_panel else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Wrote outputs under {path}")
    return 0


def cmd_run_policy_regime_panel_report(args: argparse.Namespace) -> int:
    path = run_policy_regime_panel_report(
        bank_panel_path=Path(args.bank_panel) if args.bank_panel else None,
        parent_panel_path=Path(args.parent_panel) if args.parent_panel else None,
        market_panel_path=Path(args.market_panel) if args.market_panel else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Wrote outputs under {path}")
    return 0


def cmd_run_constraint_decomposition_report(args: argparse.Namespace) -> int:
    path = run_constraint_decomposition_report(
        bank_panel_path=Path(args.bank_panel) if args.bank_panel else None,
        parent_panel_path=Path(args.parent_panel) if args.parent_panel else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Wrote outputs under {path}")
    return 0


def cmd_build_site_data(args: argparse.Namespace) -> int:
    written = build_site_data(
        reports_root=Path(args.reports_root) if args.reports_root else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    for path in written:
        print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slr-watch")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("print-sources", help="Print the source manifest")
    p.set_defaults(func=cmd_print_sources)

    p = sub.add_parser("print-plan", help="Print PLAN.md")
    p.set_defaults(func=cmd_print_plan)

    p = sub.add_parser("demo-headroom", help="Run the synthetic headroom demo")
    p.add_argument("--input", default=str(data_path("sample", "headroom_demo_input.csv")))
    p.add_argument("--output", default=str(data_path("sample", "headroom_demo_output.csv")))
    p.set_defaults(func=cmd_demo_headroom)

    p = sub.add_parser("fry9c-url", help="Build a Chicago Fed historical FR Y-9C URL")
    p.add_argument("--quarter", required=True, help="Quarter like 2020Q1")
    p.set_defaults(func=cmd_fry9c_url)

    p = sub.add_parser("merge-call-folder", help="Merge an extracted FFIEC call-report bulk schedule folder")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_merge_call_folder)

    p = sub.add_parser("download-call-reports", help="Download one FFIEC Call Report bulk ZIP")
    p.add_argument("--quarter", required=True, help="Quarter like 2025Q4")
    p.add_argument("--output")
    p.set_defaults(func=cmd_download_call_reports)

    p = sub.add_parser("stage-call-reports", help="Extract, merge, and normalize one Call Report quarter")
    p.add_argument("--quarter", help="Quarter like 2025Q4")
    p.add_argument("--zip-path")
    p.add_argument("--root")
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_stage_call_reports)

    p = sub.add_parser("download-fry9c", help="Download one FR Y-9C quarter")
    p.add_argument("--quarter", required=True, help="Quarter like 2021Q2")
    p.add_argument("--zip-path", help="Use a manually downloaded NIC ZIP instead of browser automation")
    p.add_argument("--output")
    p.set_defaults(func=cmd_download_fry9c)

    p = sub.add_parser("stage-fry9c", help="Normalize one FR Y-9C CSV or ZIP")
    p.add_argument("--quarter", required=True, help="Quarter like 2021Q2")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_stage_fry9c)

    p = sub.add_parser("build-crosswalk", help="Validate and materialize the curated v1 universe/crosswalk")
    p.add_argument("--universe", default=str(reference_data_path("v1_universe.csv")))
    p.add_argument("--output")
    p.set_defaults(func=cmd_build_crosswalk)

    p = sub.add_parser("build-insured-panel", help="Build the insured-bank panel from staged Call Reports")
    p.add_argument("--staged-path", default=str(data_path("staging", "call_reports")))
    p.add_argument("--crosswalk", default=str(derived_data_path("crosswalk_v1.parquet")))
    p.add_argument("--output")
    p.set_defaults(func=cmd_build_insured_panel)

    p = sub.add_parser("build-all-insured-panel", help="Build the full insured-bank descriptive panel and coverage artifacts")
    p.add_argument("--staged-path", default=str(data_path("staging", "call_reports")))
    p.add_argument("--crosswalk", default=str(derived_data_path("crosswalk_v1.parquet")))
    p.add_argument("--overrides", default=str(reference_data_path("insured_bank_metadata_overrides.csv")))
    p.add_argument("--fdic-metadata", default=str(derived_data_path("fdic_institutions.csv")))
    p.add_argument("--output", default=str(derived_data_path("insured_bank_descriptive_panel.parquet")))
    p.add_argument("--universe-output", default=str(derived_data_path("insured_bank_universe.csv")))
    p.add_argument("--coverage-output", default=str(derived_data_path("insured_bank_coverage_by_quarter.csv")))
    p.add_argument("--manifest-output", default=str(derived_data_path("insured_bank_sample_manifest.csv")))
    p.set_defaults(func=cmd_build_all_insured_panel)

    p = sub.add_parser("build-parent-panel", help="Build the parent and IHC panel from staged FR Y-9C")
    p.add_argument("--staged-path", default=str(data_path("staging", "fry9c")))
    p.add_argument("--crosswalk", default=str(derived_data_path("crosswalk_v1.parquet")))
    p.add_argument("--fry15-path")
    p.add_argument("--output")
    p.set_defaults(func=cmd_build_parent_panel)

    p = sub.add_parser("build-fry15-overlay", help="Build the FR Y-15 method 1 surcharge overlay")
    p.add_argument("--input", help="Optional OFR Basel Scores workbook path")
    p.add_argument("--output")
    p.set_defaults(func=cmd_build_fry15_overlay)

    p = sub.add_parser("build-fdic-institutions", help="Fetch the FDIC institutions metadata reference")
    p.add_argument("--output", default=str(derived_data_path("fdic_institutions.csv")))
    p.set_defaults(func=cmd_build_fdic_institutions)

    p = sub.add_parser("build-primary-dealer-overlay", help="Build the NY Fed primary dealer quarterly market overlay")
    p.add_argument("--start-date", default="2019-01-01")
    p.add_argument("--end-date")
    p.add_argument("--output")
    p.set_defaults(func=cmd_build_primary_dealer_overlay)

    p = sub.add_parser("build-trace-overlay", help="Build a quarterly TRACE Treasury overlay from manual CSV/XLSX input")
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_build_trace_overlay)

    p = sub.add_parser("build-market-overlay-panel", help="Merge quarterly market overlays into one panel")
    p.add_argument("--primary-dealer-path", default=str(derived_data_path("nyfed_primary_dealer_overlay.parquet")))
    p.add_argument("--trace-path", default=str(derived_data_path("trace_treasury_overlay.parquet")))
    p.add_argument("--output")
    p.set_defaults(func=cmd_build_market_overlay_panel)

    p = sub.add_parser("run-event-study", help="Run the 2020 temporary exclusion event study")
    p.add_argument("--panel", default=str(derived_data_path("insured_bank_descriptive_panel.parquet")))
    p.add_argument("--market-panel", default=str(derived_data_path("market_overlay_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("event_2020")))
    p.set_defaults(func=cmd_run_event_study)

    p = sub.add_parser("run-market-report", help="Build a compact quarterly Treasury market-context report")
    p.add_argument("--panel", default=str(derived_data_path("market_overlay_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("market_context")))
    p.set_defaults(func=cmd_run_market_report)

    p = sub.add_parser("run-reallocation-report", help="Build a 2020 balance-sheet reallocation report")
    p.add_argument("--panel", default=str(derived_data_path("insured_bank_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("reallocation_2020")))
    p.set_defaults(func=cmd_run_reallocation_report)

    p = sub.add_parser("run-parent-transmission-report", help="Build a linked insured-bank vs parent transmission report")
    p.add_argument("--bank-panel", default=str(derived_data_path("insured_bank_panel.parquet")))
    p.add_argument("--parent-panel", default=str(derived_data_path("parent_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("parent_transmission")))
    p.set_defaults(func=cmd_run_parent_transmission_report)

    p = sub.add_parser("run-safe-asset-absorption-report", help="Build a Treasury vs reserve absorption report")
    p.add_argument("--panel", default=str(derived_data_path("insured_bank_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("safe_asset_absorption")))
    p.set_defaults(func=cmd_run_absorption_report)

    p = sub.add_parser("run-treasury-intermediation-report", help="Build a Treasury intermediation sensitivity report")
    p.add_argument("--panel", default=str(derived_data_path("insured_bank_panel.parquet")))
    p.add_argument("--market-panel", default=str(derived_data_path("market_overlay_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("treasury_intermediation")))
    p.set_defaults(func=cmd_run_treasury_intermediation_report)

    p = sub.add_parser("run-policy-regime-panel-report", help="Build a broader policy-regime summary panel")
    p.add_argument("--bank-panel", default=str(derived_data_path("insured_bank_panel.parquet")))
    p.add_argument("--parent-panel", default=str(derived_data_path("parent_panel.parquet")))
    p.add_argument("--market-panel", default=str(derived_data_path("market_overlay_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("policy_regime_panel")))
    p.set_defaults(func=cmd_run_policy_regime_panel_report)

    p = sub.add_parser("run-constraint-decomposition-report", help="Build the first bank-constraint decomposition report")
    p.add_argument("--bank-panel", default=str(derived_data_path("insured_bank_panel.parquet")))
    p.add_argument("--parent-panel", default=str(derived_data_path("parent_panel.parquet")))
    p.add_argument("--output-dir", default=str(reports_path("constraint_decomposition")))
    p.set_defaults(func=cmd_run_constraint_decomposition_report)

    p = sub.add_parser("build-site-data", help="Build static-site JSON payloads from report outputs")
    p.add_argument("--reports-root", default=str(reports_path()))
    p.add_argument("--output-dir", default=str(repo_root() / "site" / "assets" / "data"))
    p.set_defaults(func=cmd_build_site_data)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
