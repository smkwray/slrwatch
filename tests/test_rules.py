from datetime import date

from slr_watch.models import EntityRegulatoryProfile, EntityType
from slr_watch.rules import RuleRegime, determine_regime, required_slr, required_slr_breakdown


def test_determine_regime_pre_2026():
    assert determine_regime(date(2025, 12, 31), early_adopter=False) == RuleRegime.PRE_2026_ESLR


def test_determine_regime_early_adoption_window_without_opt_in():
    assert determine_regime(date(2026, 1, 15), early_adopter=False) == RuleRegime.EARLY_ADOPTION_WINDOW


def test_determine_regime_post_2026_with_opt_in():
    assert determine_regime(date(2026, 1, 15), early_adopter=True) == RuleRegime.POST_2026_ESLR


def test_pre_2026_gsib_parent_requirement():
    profile = EntityRegulatoryProfile(
        entity_type=EntityType.BHC_PARENT,
        is_gsib_parent=True,
    )
    assert required_slr(date(2025, 12, 31), profile) == 0.05


def test_post_2026_gsib_parent_requirement():
    profile = EntityRegulatoryProfile(
        entity_type=EntityType.BHC_PARENT,
        is_gsib_parent=True,
        parent_method1_surcharge=0.045,
    )
    assert required_slr(date(2026, 4, 1), profile) == 0.0525


def test_post_2026_covered_sub_cap():
    profile = EntityRegulatoryProfile(
        entity_type=EntityType.INSURED_BANK_SUB,
        is_covered_bank_subsidiary=True,
        parent_method1_surcharge=0.045,
    )
    assert required_slr(date(2026, 4, 1), profile) == 0.04


def test_post_2026_covered_sub_fallback():
    profile = EntityRegulatoryProfile(
        entity_type=EntityType.INSURED_BANK_SUB,
        is_covered_bank_subsidiary=True,
        parent_method1_surcharge=None,
    )
    breakdown = required_slr_breakdown(date(2026, 4, 1), profile)
    assert breakdown.required_slr == 0.04
    assert "fallback" in breakdown.notes.lower()
