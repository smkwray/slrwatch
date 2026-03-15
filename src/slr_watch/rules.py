from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from .models import EntityRegulatoryProfile, EntityType


PRE_2026_EFFECTIVE_END = date(2025, 12, 31)
EARLY_ADOPTION_START = date(2026, 1, 1)
FINAL_RULE_EFFECTIVE = date(2026, 4, 1)


class RuleRegime(str, Enum):
    PRE_2026_ESLR = "pre_2026_eslr"
    EARLY_ADOPTION_WINDOW = "early_adoption_window"
    POST_2026_ESLR = "post_2026_eslr"


@dataclass(frozen=True)
class RequirementBreakdown:
    regime: RuleRegime
    minimum_slr: float
    leverage_buffer: float
    required_slr: float
    notes: str


def determine_regime(as_of_date: date, early_adopter: bool = False) -> RuleRegime:
    if as_of_date < EARLY_ADOPTION_START:
        return RuleRegime.PRE_2026_ESLR
    if EARLY_ADOPTION_START <= as_of_date < FINAL_RULE_EFFECTIVE:
        return (
            RuleRegime.POST_2026_ESLR
            if early_adopter
            else RuleRegime.EARLY_ADOPTION_WINDOW
        )
    return RuleRegime.POST_2026_ESLR


def _post_2026_requirement(profile: EntityRegulatoryProfile) -> RequirementBreakdown:
    minimum = 0.03

    if not profile.slr_applies:
        return RequirementBreakdown(
            regime=RuleRegime.POST_2026_ESLR,
            minimum_slr=minimum,
            leverage_buffer=0.0,
            required_slr=0.0,
            notes="SLR not applicable for this entity under current profile settings.",
        )

    surcharge = profile.parent_method1_surcharge

    if profile.is_gsib_parent:
        if surcharge is None:
            raise ValueError(
                "GSIB parent requirement under the post-2026 rule needs "
                "parent_method1_surcharge."
            )
        buffer = 0.5 * surcharge
        return RequirementBreakdown(
            regime=RuleRegime.POST_2026_ESLR,
            minimum_slr=minimum,
            leverage_buffer=buffer,
            required_slr=minimum + buffer,
            notes="Post-2026 GSIB parent rule: 3% minimum plus 50% of method 1 surcharge.",
        )

    if profile.is_covered_bank_subsidiary:
        if surcharge is None:
            # Federal Register text provides a 1% fallback where the parent does not
            # have a method 1 surcharge even though the covered-subsidiary rule applies.
            buffer = 0.01
            note = (
                "Post-2026 covered-bank rule fallback: 1% leverage buffer used because "
                "no parent method 1 surcharge was supplied."
            )
        else:
            buffer = min(0.01, 0.5 * surcharge)
            note = (
                "Post-2026 covered-bank rule: 3% minimum plus min(1%, 50% of parent "
                "method 1 surcharge)."
            )
        return RequirementBreakdown(
            regime=RuleRegime.POST_2026_ESLR,
            minimum_slr=minimum,
            leverage_buffer=buffer,
            required_slr=minimum + buffer,
            notes=note,
        )

    return RequirementBreakdown(
        regime=RuleRegime.POST_2026_ESLR,
        minimum_slr=minimum,
        leverage_buffer=0.0,
        required_slr=minimum,
        notes="Post-2026 baseline SLR minimum only.",
    )


def _pre_2026_requirement(profile: EntityRegulatoryProfile, regime: RuleRegime) -> RequirementBreakdown:
    minimum = 0.03

    if not profile.slr_applies:
        return RequirementBreakdown(
            regime=regime,
            minimum_slr=minimum,
            leverage_buffer=0.0,
            required_slr=0.0,
            notes="SLR not applicable for this entity under current profile settings.",
        )

    if profile.is_gsib_parent:
        return RequirementBreakdown(
            regime=regime,
            minimum_slr=minimum,
            leverage_buffer=0.02,
            required_slr=0.05,
            notes="Pre-2026 GSIB parent eSLR: 3% minimum plus 2% buffer.",
        )

    if profile.is_covered_bank_subsidiary:
        return RequirementBreakdown(
            regime=regime,
            minimum_slr=minimum,
            leverage_buffer=0.03,
            required_slr=0.06,
            notes="Pre-2026 covered-bank eSLR / well-capitalized threshold: 6% total.",
        )

    return RequirementBreakdown(
        regime=regime,
        minimum_slr=minimum,
        leverage_buffer=0.0,
        required_slr=minimum,
        notes="Pre-2026 baseline SLR minimum only.",
    )


def required_slr_breakdown(
    as_of_date: date,
    profile: EntityRegulatoryProfile,
) -> RequirementBreakdown:
    regime = determine_regime(as_of_date=as_of_date, early_adopter=profile.early_adopter)

    if regime is RuleRegime.POST_2026_ESLR:
        return _post_2026_requirement(profile)

    return _pre_2026_requirement(profile, regime=regime)


def required_slr(as_of_date: date, profile: EntityRegulatoryProfile) -> float:
    return required_slr_breakdown(as_of_date=as_of_date, profile=profile).required_slr
