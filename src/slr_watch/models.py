from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EntityType(str, Enum):
    BHC_PARENT = "bhc_parent"
    INSURED_BANK_SUB = "insured_bank_sub"
    IHC_FBO_US = "ihc_fbo_us"
    FOREIGN_BRANCH_AGENCY_US = "foreign_branch_agency_us"


@dataclass(frozen=True)
class EntityRegulatoryProfile:
    entity_type: EntityType
    slr_applies: bool = True
    is_gsib_parent: bool = False
    is_covered_bank_subsidiary: bool = False
    parent_method1_surcharge: Optional[float] = None
    early_adopter: bool = False
