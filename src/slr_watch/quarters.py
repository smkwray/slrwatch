from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re


_QUARTER_MONTH = {
    1: 3,
    2: 6,
    3: 9,
    4: 12,
}


@dataclass(frozen=True)
class QuarterRef:
    year: int
    quarter: int

    @classmethod
    def parse(cls, value: str) -> "QuarterRef":
        match = re.fullmatch(r"(\d{4})Q([1-4])", value.strip().upper())
        if not match:
            raise ValueError("Quarter must look like 2025Q4")
        return cls(year=int(match.group(1)), quarter=int(match.group(2)))

    @classmethod
    def from_date(cls, value: date) -> "QuarterRef":
        quarter = ((value.month - 1) // 3) + 1
        return cls(year=value.year, quarter=quarter)

    @classmethod
    def from_report_date(cls, value: str) -> "QuarterRef":
        parsed = datetime.strptime(value, "%m/%d/%Y").date()
        return cls.from_date(parsed)

    @property
    def label(self) -> str:
        return f"{self.year}Q{self.quarter}"

    @property
    def yy(self) -> str:
        return f"{self.year % 100:02d}"

    @property
    def quarter_end(self) -> date:
        month = _QUARTER_MONTH[self.quarter]
        day = 31 if month in {3, 12} else 30
        return date(self.year, month, day)

    @property
    def quarter_end_iso(self) -> str:
        return self.quarter_end.isoformat()

    @property
    def report_date_mmddyyyy(self) -> str:
        return self.quarter_end.strftime("%m/%d/%Y")

    @property
    def report_date_mmddyyyy_compact(self) -> str:
        return self.quarter_end.strftime("%m%d%Y")
