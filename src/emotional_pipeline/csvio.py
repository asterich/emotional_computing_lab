from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _cell(row.get(field, "")) for field in fields})


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return str(value)
