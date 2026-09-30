"""Read a per-file CSV report back: (summary, problem events, informational events)."""
from __future__ import annotations

import csv
from pathlib import Path

from finalpass_audiobook.output import INFO_LABEL, ISSUE_COLUMNS, PROBLEMS_LABEL


def read_report(path: Path) -> tuple[dict, list[dict], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    blank = rows.index([])
    summary = {r[0]: " ".join(c for c in r[1:] if c) for r in rows[:blank]}
    heads = [i for i, r in enumerate(rows) if r == ISSUE_COLUMNS]
    assert len(heads) == 2
    assert rows[heads[0] - 2][0] == PROBLEMS_LABEL and rows[heads[0] - 1] == [] and rows[heads[0] - 3] == []
    assert rows[heads[1] - 2][0] == INFO_LABEL and rows[heads[1] - 1] == []

    def table(i: int) -> list[dict]:
        out = []
        for r in rows[i + 1:]:
            if not r:
                break
            out.append(dict(zip(ISSUE_COLUMNS, r + [""] * (len(ISSUE_COLUMNS) - len(r)))))
        return out
    return summary, table(heads[0]), table(heads[1])
