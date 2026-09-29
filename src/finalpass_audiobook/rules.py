"""Generic pause rule sets, used only to GUESS what each measured pause is.

The tool cannot see the text, so it cannot know where a heading, section break
or paragraph actually is. It measures every pause and says which rule the
length fits. Everything derived from these rules is informational.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

HEAD_TAIL_TOLERANCE_S = 0.10
BREAK_TOLERANCE_S = 0.25
HEADING_WITHIN_S = 20.0     # a heading pause is looked for this close to the start


@dataclass(frozen=True)
class RuleSet:
    name: str
    title: str
    chapter_head_s: float
    chapter_tail_s: float
    heading_s: float
    section_s: float
    paragraph_s: tuple[float, float] | None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


RULE_SETS = {
    "standard": RuleSet(
        name="standard", title="Standard: paragraph pauses 0.5-0.8 s, credits 4 s",
        chapter_head_s=1.0, chapter_tail_s=2.5, heading_s=2.5, section_s=2.5,
        paragraph_s=(0.5, 0.8),
        notes=("between chapters 3.5 s total", "after opening credits 4.0 s total",
               "before closing credits 4.0 s total"),
    ),
    "no-paragraph": RuleSet(
        name="no-paragraph", title="No paragraph rule; credits 3 s + 1 s",
        chapter_head_s=1.0, chapter_tail_s=2.5, heading_s=2.5, section_s=2.5,
        paragraph_s=None,
        notes=("3 s at the end of opening credits, then 1 s at the start of the first chapter",
               "3 s at the end of the last chapter, then 1 s at the start of closing credits"),
    ),
}


def head_or_tail_note(seconds: float, target_s: float, what: str) -> str:
    diff = round(seconds - target_s, 3)     # to the millisecond: 1.1 - 1.0 must equal the 0.1 s tolerance
    if abs(diff) <= HEAD_TAIL_TOLERANCE_S:
        return f"{what}: matches the {target_s:g} s rule"
    return f"{what}: {abs(diff):.2f} s {'longer' if diff > 0 else 'shorter'} than the {target_s:g} s rule"


def guess_internal(seconds: float, starts_at_s: float, rules: RuleSet, heading_seen: bool) -> str:
    lo_break = rules.section_s - BREAK_TOLERANCE_S
    hi_break = rules.section_s + BREAK_TOLERANCE_S
    if lo_break <= seconds <= hi_break:
        if not heading_seen and starts_at_s <= HEADING_WITHIN_S:
            return "after chapter heading (likely)"
        return "section break"
    if seconds > hi_break:
        return "longer than a section break"
    if rules.paragraph_s is None:
        return "sentence or paragraph pause"
    p_lo, p_hi = rules.paragraph_s
    if p_lo <= seconds <= p_hi:
        return "paragraph break"
    if seconds > p_hi:
        return "between paragraph and section length"
    return "shorter than a paragraph break"
