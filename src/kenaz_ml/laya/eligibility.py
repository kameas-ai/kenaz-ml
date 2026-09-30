"""Per-host eligibility gate -- WP02 interim seam (WP03 implements the measurement).

Until the gate is implemented the honest answer is "not evaluated" and laya is
**not served** (WP02 T008 step 2: eligibility unknown means the kind refuses).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    eligible: bool
    reason: str
    detail: str


def current_verdict(checkpoint_key: str = "") -> Verdict:
    return Verdict(False, "not_evaluated", "the per-host eligibility gate has not been implemented yet")
