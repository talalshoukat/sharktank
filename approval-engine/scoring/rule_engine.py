"""
Rule-based scoring engine (Phase 1 MVP).
Each BRD criterion is a pure function returning (points, reason_string).
This makes unit testing trivial and reason messages self-assemble.
"""
from __future__ import annotations

from config.settings import settings
from scoring.contract_scorer import score_contract_verification


def score_establishment_law(law_category: str | None) -> tuple[int, str | None]:
    """
    BRD: Establishment law classification scoring.
    GOSI Gov / Semi-Gov = +10, GOSI Private = +5, PPA = 0.
    """
    if not law_category or law_category == "UNKNOWN":
        return 0, "Establishment law classification unavailable"

    points = settings.law_points.get(law_category.upper(), 0)

    label_map = {
        "GOSI_GOV": "Government",
        "GOSI_SEMI_GOV": "Semi-Government",
        "GOSI_PRIVATE": "Private",
        "PPA": "PPA",
    }
    label = label_map.get(law_category.upper(), law_category)
    reason = f"Establishment classified as {label} (law score: {points:+d})" if points != 0 else None
    return points, reason


def score_joining_date(backdated_months: float | None) -> tuple[int, str | None]:
    """
    BRD: Data validation — joining date backdating.
    2–4 months = +20, 4–12 months = +10, 1–2 years = +5.
    """
    if backdated_months is None:
        return 0, "Joining date not available for backdating check"

    for min_m, max_m, pts in settings.joining_date_points:
        if min_m <= backdated_months <= max_m:
            reason = (
                f"Joining date backdated {backdated_months:.1f} months "
                f"({min_m}–{max_m} months range, score: {pts:+d})"
            )
            return pts, reason if pts != 0 else None

    if backdated_months > 24:
        return -10, f"Joining date backdated {backdated_months:.1f} months (over 2 years, score: -10)"

    if backdated_months < 0:
        return -20, f"Joining date is in the future (score: -20)"

    return 0, None


def score_approval_rate_est(approval_rate: float | None) -> tuple[int, str | None]:
    """
    BRD: Establishment historical approval rate (2025).
    100–90% = +50, 90–80% = +30, 80–60% = 0, 60–40% = -10, 40–20% = -20.
    """
    if approval_rate is None:
        return 0, "No establishment approval history available"

    for low, high, pts in settings.est_approval_rate_points:
        if low <= approval_rate <= high:
            reason = (
                f"Establishment approval rate {approval_rate:.1f}% "
                f"(score: {pts:+d})"
            ) if pts != 0 else None
            return pts, reason

    return 0, None


def score_approval_rate_contributor(approval_rate: float | None) -> tuple[int, str | None]:
    """
    BRD: Contributor historical approval rate (2025).
    Same band structure as establishment.
    """
    if approval_rate is None:
        return 0, "No contributor approval history available"

    for low, high, pts in settings.contributor_approval_rate_points:
        if low <= approval_rate <= high:
            reason = (
                f"Contributor approval rate {approval_rate:.1f}% "
                f"(score: {pts:+d})"
            ) if pts != 0 else None
            return pts, reason

    return 0, None


def score_violation_history(violations_per_month: float) -> tuple[int, str | None]:
    """
    BRD: Establishment violation history.
    Penalty = violations_per_month × VIOLATION_PENALTY_PER_MONTH (configurable).
    """
    if violations_per_month <= 0:
        return 0, None

    penalty = -int(round(violations_per_month * settings.violation_penalty_per_month))
    reason = (
        f"Violation history detected: {violations_per_month:.1f} violations/month "
        f"(score: {penalty})"
    )
    return penalty, reason


def score_kashif(kashif_score: float | None) -> tuple[int, str | None]:
    """
    KASHIF fraud indicator (EDA showed significant correlation with rejection).
    Lower COMPOUND_VALUE_1 → lower risk → positive contribution.
    We normalise: 0 risk = +20, max risk = -20 (linear mapping).
    Adjust thresholds as EDA refines the model.
    """
    if kashif_score is None:
        return 0, "Fraud risk indicator unavailable"

    # Assume score range 0–100 where higher = more fraudulent
    # Map linearly: 0→+20, 50→0, 100→-20
    points = int(round(20 - (kashif_score / 100) * 40))
    points = max(-20, min(20, points))

    if points > 0:
        return points, f"Low fraud risk indicator (score: {points:+d})"
    elif points < 0:
        return points, f"Elevated fraud risk indicator (score: {points})"
    return 0, None


def score_nin_presence(nin_present: bool) -> tuple[int, str | None]:
    """
    BRD: Completeness check — NIN (National ID Number) presence.
    Present = minor positive signal.
    """
    if nin_present:
        return 5, None
    return -5, "National ID Number (NIN) not present in submission"


def compute_rule_score(features: dict) -> tuple[int, list[str], dict]:
    """
    Compute the total rule-based score for a transaction.

    Returns:
        total_points: raw point sum (not normalised to 0-100 yet)
        reasons: list of reason strings (only non-None, non-zero-impact reasons)
        breakdown: dict of criterion → points
    """
    scorers = [
        ("establishment_law",      score_establishment_law(features.get("law_category"))),
        ("joining_date",           score_joining_date(features.get("backdated_months"))),
        ("est_approval_rate",      score_approval_rate_est(features.get("est_approval_rate"))),
        ("contributor_approval",   score_approval_rate_contributor(features.get("contributor_approval_rate"))),
        ("violation_history",      score_violation_history(features.get("violation_count_per_month", 0.0))),
        ("kashif_score",           score_kashif(features.get("kashif_score"))),
        ("nin_presence",           score_nin_presence(features.get("nin_present", False))),
        ("contract_verification",  score_contract_verification(features.get("contract_id"))),
    ]

    total = settings.base_score
    reasons = []
    breakdown = {"base": settings.base_score}

    for name, (pts, reason) in scorers:
        total += pts
        breakdown[name] = pts
        if reason:
            reasons.append(reason)

    return total, reasons, breakdown
