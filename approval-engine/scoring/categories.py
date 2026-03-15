"""
Maps a final score (0–100) to a BRD category label and display colour.
"""
from __future__ import annotations


# BRD category thresholds (lower bound inclusive)
CATEGORIES = [
    (80, "High",               "green"),
    (60, "Moderate",           "yellow"),
    (40, "Risky",              "orange"),
    (20, "High Rejection Risk","red"),
    (0,  "Very Likely Rejection", "dark_red"),
]


def get_category(score: float) -> tuple[str, str]:
    """
    Returns (label, colour) for a given score.

    >>> get_category(85)
    ('High', 'green')
    >>> get_category(55)
    ('Moderate', 'yellow')
    >>> get_category(10)
    ('Very Likely Rejection', 'dark_red')
    """
    for threshold, label, colour in CATEGORIES:
        if score >= threshold:
            return label, colour
    return "Very Likely Rejection", "dark_red"
