"""
requirements/unit_normalizer.py
===============================
Deterministic linear unit normalization for architectural measurements.

All linear dimensions in the system are canonicalized to metres (m).
Supported input units:
  - Metres (m, metre, meter, metres, meters)
  - Centimetres (cm, centimetre, centimeter, centimetres, centimeters)
  - Millimetres (mm, millimetre, millimeter, millimetres, millimeters)
  - Feet (ft, foot, feet, ')
  - Inches (in, inch, inches, ")

Deterministic conversion factors to metres:
  - 1 m  = 1.0 m
  - 1 cm = 0.01 m      (e.g. 250 cm   -> 2.5 m)
  - 1 mm = 0.001 m     (e.g. 200 mm   -> 0.2 m)
  - 1 ft = 0.3048 m    (e.g. 10 ft    -> 3.048 m)
  - 1 in = 0.0254 m    (e.g. 12 inches -> 0.3048 m)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

CANONICAL_STORAGE_UNIT: str = "m"

# Unit synonyms mapped to canonical abbreviation
UNIT_MAP: Dict[str, str] = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "'": "ft",
    "in": "in",
    "inch": "in",
    "inches": "in",
    '"': "in",
}

# Multipliers to convert to metres
TO_METRES: Dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "in": 0.0254,
}


def canonicalize_unit_name(unit: str) -> Optional[str]:
    """
    Return the standard abbreviation ('m', 'cm', 'mm', 'ft', 'in') or None.
    """
    if not unit:
        return None
    clean = unit.strip().lower()
    return UNIT_MAP.get(clean)


def normalize_to_metres(value: float, unit: str) -> float:
    """
    Convert a numeric dimension from the specified unit to metres.

    Parameters
    ----------
    value : float
        The measured value.
    unit : str
        The unit of the measurement (e.g. 'mm', 'cm', 'm', 'ft', 'in').

    Returns
    -------
    float
        The length in canonical metres, rounded to 4 decimal places.

    Raises
    ------
    ValueError
        If the unit is not supported.
    """
    canon = canonicalize_unit_name(unit)
    if canon is None or canon not in TO_METRES:
        raise ValueError(
            f"Unsupported unit: '{unit}'. Supported units are: m, cm, mm, ft, in."
        )
    return round(value * TO_METRES[canon], 4)


def detect_unit_in_text(text: str) -> Tuple[Optional[str], bool]:
    """
    Detect explicit unit mentions in free-form text.

    Returns
    -------
    (detected_unit, is_ambiguous)
        detected_unit: canonical abbreviation or None
        is_ambiguous: True if multiple conflicting units are detected
    """
    if not text:
        return None, False

    found_units: set[str] = set()

    # Pattern for units with word boundaries or measurement syntax
    patterns = [
        (r"\b(millimet(?:er|re)s?|mm)\b", "mm"),
        (r"\b(centimet(?:er|re)s?|cm)\b", "cm"),
        (r"\b(met(?:er|re)s?|m)\b", "m"),
        (r"\b(feet|foot|ft|')\b", "ft"),
        (r"\b(inches|inch|in|\")\b", "in"),
    ]

    for pat, canonical in patterns:
        if re.search(pat, text, re.IGNORECASE):
            found_units.add(canonical)

    if len(found_units) == 1:
        return next(iter(found_units)), False
    if len(found_units) > 1:
        # Multiple units mentioned — potential ambiguity unless sub-measurements
        return None, True
    return None, False


def extract_dimensions_with_units(text: str) -> List[Tuple[float, str]]:
    """
    Extract (value, unit) pairs from text such as '200 mm', '2.5 m', '10 ft', '12 inches'.
    """
    results: List[Tuple[float, str]] = []
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(millimet(?:er|re)s?|mm|centimet(?:er|re)s?|cm|met(?:er|re)s?|m|feet|foot|ft|inches|inch|in|'|\")\b",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        val = float(m.group(1))
        unit = m.group(2)
        canon = canonicalize_unit_name(unit)
        if canon:
            results.append((val, canon))
    return results
