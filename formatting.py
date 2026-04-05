from __future__ import annotations


def clean_text(value, default: str = "-") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def format_record(wins: int, losses: int) -> str:
    return f"{wins}-{losses}"


def safe_pct(numerator: float, denominator: float, digits: int = 1) -> float:
    if not denominator:
        return 0.0
    return round((numerator / denominator) * 100, digits)


def signed(value: float, digits: int = 2) -> str:
    return f"+{value:.{digits}f}" if value >= 0 else f"{value:.{digits}f}"


def stoplight(value: float, neutral_band: float = 0.5) -> str:
    if value > neutral_band:
        return "🟢 Green"
    if value < -neutral_band:
        return "🔴 Red"
    return "🟡 Yellow"
