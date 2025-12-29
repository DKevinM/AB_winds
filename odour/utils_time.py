from datetime import datetime, timezone, timedelta

def floor_to_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)

def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def hour_key(dt: datetime) -> str:
    # consistent key for caching / file naming / dict lookup
    dt = ensure_utc(dt)
    return dt.strftime("%Y%m%d%H")  # e.g., 2025122506
