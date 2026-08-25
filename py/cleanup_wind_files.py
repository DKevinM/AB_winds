"""Delete HRDPS wind files older than RETENTION_DAYS from the Supabase
'winds' storage bucket and the wind_files table that indexes them.

Why this exists: ab_met_pull.py has uploaded every HRDPS pull since
2026-03-25 with no cleanup at all - 49 GB / 7,182 files as of 2026-08-25,
growing ~630 MB/day. Nothing reads files older than a few days in the
normal case (AB_winds/odour/wind_loader.py's get_nearest_wind_record()
just wants the snapshot nearest to "now"), but the odour back-trajectory
tool *does* support checking a past date (a real request seen 2026-08-24
looked back 6 days), so the cutoff is a rolling month, not a rolling week,
to keep that working for realistic complaint-investigation lookback.

Retention is keyed off the file's own date (the year/month/day columns,
i.e. what the forecast run was FOR), not created_at (when it was
uploaded) - upload_historical_winds.py backfilled some old dates well
after the fact, so created_at doesn't reliably reflect data age.
"""
import os
from datetime import datetime, timedelta, timezone

import requests

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
HDR = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
BUCKET = "winds"
RETENTION_DAYS = 30


def distinct_days():
    days = set()
    offset, page = 0, 1000
    while True:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/wind_files",
            headers=HDR,
            params={"select": "year,month,day", "limit": page, "offset": offset},
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for row in rows:
            days.add((row["year"], row["month"], row["day"]))
        offset += page
    return days


def delete_day(y, m, d):
    prefix = f"hrdps/{y}/{m:02d}/{d:02d}"
    r = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/list/{BUCKET}",
        headers={**HDR, "Content-Type": "application/json"},
        json={"prefix": prefix, "limit": 1000, "offset": 0},
    )
    r.raise_for_status()
    items = r.json()
    paths = [f"{prefix}/{it['name']}" for it in items] if isinstance(items, list) else []
    freed = sum((it.get("metadata") or {}).get("size", 0) for it in items) if isinstance(items, list) else 0

    if paths:
        r = requests.delete(
            f"{SUPABASE_URL}/storage/v1/object/{BUCKET}",
            headers={**HDR, "Content-Type": "application/json"},
            json={"prefixes": paths},
        )
        r.raise_for_status()

    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/wind_files",
        headers=HDR,
        params={"year": f"eq.{y}", "month": f"eq.{m}", "day": f"eq.{d}"},
    )
    r.raise_for_status()

    return len(paths), freed


def main():
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    old_days = [
        (y, m, d) for (y, m, d) in distinct_days()
        if datetime(y, m, d, tzinfo=timezone.utc) < cutoff - timedelta(days=1)
    ]

    total_files = total_bytes = 0
    for (y, m, d) in sorted(old_days):
        n, freed = delete_day(y, m, d)
        total_files += n
        total_bytes += freed
        print(f"{y}-{m:02d}-{d:02d}: deleted {n} files ({freed/1024/1024:.1f} MB)")

    print(f"Done. Deleted {total_files} files, freed {total_bytes/1024/1024/1024:.2f} GB "
          f"(retention: {RETENTION_DAYS} days)")


if __name__ == "__main__":
    raise SystemExit(main())
