from pathlib import Path
import os
import re
from datetime import datetime, timedelta
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_DIR = Path("met_data")


def process_file(fp):

    filename = fp.name

    m = re.search(r"ab_met_(\d{8})_(\d{2})Z_f(\d{3})", filename)
    if not m:
        print("Skipping (bad format):", filename)
        return

    ymd = m.group(1)
    hh = int(m.group(2))
    lead = int(m.group(3))

    run = datetime(
        int(ymd[:4]),
        int(ymd[4:6]),
        int(ymd[6:8]),
        hh
    )

    valid_time = run + timedelta(hours=lead)

    storage_path = f"hrdps/{valid_time.year}/{valid_time.month:02d}/{valid_time.day:02d}/{filename}"

    # Check DB
    exists = supabase.table("wind_files") \
        .select("id") \
        .eq("file_path", storage_path) \
        .limit(1) \
        .execute()

    if exists.data:
        print("Skipping existing:", filename)
        return

    print("Uploading:", filename)

    # Upload file
    with open(fp, "rb") as f:
        try:
            supabase.storage.from_("winds").upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "application/gzip"}
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

    # Insert metadata
    supabase.table("wind_files").insert({
        "model": "HRDPS",
        "run_time": run.isoformat(),
        "forecast_hour": lead,
        "valid_time": valid_time.isoformat(),
        "year": valid_time.year,
        "month": valid_time.month,
        "day": valid_time.day,
        "hour": valid_time.hour,
        "file_path": storage_path,
        "file_format": "json.gz"
    }).execute()


def main():
    files = sorted(BASE_DIR.glob("*.json.gz"))
    print(f"Found {len(files)} files")
    uploaded = 0
    skipped = 0
    for i, fp in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing {fp.name}")
        if process_file(fp):
            uploaded += 1
        else:
            skipped += 1

    print("Done.")
    print("Uploaded:", uploaded)
    print("Skipped:", skipped)

if __name__ == "__main__":
    main()
