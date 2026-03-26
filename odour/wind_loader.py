import os
import requests
import gzip
import json
from datetime import datetime
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_nearest_wind_record(target_time):

    resp = supabase.table("wind_files") \
        .select("*") \
        .order("valid_time") \
        .execute()

    rows = resp.data

    best = min(rows, key=lambda r: abs(
        datetime.fromisoformat(r["valid_time"].replace("Z","")) - target_time
    ))

    return best


def download_wind(file_path):

    signed = supabase.storage.from_("winds").create_signed_url(
        file_path, 60
    )

    url = signed["signedURL"]

    r = requests.get(url)
    r.raise_for_status()

    return gzip.decompress(r.content)


def load_wind_for_time(target_time):

    record = get_nearest_wind_record(target_time)

    content = download_wind(record["file_path"])

    return json.loads(content)
