# src/dsai/build_climatology_cache.py
#
# Refreshes the climatology cache for every watched station/parameter.
# Run daily (not hourly) - the underlying distribution barely shifts
# hour to hour, and pulling full history (10k+ rows) per station/
# parameter combo is too expensive to repeat every check cycle. The
# hourly exceedance check (check_exceedances.py) reads this cache
# instead of re-querying full history each time.

import os
from supabase import create_client

from stations import WATCH_STATIONS, PARAMETERS
from climatology import (
    fetch_station_series,
    build_climatology,
    save_cache,
    cache_key,
)

HISTORY_SINCE = "2025-01-01T00:00:00"


def main():
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

    cache = {}
    for station in WATCH_STATIONS:
        for parameter in PARAMETERS:
            series = fetch_station_series(sb, station, parameter, since_iso=HISTORY_SINCE)
            if not series:
                print(f"No data: {station} / {parameter}")
                continue
            clim = build_climatology(series)
            cache[cache_key(station, parameter)] = clim
            print(f"{station} / {parameter}: {clim['n_values']} values cached")

    save_cache(cache)
    print(f"Cache saved: {len(cache)} station/parameter combinations")


if __name__ == "__main__":
    main()
