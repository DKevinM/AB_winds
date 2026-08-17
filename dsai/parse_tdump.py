# src/dsai/parse_tdump.py
#
# Parses raw HYSPLIT tdump trajectory files into the exact column
# format openair's importTraj()/trajLevel() expect (confirmed against
# openair's own Rd docs, since importTraj() itself only supports a
# fixed list of pre-built UK/international sites, not Alberta -
# Henry Pirker's trajectories have to be assembled into this shape by
# hand rather than fetched through importTraj()):
#   date, year, month, day, hour, hour.inc, lat, lon, height, pressure
# `date` is the arrival (receptor) timestamp - constant across every
# row of one trajectory, used to merge with real pollutant readings.

import csv
import glob
import os


def parse_one_tdump(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]

    n_grids = int(lines[0].split()[0])
    idx = 1 + n_grids  # skip past the met-grid header lines

    traj_header = lines[idx].split()
    n_traj = int(traj_header[0])
    idx += 1 + n_traj  # skip past the trajectory start-point lines

    idx += 1  # skip the "N_VARS LABEL..." line

    rows = []
    arrival = None
    for line in lines[idx:]:
        parts = line.split()
        if len(parts) < 12:
            continue
        yy, mm, dd, hh = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        hour_inc = float(parts[8])
        lat, lon, height, pressure = float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12])
        year_full = 2000 + yy

        if hour_inc == 0.0:
            arrival = (year_full, mm, dd, hh)

        rows.append({
            "year": year_full, "month": mm, "day": dd, "hour": hh,
            "hour.inc": hour_inc, "lat": lat, "lon": lon,
            "height": height, "pressure": pressure,
        })

    if arrival is None:
        return []

    ay, am, ad, ah = arrival
    date_str = f"{ay:04d}-{am:02d}-{ad:02d} {ah:02d}:00:00"
    for r in rows:
        r["date"] = date_str
    return rows


def parse_batch_dir(batch_dir, out_csv):
    """Parses every tdump file under batch_dir (one subdir per day) into one combined CSV."""
    all_rows = []
    tdump_files = sorted(glob.glob(os.path.join(batch_dir, "*", "tdump")))
    for path in tdump_files:
        rows = parse_one_tdump(path)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(f"No trajectory rows parsed from {batch_dir}")

    fieldnames = ["date", "year", "month", "day", "hour", "hour.inc", "lat", "lon", "height", "pressure"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(tdump_files), len(all_rows)


if __name__ == "__main__":
    import sys
    batch_dir = sys.argv[1]
    out_csv = sys.argv[2]
    n_files, n_rows = parse_batch_dir(batch_dir, out_csv)
    print(f"Parsed {n_files} trajectory files -> {n_rows} rows -> {out_csv}")
