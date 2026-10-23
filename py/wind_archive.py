"""HRDPS wind archive: last KEEP_DAYS on this server, everything in Dropbox.

AB_winds/met_data grows ~0.6 GB/day and old HRDPS runs can't be
downloaded again (dd.weather.gc.ca keeps ~1 day, HPFX ~50 days), so the
full history is kept in the encrypted offsite backup instead
(dropbox-crypt:, see scripts/backup_offsite_dropbox.sh) and only recent
months stay on disk (Kevin, 2026-10-23).

  prune    (cron, weekly) - files older than KEEP_DAYS are checked against
           Dropbox with `rclone cryptcheck` (content, not just names); only
           matching files are listed in ARCHIVE_INDEX and deleted locally.
  restore  (on demand)    - met_store.LocalMetStore calls ensure_window()
           before loading, which copies any archived files for the run's
           hours back into met_data (~8 s per day of data). Restored files
           are old enough to be pruned again on the next weekly run.

ARCHIVE_INDEX lists the archived file names, one per line, so the
historical server can still offer dates that are only in Dropbox.

Usage: wind_archive.py prune [--apply]
       wind_archive.py restore 2026-08-15T06:00 2026-08-15T18:00
"""

import argparse
import datetime as dt
import fcntl
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET_DIR = os.path.join(ROOT, "met_data")
ARCHIVE_INDEX = os.path.join(MET_DIR, "archived_files.txt")
LOCK = "/opt/airquality/locks/wind_archive.lock"
REMOTE = "dropbox-crypt:current/airquality/github/AB_winds/met_data"
RCLONE = "/usr/local/bin/rclone"
KEEP_DAYS = 92
_FN = re.compile(r"ab_met_(\d{8})_(\d{2})Z_f(\d{3})\.json\.gz$")


def valid_time(name):
    m = _FN.search(name)
    if not m:
        return None
    run = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    return run + dt.timedelta(hours=int(m.group(3)))


def archived_names():
    if not os.path.exists(ARCHIVE_INDEX):
        return set()
    with open(ARCHIVE_INDEX) as f:
        return {line.strip() for line in f if line.strip()}


def archived_first_valid():
    times = [t for t in map(valid_time, archived_names()) if t]
    return min(times) if times else None


def _locked():
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    fh = open(LOCK, "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def _files_from(names):
    tf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    tf.write("\n".join(sorted(names)) + "\n")
    tf.close()
    return tf.name


def ensure_window(t0, t1, met_dir=MET_DIR, pad_hours=6):
    """Copy archived files with valid times in [t0 - pad, t1 + pad] back
    into met_dir. Returns how many files were restored."""
    lo, hi = t0 - dt.timedelta(hours=pad_hours), t1 + dt.timedelta(hours=pad_hours)
    want = {n for n in archived_names() if (v := valid_time(n)) and lo <= v <= hi
            and not os.path.exists(os.path.join(met_dir, n))}
    if not want:
        return 0
    with _locked():
        want = {n for n in want if not os.path.exists(os.path.join(met_dir, n))}  # another run may have fetched them
        if not want:
            return 0
        lst = _files_from(want)
        try:
            subprocess.run([RCLONE, "copy", REMOTE, met_dir, "--files-from", lst, "--transfers", "8",
                            "--no-traverse", "-q"], check=True, timeout=1800)
        finally:
            os.unlink(lst)
    print(f"wind_archive: restored {len(want)} file(s) from Dropbox for {lo:%Y-%m-%d %H}Z-{hi:%Y-%m-%d %H}Z",
          file=sys.stderr)
    return len(want)


def prune(apply=False, keep_days=KEEP_DAYS):
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    old = sorted(n for n in os.listdir(MET_DIR) if (v := valid_time(n)) and v < cutoff)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print(f"{stamp} prune: {len(old)} local file(s) older than {cutoff:%Y-%m-%d} (keep {keep_days} days)")
    if not old:
        return 0
    with _locked():
        lst = _files_from(old)
        try:
            # cryptcheck encrypts each local file with the remote's nonce and compares
            # checksums, so a pass means Dropbox really holds this exact content.
            r = subprocess.run([RCLONE, "cryptcheck", MET_DIR, REMOTE, "--files-from", lst, "--one-way",
                                "--checkers", "8", "--combined", "-", "-q"],
                               capture_output=True, text=True, timeout=7200)
        finally:
            os.unlink(lst)
        ok = {line[2:] for line in r.stdout.splitlines() if line.startswith("= ")}
        bad = [n for n in old if n not in ok]
        if bad:
            print(f"  {len(bad)} file(s) NOT confirmed in Dropbox - kept locally, e.g. {bad[:3]}")
        size = sum(os.path.getsize(os.path.join(MET_DIR, n)) for n in ok)
        print(f"  {len(ok)} file(s) confirmed in Dropbox ({size / 1e9:.1f} GB)")
        if not apply:
            print("  dry run - nothing deleted (use --apply)")
            return 0
        with open(ARCHIVE_INDEX, "w") as f:
            f.write("\n".join(sorted(archived_names() | ok)) + "\n")
        for n in ok:
            os.remove(os.path.join(MET_DIR, n))
    print(f"  deleted {len(ok)} local file(s), freed {size / 1e9:.1f} GB; index now lists {len(archived_names())}")
    return len(ok)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--keep-days", type=int, default=KEEP_DAYS)
    r = sub.add_parser("restore")
    r.add_argument("start")
    r.add_argument("end")
    a = ap.parse_args()
    if a.cmd == "prune":
        prune(a.apply, a.keep_days)
    else:
        to_utc = lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)
        print(ensure_window(to_utc(a.start), to_utc(a.end), pad_hours=0), "file(s) restored")


if __name__ == "__main__":
    main()
