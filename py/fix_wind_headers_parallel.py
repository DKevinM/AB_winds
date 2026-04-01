import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "winds"
MAX_WORKERS = 10   # 🔥 tune this (start with 5–10)

def fetch_file(path):
    url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def upload_fixed(path, data):
    supabase.storage.from_(BUCKET).upload(
        path=path,
        file=data,
        file_options={
            "content-type": "application/json",
            "content-encoding": "gzip",
            "upsert": "true"
        }
    )


def fix_file(path):
    try:
        data = fetch_file(path)
        upload_fixed(path, data)
        return (path, "OK")
    except Exception as e:
        return (path, f"FAIL: {e}")


def main():
    # Pull file list
    resp = supabase.table("wind_files").select("file_path").execute()
    files = [row["file_path"] for row in resp.data]

    print(f"Total files: {len(files)}")

    results = []
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fix_file, f): f for f in files}

        for i, future in enumerate(as_completed(futures), 1):
            path, status = future.result()

            if "OK" in status:
                print(f"[{i}] ✔ {path}")
            else:
                print(f"[{i}] ❌ {path} → {status}")
                failures.append(path)

    print("\n--- SUMMARY ---")
    print(f"Total: {len(files)}")
    print(f"Failures: {len(failures)}")

    if failures:
        with open("failed_files.txt", "w") as f:
            for item in failures:
                f.write(item + "\n")

        print("Saved failed files to failed_files.txt")


if __name__ == "__main__":
    main()
