#!/usr/bin/env python3
"""Upload weights/ and libero/ to ModelScope model leafsflower/stwam."""

from __future__ import annotations

import os
import sys
import time
import traceback

from modelscope.hub.api import HubApi

REPO_ID = "leafsflower/stwam"
TOKEN = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_SDK_TOKEN")
if not TOKEN:
    print("MODELSCOPE_API_TOKEN not set", file=sys.stderr)
    sys.exit(2)

IGNORE = [
    ".cache",
    ".cache/**",
    "**/.cache/**",
    "**/.cache/**/*",
    "**/download/**",
    ".ms_upload_cache",
    "**/.ms_upload_cache/**",
    "*.metadata",
]

JOBS = [
    ("/mnt/sdb/feng/stwam/libero", "libero", "Upload libero dataset"),
    ("/mnt/sdb/feng/stwam/weights", "weights", "Upload weights"),
]


def login_with_retry(max_tries: int = 8) -> HubApi:
    last: Exception | None = None
    for i in range(max_tries):
        try:
            api = HubApi(token=TOKEN)
            api.login(TOKEN)
            print(f"login ok on try {i + 1}", flush=True)
            return api
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"login try {i + 1} fail: {e}", flush=True)
            time.sleep(min(30, 2**i))
    assert last is not None
    raise last


def inventory(local: str) -> tuple[int, int]:
    nfiles = 0
    nbytes = 0
    for root, dirs, files in os.walk(local):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for fn in files:
            if fn.endswith(".metadata"):
                continue
            p = os.path.join(root, fn)
            nfiles += 1
            try:
                nbytes += os.path.getsize(p)
            except OSError:
                pass
    return nfiles, nbytes


def main() -> int:
    api = login_with_retry()

    for local, path_in_repo, msg in JOBS:
        t0 = time.time()
        print(f"\n=== START {local} -> {REPO_ID}:{path_in_repo} ===", flush=True)
        nfiles, nbytes = inventory(local)
        print(
            f"local inventory (no .cache): files={nfiles} size={nbytes / 1e9:.2f}GB",
            flush=True,
        )

        for attempt in range(1, 6):
            try:
                result = api.upload_folder(
                    repo_id=REPO_ID,
                    repo_type="model",
                    folder_path=local,
                    path_in_repo=path_in_repo,
                    commit_message=msg,
                    ignore_patterns=IGNORE,
                    token=TOKEN,
                    max_workers=2,
                    use_cache=True,
                )
                dt = time.time() - t0
                print(
                    f"=== DONE {path_in_repo} in {dt / 60:.1f} min, result={result!r} ===",
                    flush=True,
                )
                break
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                print(
                    f"=== FAIL {path_in_repo} attempt {attempt} after {dt / 60:.1f} min: "
                    f"{type(e).__name__}: {e} ===",
                    flush=True,
                )
                traceback.print_exc()
                if attempt >= 5:
                    print("giving up", flush=True)
                    return 1
                time.sleep(15)
                try:
                    api = login_with_retry()
                except Exception as le:  # noqa: BLE001
                    print("relogin failed", le, flush=True)

    print("\nALL UPLOADS COMPLETE", flush=True)
    try:
        api = login_with_retry()
        files = api.get_model_files(REPO_ID, recursive=True)
        print(f"remote files: {len(files)}", flush=True)
        total = 0
        for f in files:
            size = f.get("Size") or 0
            total += size
            p = f.get("Path") or ""
            if size > 10_000_000 or p.endswith(
                (".md", ".json", ".pt", ".safetensors", ".hdf5")
            ):
                print(f"  {p}: {size}", flush=True)
        print(f"total remote size: {total / 1e9:.2f} GB", flush=True)
    except Exception as e:  # noqa: BLE001
        print("list files failed", e, flush=True)
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
