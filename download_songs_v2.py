"""
download_songs_v2.py — Download top 10,000 songs from top_10000_songs.csv via YouTube.

Reads top_10000_songs.csv (columns: rank, track_name, artists, popularity, ...),
searches YouTube for each "track_name artists", and saves to songs/ as MP3 128 kbps.

Usage:
    python download_songs_v2.py

Requirements:
    pip install yt-dlp pandas
    ffmpeg (brew install ffmpeg or apt install ffmpeg)

Output:
    songs/<sanitized_title>.mp3 — target 10,000 files (~5 MB each).
"""

import os
import re
import sys
import time
import shutil
import random

import pandas as pd
import yt_dlp

# ─── Config ────────────────────────────────────────────────────────────────────

CSV_PATH = "top_10000_songs.csv"
SONGS_DIR = "songs"
ARCHIVE_FILE = "downloaded_ids_v2.txt"
TARGET_COUNT = 10_000
MP3_BITRATE = "128"

# Anti-bot / rate-limit mitigation (YouTube may return 403 without cookies/throttling)
COOKIES_BROWSER = os.environ.get("TIMBRE_COOKIES_BROWSER", "chrome").strip().lower()
SLEEP_BETWEEN_SONGS_MIN_S = float(os.environ.get("TIMBRE_SLEEP_MIN_S", "1.5"))
SLEEP_BETWEEN_SONGS_MAX_S = float(os.environ.get("TIMBRE_SLEEP_MAX_S", "4.0"))
MAX_403_RETRIES = int(os.environ.get("TIMBRE_MAX_403_RETRIES", "6"))
BACKOFF_BASE_S = float(os.environ.get("TIMBRE_BACKOFF_BASE_S", "15"))

# ─── Helpers ────────────────────────────────────────────────────────────────────


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install: brew install ffmpeg  (or apt install ffmpeg)")
        sys.exit(1)


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """Make a string safe for use as a filename."""
    s = re.sub(r'[<>:"/\\|?*]', "", name)
    s = s.strip().strip(".") or "unknown"
    return s[:max_len]


# ─── Progress state (shared with hooks) ─────────────────────────────────────────

class RunState:
    """Shared across all downloads for accurate total ETA."""
    def __init__(self):
        self.start_time = time.time()
        self.done_count = 0
        self.total = 0

    def eta_remaining(self) -> str:
        if self.done_count <= 0:
            return "?"
        elapsed = time.time() - self.start_time
        avg = elapsed / self.done_count
        left = max(0, self.total - self.done_count)
        secs = int(avg * left)
        if secs >= 3600:
            return f"~{secs // 3600}h {(secs % 3600) // 60}m"
        if secs >= 60:
            return f"~{secs // 60}m {secs % 60}s"
        return f"~{secs}s"


class ProgressState:
    def __init__(self, current: int, total: int, track_name: str, popularity: int, run_state: RunState):
        self.current = current
        self.total = total
        self.track_name = track_name
        self.popularity = popularity
        self.phase = "idle"
        self.eta_str = "?"
        self.run_state = run_state

    def eta_remaining(self) -> str:
        return self.run_state.eta_remaining()


def make_progress_hook(state: ProgressState):
    def progress_hook(d: dict) -> None:
        if d["status"] == "downloading":
            state.phase = "downloading"
            state.eta_str = d.get("_eta_str") or "?"
            downloaded = d.get("_downloaded_bytes_str", "?")
            total = d.get("_total_bytes_str") or d.get("_total_bytes_estimate_str", "?")
            speed = d.get("_speed_str", "?")
            print(
                f"\r  [{state.current}/{state.total}] {state.track_name[:50]:<50} "
                f"popularity={state.popularity}  ⬇ Downloading  {downloaded}/{total}  {speed}  ETA {state.eta_str}  ",
                end="",
                flush=True,
            )
        elif d["status"] == "finished":
            state.phase = "converting"
            print(
                f"\r  [{state.current}/{state.total}] {state.track_name[:50]:<50} "
                f"popularity={state.popularity}  🔄 Converting to MP3  ETA total: {state.eta_remaining()}   ",
                end="",
                flush=True,
            )
    return progress_hook


def make_postprocessor_hook(state: ProgressState):
    def postprocessor_hook(d: dict) -> None:
        if d["status"] == "started":
            state.phase = "converting"
            print(
                f"\r  [{state.current}/{state.total}] {state.track_name[:50]:<50} "
                f"popularity={state.popularity}  🔄 Converting to MP3  ETA total: {state.eta_remaining()}   ",
                end="",
                flush=True,
            )
        elif d["status"] == "finished":
            state.run_state.done_count += 1
            print(
                f"\r  [{state.current}/{state.total}] {state.track_name[:50]:<50} "
                f"popularity={state.popularity}  ✅ Done  (total ETA: {state.eta_remaining()})                    ",
                flush=True,
            )
    return postprocessor_hook


def skip_if_too_long(info, *, incomplete):
    """Skip live streams and videos longer than 10 minutes."""
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        return "Skipping — live stream or upcoming"
    duration = info.get("duration")
    if duration and duration > 600:
        return f"Skipping — too long ({duration // 60:.0f} min)"
    return None


def download_one(
    index: int,
    total: int,
    track_name: str,
    artists: str,
    popularity: int,
    songs_dir: str,
    archive_file: str,
    run_state: RunState,
) -> bool:
    """Download one song by searching YouTube. Returns True if a new file was added."""
    state = ProgressState(index, total, track_name, popularity, run_state)
    safe_title = sanitize_filename(f"{track_name} - {artists}".strip(" -"))
    outtmpl = os.path.join(songs_dir, f"{safe_title}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": MP3_BITRATE,
        }],
        "outtmpl": outtmpl,
        "download_archive": archive_file,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "socket_timeout": 30,
        "retries": 3,
        "extractor_retries": 2,
        # Slow down requests to reduce 403s
        "sleep_interval": 1,
        "max_sleep_interval": 3,
        "progress_hooks": [make_progress_hook(state)],
        "postprocessor_hooks": [make_postprocessor_hook(state)],
        "match_filter": skip_if_too_long,
    }

    query = f"{track_name} {artists}".strip()
    search_url = f"ytsearch1:{query}"

    before = set()
    if os.path.isdir(songs_dir):
        before = set(f for f in os.listdir(songs_dir) if f.endswith(".mp3"))

    # Cookies from browser help a lot with 403s (Chrome/Safari supported depending on local setup).
    cookies_enabled = False
    if COOKIES_BROWSER:
        try:
            ydl_opts["cookiesfrombrowser"] = (COOKIES_BROWSER,)
            cookies_enabled = True
        except Exception:
            cookies_enabled = False

    def _is_403(err: Exception) -> bool:
        s = str(err)
        return ("HTTP Error 403" in s) or ("403" in s and "Forbidden" in s)

    attempts = 0
    while True:
        try:
            state.phase = "fetching"
            cookie_note = f" cookies={COOKIES_BROWSER}" if cookies_enabled else ""
            print(
                f"  [{state.current}/{state.total}] {track_name[:50]:<50} "
                f"popularity={popularity}  📄 Fetching info{cookie_note}  ETA total: {state.eta_remaining()}",
                flush=True,
            )
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([search_url])
            break
        except Exception as e:
            attempts += 1
            if _is_403(e) and attempts <= MAX_403_RETRIES:
                backoff = BACKOFF_BASE_S * (2 ** (attempts - 1))
                jitter = random.uniform(0, min(10.0, backoff * 0.1))
                wait_s = backoff + jitter
                print(
                    f"\r  [{state.current}/{state.total}] {track_name[:50]:<50} "
                    f"popularity={popularity}  ⚠️  403 blocked (attempt {attempts}/{MAX_403_RETRIES}) — "
                    f"sleeping {wait_s:.0f}s then retry",
                    flush=True,
                )
                time.sleep(wait_s)
                continue

            print(
                f"\r  [{state.current}/{state.total}] {track_name[:50]:<50} "
                f"popularity={popularity}  ❌ Error: {e}",
                flush=True,
            )
            return False

    if os.path.isdir(songs_dir):
        after = set(f for f in os.listdir(songs_dir) if f.endswith(".mp3"))
        return len(after) > len(before)
    return False


def main() -> None:
    check_ffmpeg()

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    total = min(len(df), TARGET_COUNT)
    df = df.head(total)

    os.makedirs(SONGS_DIR, exist_ok=True)

    # Optional: archive file so we skip already-downloaded video IDs
    archive_path = os.path.join(os.path.dirname(CSV_PATH) or ".", ARCHIVE_FILE)

    print(f"Source:          {CSV_PATH}")
    print(f"Songs to process: {total}")
    print(f"Output dir:      {SONGS_DIR}/")
    print(f"Format:         MP3 @ {MP3_BITRATE} kbps (~5 MB per song)")
    print(f"Archive (resume): {archive_path}")
    if COOKIES_BROWSER:
        print(f"Cookies:         from browser = {COOKIES_BROWSER}  (set TIMBRE_COOKIES_BROWSER to change)")
    print(f"Throttle:        sleep {SLEEP_BETWEEN_SONGS_MIN_S:.1f}–{SLEEP_BETWEEN_SONGS_MAX_S:.1f}s between songs")
    print()

    start = time.time()
    ok = 0
    run_state = RunState()
    run_state.total = total
    for i, row in df.iterrows():
        idx = i + 1
        track_name = str(row.get("track_name", "")).strip()
        artists = str(row.get("artists", "")).strip()
        popularity = int(row.get("popularity", 0))
        if not track_name:
            continue
        if download_one(
            idx,
            total,
            track_name,
            artists,
            popularity,
            SONGS_DIR,
            archive_path,
            run_state,
        ):
            ok += 1
        time.sleep(random.uniform(SLEEP_BETWEEN_SONGS_MIN_S, SLEEP_BETWEEN_SONGS_MAX_S))

    elapsed = time.time() - start
    count = sum(1 for f in os.listdir(SONGS_DIR) if f.endswith(".mp3")) if os.path.isdir(SONGS_DIR) else 0
    print()
    print("=" * 60)
    print("Done.")
    print(f"  New this run : {ok}")
    print(f"  Total in dir : {count}")
    print(f"  Time         : {elapsed / 60:.1f} min")
    print()
    print("Next steps:")
    print("  python build_library.py")
    print("  python extract_features.py")
    print("  python get_song_emotions.py")


if __name__ == "__main__":
    main()
