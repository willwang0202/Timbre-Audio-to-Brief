"""
download_songs.py — Bulk song downloader for Timbre Audio-to-Brief
Downloads ~10,000 songs covering all music genres + every year 1950–2025
as 128kbps MP3 into songs/

Uses yt-dlp's built-in YouTube search — no ytmusicapi needed.

Usage:
    python3 download_songs.py

Requirements:
    python3 -m pip install yt-dlp
    brew install ffmpeg  (or apt install ffmpeg on Linux)

After downloading:
    python3 build_library.py      # regenerate song_library.csv
    python3 extract_features.py   # regenerate song_features.csv
    python3 get_song_emotions.py  # assign emotion labels
"""

import os
import sys
import time
import shutil

import yt_dlp

# ─── Configuration ────────────────────────────────────────────────────────────

SONGS_DIR = "songs"
ARCHIVE_FILE = "downloaded_ids.txt"   # yt-dlp tracks downloaded IDs here (resumable)
TARGET = 10_000                        # stop after this many new downloads
RESULTS_PER_QUERY = 15                 # YouTube search results to attempt per query
QUERY_DELAY = 4.0                      # seconds between search queries (rate limiting)

def progress_hook(d: dict) -> None:
    if d["status"] == "downloading":
        title = d.get("info_dict", {}).get("title", "…")
        downloaded = d.get("_downloaded_bytes_str", "?")
        total = d.get("_total_bytes_str") or d.get("_total_bytes_estimate_str", "?")
        speed = d.get("_speed_str", "?")
        eta = d.get("_eta_str", "?")
        print(f"\r  ⬇  {title[:55]:<55}  {downloaded}/{total}  {speed}  ETA {eta}", end="", flush=True)
    elif d["status"] == "finished":
        title = d.get("info_dict", {}).get("title", "…")
        print(f"\r  ✓  {title[:70]}", flush=True)
    elif d["status"] == "error":
        print(f"\r  ✗  error downloading", flush=True)


YDL_OPTS = {
    "format": "bestaudio/best",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "128",
    }],
    "outtmpl": f"{SONGS_DIR}/%(title)s.%(ext)s",
    "download_archive": ARCHIVE_FILE,  # skip already-downloaded videos on re-run
    "quiet": True,
    "no_warnings": True,
    "ignoreerrors": True,
    "socket_timeout": 30,
    "retries": 3,
    "extractor_retries": 2,
    "progress_hooks": [progress_hook],
}

# ─── Genre Queries ────────────────────────────────────────────────────────────

GENRE_QUERIES = [
    # ── ROCK ──────────────────────────────────────────────────────────────────
    "classic rock hits songs",
    "hard rock anthems songs",
    "soft rock ballads songs",
    "indie rock songs",
    "alternative rock 90s songs",
    "grunge rock songs",
    "punk rock songs",
    "post-punk songs",
    "new wave 80s songs",
    "garage rock songs",
    "psychedelic rock songs",
    "progressive rock songs",
    "glam rock 70s songs",
    "southern rock songs",
    "blues rock guitar songs",
    "folk rock songs",
    "shoegaze wall of sound songs",
    "emo rock songs",
    "post-rock instrumental songs",
    "britpop 90s UK songs",
    "surf rock 60s songs",
    "math rock instrumental songs",

    # ── METAL ─────────────────────────────────────────────────────────────────
    "heavy metal songs",
    "thrash metal songs",
    "death metal songs",
    "black metal songs",
    "doom metal songs",
    "power metal epic songs",
    "nu metal 2000s songs",
    "metalcore songs",
    "symphonic metal songs",
    "folk metal songs",
    "industrial metal songs",
    "stoner rock fuzz songs",

    # ── ELECTRONIC ────────────────────────────────────────────────────────────
    "house music songs",
    "deep house music songs",
    "tech house songs",
    "progressive house EDM songs",
    "techno Berlin underground songs",
    "trance music uplifting songs",
    "psytrance psychedelic songs",
    "drum and bass DnB songs",
    "dubstep songs",
    "future bass songs",
    "ambient music relaxing songs",
    "dark ambient atmospheric songs",
    "IDM intelligent dance music songs",
    "electronica downtempo songs",
    "chillwave lo-fi retro songs",
    "synthwave retro 80s songs",
    "vaporwave aesthetic music songs",
    "lo-fi hip hop beats songs",
    "electropop songs",
    "dance pop hits songs",
    "eurodance 90s songs",
    "hardstyle electronic songs",
    "UK garage 2-step songs",
    "tropical house melodic songs",
    "melodic techno songs",
    "afro house music songs",

    # ── HIP-HOP / RAP ─────────────────────────────────────────────────────────
    "golden age hip hop 90s songs",
    "boom bap hip hop songs",
    "conscious rap songs",
    "trap rap songs",
    "cloud rap atmospheric songs",
    "drill rap songs",
    "gangsta rap west coast songs",
    "East Coast hip hop songs",
    "Southern rap crunk songs",
    "lo-fi rap beats songs",
    "alternative hip hop songs",
    "jazz rap hip hop songs",
    "instrumental hip hop beats songs",
    "phonk rap dark songs",
    "trap soul R&B songs",
    "emo rap songs",

    # ── POP ───────────────────────────────────────────────────────────────────
    "pop hits top 40 songs",
    "indie pop songs",
    "art pop experimental songs",
    "synth pop 80s songs",
    "pop punk songs",
    "dream pop ethereal songs",
    "K-pop songs Korean",
    "J-pop songs Japanese",
    "C-pop Mandarin songs",
    "bedroom pop lo-fi indie songs",
    "hyperpop experimental songs",
    "city pop Japanese 80s songs",

    # ── R&B / SOUL / FUNK ─────────────────────────────────────────────────────
    "classic soul music 60s 70s songs",
    "neo soul R&B songs",
    "contemporary R&B songs",
    "new jack swing 90s songs",
    "funk music classic songs",
    "quiet storm smooth R&B songs",
    "Motown classic songs",
    "gospel music songs",
    "disco songs 70s dance",

    # ── JAZZ ──────────────────────────────────────────────────────────────────
    "bebop jazz songs",
    "cool jazz Miles Davis songs",
    "hard bop jazz songs",
    "free jazz avant-garde songs",
    "jazz fusion electric songs",
    "smooth jazz songs",
    "vocal jazz standards songs",
    "Latin jazz songs",
    "gypsy jazz Django Reinhardt songs",
    "acid jazz funk hip hop songs",
    "big band swing jazz songs",

    # ── BLUES ─────────────────────────────────────────────────────────────────
    "delta blues Mississippi songs",
    "Chicago blues electric songs",
    "Texas blues guitar songs",
    "soul blues music songs",
    "jump blues upbeat songs",

    # ── CLASSICAL ─────────────────────────────────────────────────────────────
    "baroque classical Bach Vivaldi songs",
    "classical Mozart Beethoven songs",
    "romantic classical Chopin Brahms songs",
    "minimalist classical Philip Glass songs",
    "opera arias classical songs",
    "chamber music string quartet songs",
    "piano solo classical songs",
    "orchestral symphonic music songs",
    "film score soundtrack songs",

    # ── COUNTRY / FOLK ────────────────────────────────────────────────────────
    "classic country songs Nashville",
    "outlaw country Willie Nelson songs",
    "modern country hits songs",
    "bluegrass music songs",
    "Americana roots music songs",
    "folk music acoustic songs",
    "indie folk singer songwriter songs",
    "singer songwriter acoustic songs",

    # ── LATIN ─────────────────────────────────────────────────────────────────
    "salsa music songs",
    "reggaeton hits songs",
    "bossa nova Brazilian songs",
    "samba Brazilian music songs",
    "cumbia songs Colombia",
    "Latin pop hits songs",
    "flamenco Spanish guitar songs",
    "tango Argentine songs",
    "bachata music songs",
    "Latin trap songs",
    "bolero romantic Latin songs",

    # ── REGGAE / SKA ──────────────────────────────────────────────────────────
    "roots reggae Bob Marley songs",
    "dub reggae songs",
    "dancehall reggae songs",
    "ska music songs",

    # ── WORLD MUSIC ───────────────────────────────────────────────────────────
    "afrobeat Fela Kuti Nigerian songs",
    "Afrobeats Nigerian modern songs",
    "highlife Ghana Africa songs",
    "Celtic folk Irish traditional songs",
    "Indian classical Raga sitar songs",
    "Bollywood film music songs",
    "qawwali Sufi devotional songs",
    "Arabic pop music songs",
    "Turkish pop music songs",
    "fado Portuguese melancholic songs",
    "zouk Caribbean songs",
    "calypso Caribbean songs",
    "soca music songs",
    "klezmer Jewish folk songs",
    "Balkan folk brass songs",
    "chanson French songs",
    "amapiano South African songs",
    "bhangra Punjabi songs",
    "enka traditional Japanese songs",
    "sertanejo Brazilian country songs",

    # ── EXPERIMENTAL / NEW AGE ───────────────────────────────────────────────
    "experimental avant-garde music songs",
    "neo-classical piano modern songs",
    "new age meditation music songs",
    "sleep music deep relaxation songs",
]

# ─── Decade × Style Queries ───────────────────────────────────────────────────

DECADE_QUERIES = [
    "rock and roll 1950s songs",
    "pop songs 1950s",
    "British Invasion 1960s Beatles songs",
    "Motown 1960s soul songs",
    "folk protest songs 1960s",
    "70s rock classic songs",
    "70s soul funk songs",
    "70s disco songs",
    "80s pop songs",
    "80s rock songs",
    "80s hip hop rap songs",
    "90s alternative rock songs",
    "90s hip hop songs",
    "90s R&B songs",
    "90s pop songs",
    "2000s pop songs",
    "2000s hip hop songs",
    "2000s R&B songs",
    "2010s pop songs",
    "2010s hip hop songs",
    "2020s pop songs",
    "2020s hip hop songs",
]

# ─── Year-by-Year Queries (1950–2025) ────────────────────────────────────────

YEAR_QUERIES = []
for year in range(1950, 2026):
    YEAR_QUERIES.append(f"top hits {year} songs")
    YEAR_QUERIES.append(f"best songs {year}")

# ─── All Queries Combined ─────────────────────────────────────────────────────

ALL_QUERIES = GENRE_QUERIES + DECADE_QUERIES + YEAR_QUERIES

# ─── Helpers ──────────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found.")
        print("Install with:")
        print("  macOS:  brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        sys.exit(1)


def check_disk_space() -> None:
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    if free_gb < 50:
        print(f"WARNING: Only {free_gb:.1f} GB free. ~40 GB needed for 10,000 songs.")
        print("Press Ctrl+C to cancel, or wait 10 seconds to continue...")
        time.sleep(10)


def count_songs() -> int:
    if not os.path.isdir(SONGS_DIR):
        os.makedirs(SONGS_DIR, exist_ok=True)
        return 0
    return sum(1 for f in os.listdir(SONGS_DIR) if f.endswith(".mp3"))


def count_archived() -> int:
    if not os.path.exists(ARCHIVE_FILE):
        return 0
    with open(ARCHIVE_FILE) as f:
        return sum(1 for line in f if line.strip())


def download_query(query: str, n: int) -> int:
    """Search YouTube for query, download top n results. Returns number of new files."""
    before = count_songs()
    search_url = f"ytsearch{n}:{query}"
    try:
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            ydl.download([search_url])
    except Exception as e:
        print(f"  [ERROR] {e}")
    after = count_songs()
    return after - before


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    check_ffmpeg()
    check_disk_space()

    os.makedirs(SONGS_DIR, exist_ok=True)

    print(f"Songs directory  : {SONGS_DIR}/")
    print(f"Archive file     : {ARCHIVE_FILE}  (tracks downloaded IDs for resume)")
    print(f"Already on disk  : {count_songs()} MP3 files")
    print(f"Target           : {TARGET} total downloads")
    print(f"Total queries    : {len(ALL_QUERIES)}")
    print(f"  - Genre        : {len(GENRE_QUERIES)}")
    print(f"  - Decade style : {len(DECADE_QUERIES)}")
    print(f"  - Year 1950–25 : {len(YEAR_QUERIES)}")
    print()

    total_new = 0

    for q_idx, query in enumerate(ALL_QUERIES):
        current_total = count_songs()
        if current_total >= TARGET:
            print(f"\nTarget of {TARGET} reached ({current_total} songs on disk). Stopping.")
            break

        remaining = TARGET - current_total
        n = min(RESULTS_PER_QUERY, remaining)

        print(f"[{q_idx + 1}/{len(ALL_QUERIES)}] '{query}'  (have {current_total}, need {remaining} more)")

        new_this_query = download_query(query, n)
        total_new += new_this_query

        if new_this_query:
            print(f"  => +{new_this_query} new  |  total on disk: {count_songs()}")
        else:
            print(f"  => no new files (all skipped or restricted)")

        time.sleep(QUERY_DELAY)

    final_count = count_songs()
    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"  New this run     : {total_new}")
    print(f"  Total on disk    : {final_count}")
    print(f"  Archive entries  : {count_archived()}")
    print()
    print("Next steps:")
    print("  python3 build_library.py      # update song_library.csv")
    print("  python3 extract_features.py   # update song_features.csv")
    print("  python3 get_song_emotions.py  # assign emotion labels")


if __name__ == "__main__":
    main()
