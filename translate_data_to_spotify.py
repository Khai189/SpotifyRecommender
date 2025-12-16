import csv
import os
import re
import time
from pathlib import Path
from typing import Any
from TrainingData.data.scraper_auth import *

import requests


SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:120] if s else "unknown"


def get_spotify_token(client_id: str, client_secret: str, timeout: float = 30.0) -> str:
    r = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["access_token"]


def spotify_get(url: str, token: str, *, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=timeout,
    )
    # Simple handling for rate limits
    if r.status_code == 429:
        retry_after = r.headers.get("Retry-After")
        time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2.0)
        return spotify_get(url, token, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def search_track(token: str, artist: str, track: str) -> dict[str, Any] | None:
    q = f'track:"{track}" artist:"{artist}"'
    data = spotify_get(f"{SPOTIFY_API_BASE}/search", token, params={"q": q, "type": "track", "limit": 1})
    items = (((data.get("tracks") or {}).get("items")) or [])
    return items[0] if items else None


def download_preview(preview_url: str, out_path: Path, timeout: float = 60.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(preview_url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)


def translate_lastfm_to_spotify(
    in_csv: Path,
    out_csv: Path,
    mp3_dir: Path,
    *,
    max_rows: int | None = None,
    sleep_sec: float = 0.10,
) -> None:
    client_id = SPOTIFY_CLIENT_ID
    client_secret = SPOTIFY_CLIENT_SECRET
    if not client_id or not client_secret:
        raise RuntimeError("Missing SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET environment variables.")

    token = get_spotify_token(client_id, client_secret)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(in_csv, "r", encoding="utf-8", newline="") as f_in, open(out_csv, "w", encoding="utf-8", newline="") as f_out:
        r = csv.DictReader(f_in)
        fieldnames = list(r.fieldnames or []) + [
            "spotify_track_id",
            "spotify_uri",
            "spotify_url",
            "preview_url",
            "matched_track_name",
            "matched_artist_name",
            "popularity",
            "downloaded_preview_path",
            "match_status",
        ]
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()

        for i, row in enumerate(r, start=1):
            if max_rows is not None and i > max_rows:
                break

            artist = (row.get("artist_name") or "").strip()
            track = (row.get("track_name") or "").strip()

            result = {
                **row,
                "spotify_track_id": "",
                "spotify_uri": "",
                "spotify_url": "",
                "preview_url": "",
                "matched_track_name": "",
                "matched_artist_name": "",
                "popularity": "",
                "downloaded_preview_path": "",
                "match_status": "",
            }

            if not artist or not track:
                result["match_status"] = "skipped_empty"
                w.writerow(result)
                continue

            item = search_track(token, artist, track)
            if not item:
                result["match_status"] = "not_found"
                w.writerow(result)
                time.sleep(sleep_sec)
                continue

            result["spotify_track_id"] = item.get("id") or ""
            result["spotify_uri"] = item.get("uri") or ""
            result["spotify_url"] = ((item.get("external_urls") or {}).get("spotify")) or ""
            result["preview_url"] = item.get("preview_url") or ""
            result["matched_track_name"] = item.get("name") or ""
            artists = item.get("artists") or []
            result["matched_artist_name"] = (artists[0].get("name") if artists else "") or ""
            result["popularity"] = str(item.get("popularity") or "")

            preview_url = result["preview_url"]
            if preview_url:
                filename = f"{slugify(result['matched_artist_name'] or artist)}__{slugify(result['matched_track_name'] or track)}__{result['spotify_track_id']}.mp3"
                out_path = mp3_dir / filename
                try:
                    download_preview(preview_url, out_path)
                    result["downloaded_preview_path"] = str(out_path)
                    result["match_status"] = "matched_preview_downloaded"
                except requests.RequestException:
                    result["match_status"] = "matched_preview_download_failed"
            else:
                result["match_status"] = "matched_no_preview"

            w.writerow(result)
            time.sleep(sleep_sec)


if __name__ == "__main__":
    base = Path("TrainingData") / "data"
    in_csv = base / "lastfm_candidates_pop.csv"

    resolved_dir = base / "spotify_resolved"
    out_csv = resolved_dir / "spotify_resolved_pop.csv"
    mp3_dir = base / "mp3_files" / "spotify_previews"

    translate_lastfm_to_spotify(in_csv, out_csv, mp3_dir, max_rows=100)
    print(f"Wrote: {out_csv}")