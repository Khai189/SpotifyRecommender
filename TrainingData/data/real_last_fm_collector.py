import csv
import os
import time
from dataclasses import dataclass
from typing import Iterable

import requests

from scraper_auth import api_key_last_fm


LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"


@dataclass(frozen=True)
class TrackCandidate:
    artist_name: str
    track_name: str
    rank: int | None = None
    playcount: int | None = None
    source: str | None = None
    page: int | None = None


class LastFMClient:
    def __init__(self, api_key: str, user_agent: str, timeout_sec: int = 30):
        if not api_key:
            raise ValueError("Missing Last.fm API key")
        if not user_agent:
            raise ValueError("Missing USER_AGENT (set environment variable USER_AGENT)")

        self.api_key = api_key
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({"user-agent": self.user_agent})

    def get(self, payload: dict) -> dict:
        payload = dict(payload)
        payload["api_key"] = self.api_key
        payload["format"] = "json"

        r = self.session.get(LASTFM_API_URL, params=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()

        # Last.fm errors come back as JSON { "error": ..., "message": ... }
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"Last.fm API error {data.get('error')}: {data.get('message')}")

        return data

    def iter_chart_top_tracks(self, pages: int = 10, limit: int = 200, sleep_sec: float = 0.25) -> Iterable[TrackCandidate]:
        """
        Collect from Last.fm global charts.
        """
        for page in range(1, pages + 1):
            data = self.get({"method": "chart.getTopTracks", "page": page, "limit": limit})
            tracks = data.get("tracks", {}).get("track", []) or []
            for i, tr in enumerate(tracks, start=1):
                artist_name = (tr.get("artist") or {}).get("name") or ""
                track_name = tr.get("name") or ""
                if artist_name and track_name:
                    playcount = tr.get("playcount")
                    yield TrackCandidate(
                        artist_name=artist_name,
                        track_name=track_name,
                        rank=i,
                        playcount=int(playcount) if str(playcount).isdigit() else None,
                        source="chart.getTopTracks",
                        page=page,
                    )
            time.sleep(sleep_sec)

    def iter_tag_top_tracks(self, tag: str, pages: int = 10, limit: int = 200, sleep_sec: float = 0.25) -> Iterable[TrackCandidate]:
        """
        Collect from a specific tag (genre-ish).
        Great when you want coverage like metal/rock/hip-hop, etc.
        """
        tag = tag.strip()
        if not tag:
            raise ValueError("tag must be non-empty")

        for page in range(1, pages + 1):
            data = self.get({"method": "tag.getTopTracks", "tag": tag, "page": page, "limit": limit})
            tracks = data.get("tracks", {}).get("track", []) or []
            for i, tr in enumerate(tracks, start=1):
                artist_name = (tr.get("artist") or {}).get("name") or ""
                track_name = tr.get("name") or ""
                if artist_name and track_name:
                    # tag.getTopTracks often returns "playcount" too, but not always
                    playcount = tr.get("playcount")
                    yield TrackCandidate(
                        artist_name=artist_name,
                        track_name=track_name,
                        rank=i,
                        playcount=int(playcount) if str(playcount).isdigit() else None,
                        source=f"tag.getTopTracks:{tag}",
                        page=page,
                    )
            time.sleep(sleep_sec)


def write_candidates_csv(candidates: Iterable[TrackCandidate], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["artist_name", "track_name", "rank", "playcount", "source", "page"],
        )
        w.writeheader()
        for c in candidates:
            w.writerow(
                {
                    "artist_name": c.artist_name,
                    "track_name": c.track_name,
                    "rank": c.rank,
                    "playcount": c.playcount,
                    "source": c.source,
                    "page": c.page,
                }
            )
            count += 1
    return count


def main():
    user_agent = os.environ.get("USER_AGENT") or "spotifyrecommender-dev/1.0"
    client = LastFMClient(api_key=api_key_last_fm, user_agent=user_agent)

    # Example 1: genre-focused collection (recommended)
    tag = "metal"
    candidates = client.iter_tag_top_tracks(tag=tag, pages=60, limit=200)  # up to 12,000 rows
    out_csv = os.path.join("TrainingData", "data", f"lastfm_candidates_{tag}.csv")
    n = write_candidates_csv(candidates, out_csv)
    print(f"Wrote {n} candidates to {out_csv}")

    # Example 2: global chart (broad)
    # candidates = client.iter_chart_top_tracks(pages=50, limit=200)
    # out_csv = os.path.join("TrainingData", "data", "lastfm_candidates_chart.csv")
    # n = write_candidates_csv(candidates, out_csv)
    # print(f"Wrote {n} candidates to {out_csv}")


if __name__ == "__main__":
    main()