import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from TrainingData.data.scraper_auth import api_key_last_fm


LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"


class LastFMApiError(RuntimeError):
    """Raised for non-OK Last.fm responses, rate limit exhaustion, or invalid payloads."""


@dataclass(frozen=True)
class TrackCandidate:
    artist_name: str
    track_name: str
    rank: int | None = None
    playcount: int | None = None
    source: str | None = None
    page: int | None = None


class LastFMClient:
    def __init__(
        self,
        api_key: str,
        user_agent: str,
        timeout_sec: int = 30,
        max_retries: int = 7,
    ):
        if not api_key:
            raise ValueError("Missing Last.fm API key")
        if not user_agent:
            raise ValueError("Missing USER_AGENT (set environment variable USER_AGENT)")

        self.api_key = api_key
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({"user-agent": self.user_agent})

    def _sleep_backoff(self, attempt: int, retry_after_header: str | None) -> None:
        if retry_after_header and retry_after_header.isdigit():
            time.sleep(float(retry_after_header))
            return
        # Exponential backoff with jitter, capped
        sleep_s = min(60.0, (2**attempt) + random.uniform(0.0, 0.75))
        time.sleep(sleep_s)

    def get(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["api_key"] = self.api_key
        payload["format"] = "json"

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                r = self.session.get(
                    LASTFM_API_URL,
                    params=payload,
                    timeout=self.timeout_sec,
                )

                # Rate limiting
                if r.status_code == 429:
                    self._sleep_backoff(attempt, r.headers.get("Retry-After"))
                    continue

                # Transient server errors: retry
                if 500 <= r.status_code <= 599:
                    self._sleep_backoff(attempt, r.headers.get("Retry-After"))
                    continue

                # Non-success: raise with context
                if not (200 <= r.status_code < 300):
                    body_preview = (r.text or "")[:400]
                    raise LastFMApiError(
                        f"Last.fm HTTP {r.status_code} for {r.url}\n"
                        f"Content-Type: {r.headers.get('Content-Type')}\n"
                        f"Body preview: {body_preview}"
                    )

                # Guard against empty / non-JSON bodies before parsing
                if not r.content:
                    raise LastFMApiError(
                        f"Last.fm returned empty response body for {r.url} (HTTP {r.status_code})."
                    )

                content_type = (r.headers.get("Content-Type") or "").lower()
                if "json" not in content_type:
                    body_preview = (r.text or "")[:400]
                    raise LastFMApiError(
                        f"Expected JSON but got Content-Type={content_type!r} for {r.url}\n"
                        f"Body preview: {body_preview}"
                    )

                try:
                    data = r.json()
                except ValueError as e:
                    body_preview = (r.text or "")[:400]
                    raise LastFMApiError(
                        f"Invalid JSON from Last.fm for {r.url}\n"
                        f"Body preview: {body_preview}"
                    ) from e

                # Last.fm errors are JSON: { "error": ..., "message": ... }
                if isinstance(data, dict) and "error" in data:
                    raise LastFMApiError(
                        f"Last.fm API error {data.get('error')}: {data.get('message')}"
                    )

                if not isinstance(data, dict):
                    raise LastFMApiError(f"Unexpected JSON type from Last.fm: {type(data)!r}")

                return data

            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                self._sleep_backoff(attempt, None)
                continue
            except LastFMApiError as e:
                # For non-retryable API errors, fail fast
                raise

        if last_error is not None:
            raise LastFMApiError(
                f"Failed to call Last.fm after {self.max_retries} retries due to network issues."
            ) from last_error

        raise LastFMApiError(f"Failed to call Last.fm after {self.max_retries} retries (likely rate limiting).")

    def iter_chart_top_tracks(
        self,
        pages: int = 10,
        limit: int = 200,
        sleep_sec: float = 0.25,
    ) -> Iterable[TrackCandidate]:
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

    def iter_tag_top_tracks(
        self,
        tag: str,
        pages: int = 10,
        limit: int = 200,
        sleep_sec: float = 0.25,
    ) -> Iterable[TrackCandidate]:
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


def main() -> None:
    user_agent = os.environ.get("USER_AGENT") or "spotifyrecommender-dev/1.0"
    client = LastFMClient(api_key=api_key_last_fm, user_agent=user_agent)

    genres_to_collect = ["rock", "alternative", "punk", "metal"]

    for tag in genres_to_collect:
        print(f"\n--- Collecting data for tag: {tag} ---")
        candidates = client.iter_tag_top_tracks(tag=tag, pages=30, limit=200)
        out_csv = os.path.join("TrainingData", "data", f"lastfm_candidates_{tag}.csv")
        n = write_candidates_csv(candidates, out_csv)
        print(f"Wrote {n} candidates to {out_csv}")


if __name__ == "__main__":
    main()