import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from TrainingData.data.scraper_auth import api_key_last_fm
from TrainingData.data.scraper_auth import *

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from dataclasses import dataclass

LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


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

    tag = "pop"
    candidates = client.iter_tag_top_tracks(tag=tag, pages=60, limit=200)  # up to 12,000 rows
    out_csv = os.path.join("TrainingData", "data", f"lastfm_candidates_{tag}.csv")
    n = write_candidates_csv(candidates, out_csv)
    print(f"Wrote {n} candidates to {out_csv}")

    candidates = client.iter_chart_top_tracks(pages=50, limit=200)
    out_csv = os.path.join("TrainingData", "data", "lastfm_candidates_chart.csv")
    n = write_candidates_csv(candidates, out_csv)
    print(f"Wrote {n} candidates to {out_csv}")

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:120] if s else "unknown"


# Spotify API

def get_spotify_token(client_id: str, client_secret: str) -> str:
    payload = {"grant_type": "client_credentials"}
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")
    headers = {"Authorization": f"Basic {auth_b64}"}
    r = requests.post(SPOTIFY_TOKEN_URL, data=payload, headers=headers, timeout=10.0)
    r.raise_for_status()
    data = r.json()
    return data["access_token"]


def search_track(token: str, artist: str, track: str) -> dict[str, Any] | None:
    q = f"artist:{artist} track:{track}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": q, "type": "track", "limit": 3}
    r = requests.get(f"{SPOTIFY_API_BASE}/search", headers=headers, params=params, timeout=10.0)
    r.raise_for_status()
    data = r.json()
    items = data.get("tracks", {}).get("items") or []

    for item in items:
        artists = item.get("artists") or []
        artist_names = [x.get("name", "").lower() for x in artists]
        track_name = (item.get("name") or "").lower()

        if (artist.lower() in artist_names) and (track.lower() in track_name):
            return item
    return None


def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. It is required to decode Spotify preview MP3s into PCM audio "
            "to compute mel-spectrograms."
        )
    return ffmpeg


def mp3_to_wav_ffmpeg(mp3_path: Path, wav_path: Path, *, target_sr: int = 22050) -> None:
    ffmpeg = _ensure_ffmpeg()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-vn",
        str(wav_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed decoding {mp3_path.name}:\n{p.stderr[:800]}")


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    fmax = min(float(fmax), float(sr) / 2.0)

    m_min = _hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    m_max = _hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    hz_pts = _mel_to_hz(m_pts)

    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = min(left + 1, n_freqs - 1)
        if right == center:
            right = min(center + 1, n_freqs - 1)

        if left < center:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / float(center - left)
        if center < right:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / float(right - center)

    # Slaney-style normalization to make energy roughly comparable across mel bands
    enorm = 2.0 / (hz_pts[2:n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, None].astype(np.float32)
    return fb


def wav_to_log_mel_npz(
    wav_path: Path,
    out_npz_path: Path,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 11025.0,
) -> None:
    sr, x = wavfile.read(str(wav_path))

    # Ensure mono
    if x.ndim != 1:
        x = x.mean(axis=1)

    # Convert to float32 in [-1, 1]
    if np.issubdtype(x.dtype, np.integer):
        maxv = float(np.iinfo(x.dtype).max)
        x = (x.astype(np.float32) / maxv).clip(-1.0, 1.0)
    else:
        x = x.astype(np.float32)

    # STFT power spectrogram
    f, t, z = stft(
        x,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        padded=False,
        boundary=None,
    )
    power = (np.abs(z) ** 2).astype(np.float32)  # [freq, time]

    fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=min(fmax, sr / 2))
    mel = fb @ power  # [mels, time]
    mel = np.maximum(mel, 1e-10)
    log_mel = np.log(mel).astype(np.float16)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        log_mel=log_mel,
        sr=np.int32(sr),
        n_fft=np.int32(n_fft),
        hop_length=np.int32(hop_length),
        n_mels=np.int32(n_mels),
    )


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
    features_dir: Path,
    max_rows: int | None = None,
    sleep_sec: float = 0.10,
    keep_mp3: bool = False,
) -> None:
    client_id = SPOTIFY_CLIENT_ID
    client_secret = SPOTIFY_CLIENT_SECRET
    if not client_id or not client_secret:
        raise RuntimeError("Missing Spotify client credentials (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET).")

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
            "feature_path",
            "feature_status",
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
                "feature_path": "",
                "feature_status": "",
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
            track_id = result["spotify_track_id"]

            if preview_url and track_id:
                # Output paths
                mp3_name = f"{slugify(result['matched_artist_name'] or artist)}__{slugify(result['matched_track_name'] or track)}__{track_id}.mp3"
                mp3_path = mp3_dir / mp3_name
                feature_path = features_dir / f"{track_id}.npz"
                result["feature_path"] = str(feature_path)

                try:
                    # If features already exist, skip work (resumable)
                    if feature_path.exists() and feature_path.stat().st_size > 0:
                        result["feature_status"] = "features_already_exist"
                        result["match_status"] = "matched_features_ready"
                        w.writerow(result)
                        time.sleep(sleep_sec)
                        continue

                    # Download MP3 preview if missing
                    if not (mp3_path.exists() and mp3_path.stat().st_size > 0):
                        download_preview(preview_url, mp3_path)
                    result["downloaded_preview_path"] = str(mp3_path)

                    # Decode and extract mel
                    tmp_wav = feature_path.with_suffix(".tmp.wav")
                    mp3_to_wav_ffmpeg(mp3_path, tmp_wav, target_sr=22050)
                    wav_to_log_mel_npz(tmp_wav, feature_path)

                    # Cleanup temp WAV
                    try:
                        tmp_wav.unlink(missing_ok=True)
                    except Exception:
                        pass

                    # Delete MP3 unless you want to keep it
                    if not keep_mp3:
                        try:
                            mp3_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                    result["feature_status"] = "features_created"
                    result["match_status"] = "matched_features_ready"

                except requests.RequestException:
                    result["feature_status"] = "preview_download_failed"
                    result["match_status"] = "matched_preview_download_failed"
                except Exception:
                    # Keep MP3 around for debugging if something goes wrong
                    result["feature_status"] = "feature_extraction_failed"
                    result["match_status"] = "matched_feature_extraction_failed"

            else:
                result["feature_status"] = "no_preview"
                result["match_status"] = "matched_no_preview"

            w.writerow(result)
            time.sleep(sleep_sec)


if __name__ == "__main__":
    base = Path("TrainingData") / "data"
    in_csv = base / "lastfm_candidates_pop.csv"

    resolved_dir = base / "spotify_resolved" / "pop"
    out_csv = resolved_dir / "resolved.csv"
    mp3_dir = resolved_dir / "previews"
    features_dir = resolved_dir / "features"

    translate_lastfm_to_spotify(
        in_csv,
        out_csv,
        mp3_dir,
        features_dir=features_dir,
        max_rows=100,
        keep_mp3=False,
        sleep_sec=0.10,
    )
    print(f"Wrote: {out_csv}")