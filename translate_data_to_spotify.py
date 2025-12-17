import csv
import os
import re
import time
import shutil
import subprocess
from pathlib import Path
from typing import Any, Tuple

from TrainingData.data.scraper_auth import *

import numpy as np
import requests
from scipy.io import wavfile
from scipy.signal import stft


SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

# --- Token Manager Class ---
class SpotifyTokenManager:
    """A class to manage Spotify API tokens, handling automatic refreshing."""
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expiry_time = 0
        self.refresh_token()

    def get_token(self) -> str:
        """Returns a valid token, refreshing if necessary."""
        # Refresh 5 minutes before actual expiry to be safe
        if time.time() > self.token_expiry_time - 300:
            print("Spotify token expired. Refreshing...")
            self.refresh_token()
        return self.token

    def refresh_token(self):
        """Fetches a new token from the Spotify API."""
        r = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        # Set expiry time based on 'expires_in' field (usually 3600 seconds)
        self.token_expiry_time = time.time() + data["expires_in"]
        print("Successfully refreshed Spotify token.")


# Utility
def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:120] if s else "unknown"

def clean_track_name(track_name: str) -> str:
    """Removes common extraneous text from track names."""
    track_name = re.sub(r"\s*\(.*?\)\s*", "", track_name)
    track_name = track_name.split("-")[0]
    return track_name.strip()

def spotify_get(url: str, token_manager: SpotifyTokenManager, *, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    token = token_manager.get_token()
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=timeout,
    )
    if r.status_code == 429:
        retry_after = r.headers.get("Retry-After")
        time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2.0)
        return spotify_get(url, token_manager, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# Spotify API
def search_track(token_manager: SpotifyTokenManager, artist: str, track: str, *, market: str = "US") -> dict[str, Any] | None:
    q = f'track:"{track}" artist:"{artist}"'
    data = spotify_get(
        f"{SPOTIFY_API_BASE}/search",
        token_manager,
        params={"q": q, "type": "track", "limit": 1, "market": market},
    )
    items = (((data.get("tracks") or {}).get("items")) or [])
    return items[0] if items else None

def download_audio_from_youtube(artist: str, track: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    search_query = f"ytsearch1:{artist} - {track} audio"
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--cookies-from-browser", "chrome",
        "--extract-audio",
        "--audio-format", "m4a",
        "--audio-quality", "0",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--output", str(out_path),
        search_query,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for '{artist} - {track}': {result.stderr[:500]}")

def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH.")
    return ffmpeg

def audio_to_wav_ffmpeg(audio_path: Path, wav_path: Path, *, target_sr: int = 22050) -> None:
    ffmpeg = _ensure_ffmpeg()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(audio_path), "-ac", "1", "-ar", str(target_sr), "-vn", str(wav_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed decoding {audio_path.name}:\n{p.stderr[:800]}")

def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    fmax = min(float(fmax), float(sr) / 2.0)
    m_min, m_max = _hz_to_mel(np.array([fmin, fmax], dtype=np.float64))
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    hz_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1:m + 2]
        if center == left: center = min(left + 1, n_freqs - 1)
        if right == center: right = min(center + 1, n_freqs - 1)
        if left < center: fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if center < right: fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    enorm = 2.0 / (hz_pts[2:n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, None]
    return fb

def wav_to_log_mel_npz(wav_path: Path, out_npz_path: Path, *, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 128, fmin: float = 20.0, fmax: float = 11025.0) -> None:
    sr, x = wavfile.read(str(wav_path))
    if x.ndim != 1: x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer): x = (x.astype(np.float32) / float(np.iinfo(x.dtype).max)).clip(-1.0, 1.0)
    else: x = x.astype(np.float32)
    _f, _t, z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft, padded=False, boundary=None)
    power = np.abs(z) ** 2
    fb = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=min(fmax, sr / 2))
    mel = fb @ power
    log_mel = np.log(np.maximum(mel, 1e-10)).astype(np.float16)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz_path, log_mel=log_mel, sr=np.int32(sr), n_fft=np.int32(n_fft), hop_length=np.int32(hop_length), n_mels=np.int32(n_mels))

def translate_lastfm_to_spotify(in_csv: Path, out_csv: Path, audio_dir: Path, *, features_dir: Path, max_rows: int | None = None, sleep_sec: float = 0.10, keep_audio: bool = False) -> None:
    client_id = SPOTIFY_CLIENT_ID
    client_secret = SPOTIFY_CLIENT_SECRET
    if not client_id or not client_secret:
        raise RuntimeError("Missing Spotify client credentials.")
    
    token_manager = SpotifyTokenManager(client_id, client_secret)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(in_csv, "r", encoding="utf-8", newline="") as f_in, open(out_csv, "w", encoding="utf-8", newline="") as f_out:
        r = csv.DictReader(f_in)
        fieldnames = list(r.fieldnames or []) + ["spotify_track_id", "spotify_uri", "spotify_url", "preview_url", "matched_track_name", "matched_artist_name", "popularity", "downloaded_preview_path", "feature_path", "feature_status", "match_status"]
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()

        for i, row in enumerate(r, start=1):
            if max_rows is not None and i > max_rows: break
            artist, track = (row.get("artist_name") or "").strip(), (row.get("track_name") or "").strip()
            cleaned_track = clean_track_name(track)
            result = {**row, "spotify_track_id": "", "spotify_uri": "", "spotify_url": "", "preview_url": "", "matched_track_name": "", "matched_artist_name": "", "popularity": "", "downloaded_preview_path": "", "feature_path": "", "feature_status": "", "match_status": ""}
            if not artist or not cleaned_track:
                result["match_status"] = "skipped_empty"
                w.writerow(result)
                continue

            item = search_track(token_manager, artist, cleaned_track)
            if not item:
                result["match_status"] = "not_found"
                w.writerow(result)
                time.sleep(sleep_sec)
                continue

            result.update({
                "spotify_track_id": item.get("id", ""), "spotify_uri": item.get("uri", ""),
                "spotify_url": (item.get("external_urls", {}).get("spotify", "")),
                "preview_url": item.get("preview_url", ""), "matched_track_name": item.get("name", ""),
                "matched_artist_name": (item.get("artists", [{}])[0].get("name", "")),
                "popularity": str(item.get("popularity", ""))
            })
            track_id, track_duration_ms = result["spotify_track_id"], item.get("duration_ms")

            if track_id and track_duration_ms:
                safe_artist, safe_track = slugify(result['matched_artist_name'] or artist), slugify(result['matched_track_name'] or track)
                m4a_name = f"{safe_artist}__{safe_track}__{track_id}.m4a"
                audio_path, feature_path = audio_dir / m4a_name, features_dir / f"{track_id}.npz"
                result["feature_path"] = str(feature_path)
                try:
                    if not (feature_path.exists() and feature_path.stat().st_size > 0):
                        if not (audio_path.exists() and audio_path.stat().st_size > 0):
                            download_audio_from_youtube(result["matched_artist_name"] or artist, result["matched_track_name"] or track, audio_path)
                        result["downloaded_preview_path"] = str(audio_path)
                        tmp_wav = feature_path.with_suffix(".tmp.wav")
                        audio_to_wav_ffmpeg(audio_path, tmp_wav)
                        wav_to_log_mel_npz(tmp_wav, feature_path)
                        if tmp_wav.exists(): tmp_wav.unlink()
                        if not keep_audio and audio_path.exists(): audio_path.unlink()
                    result["feature_status"], result["match_status"] = "features_created", "matched_features_ready"
                except Exception as e:
                    print(f"Failed processing for {artist} - {track}: {e}")
                    result["feature_status"], result["match_status"] = "processing_failed", "matched_processing_failed"
            else:
                result["feature_status"], result["match_status"] = "no_duration_data", "matched_no_duration_data"
            w.writerow(result)
            time.sleep(sleep_sec)

if __name__ == "__main__":
    base = Path("TrainingData") / "data"
    genres_to_process = ["pop", "indie pop", "synthpop", "hip-hop", "electronic", "rock", "chart"]
    for tag in genres_to_process:
        print(f"\n--- Processing data for tag: {tag} ---")
        in_csv = base / f"lastfm_candidates_{tag}.csv"
        resolved_dir = base / "spotify_resolved" / tag
        out_csv, audio_dir, features_dir = resolved_dir / "resolved.csv", resolved_dir / "previews", resolved_dir / "features"
        if not in_csv.exists():
            print(f"Warning: Input file {in_csv} not found. Skipping.")
            continue
        translate_lastfm_to_spotify(in_csv,
                                    out_csv,
                                    audio_dir,
                                    features_dir=features_dir,
                                    max_rows=350,
                                    keep_audio=False,
                                    sleep_sec=0.10)
        print(f"Finished processing for {tag}. Wrote: {out_csv}")