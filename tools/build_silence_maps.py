# tools/build_silence_maps.py
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np


DEFAULT_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac"}
DEFAULT_BATCH_SIZE = 10
DEFAULT_FRAME_MS = 20
DEFAULT_THRESHOLD = 0.015
DEFAULT_MIN_SILENCE_MS = 900
DEFAULT_MERGE_GAP_MS = 180
DEFAULT_KEEP_LEAD_MS = 60
DEFAULT_KEEP_TAIL_MS = 80
DEFAULT_MIN_SKIP_MS = 350
DEFAULT_MAX_SKIP_MS = 15000
DEFAULT_ALGO_VERSION = "sm-v1"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH_BYTES = 2

SOURCE_CONFIG = {
    "current": {
        "identifier": "vorhofflimmern-bei-bekannter-khk-dr-oemer-dr-remzi-09.05.25",
        "src": "S1",
        "base_dir": "S1",
    },
    "fspneu": {
        "identifier": "FSPneu",
        "src": "S2",
        "base_dir": "S2",
    },
}


@dataclass
class SilenceSegment:
    start: float
    end: float


@dataclass
class SilenceMap:
    id: str
    name: str
    title: str
    src: str
    url: str
    duration: float
    sample_rate: int
    threshold: float
    frame_ms: int
    min_silence_ms: int
    merge_gap_ms: int
    keep_lead_ms: int
    keep_tail_ms: int
    min_skip_ms: int
    max_skip_ms: int
    algo_version: str
    analyzed_at: str
    segments: List[SilenceSegment]


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def chunked(seq: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def http_get_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "silence-map-builder/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_download(url: str, dst: Path, timeout: int = 180) -> None:
    req = Request(url, headers={"User-Agent": "silence-map-builder/1.0"})
    with urlopen(req, timeout=timeout) as resp, dst.open("wb") as f:
        shutil.copyfileobj(resp, f)


def build_archive_download_url(identifier: str, name: str) -> str:
    parts = [quote(p) for p in name.split("/")]
    return f"https://archive.org/download/{quote(identifier)}/{'/'.join(parts)}"


def decode_uri_component_safe(value: str) -> str:
    try:
        from urllib.parse import unquote
        return unquote(value)
    except Exception:
        return value


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        return {
            "version": DEFAULT_ALGO_VERSION,
            "updated_at": None,
            "items": {},
        }
    return json.loads(index_path.read_text(encoding="utf-8"))


def save_index(index_path: Path, index_data: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_data["updated_at"] = utc_now_iso()
    index_path.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_to_wav(src: Path, dst_wav: Path, sample_rate: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-ac", str(DEFAULT_CHANNELS),
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(dst_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")


def read_wav_mono_pcm16(wav_path: Path) -> Tuple[np.ndarray, int, float]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        if channels != 1:
            raise ValueError(f"expected mono wav, got {channels} channels")
        if sampwidth != DEFAULT_WIDTH_BYTES:
            raise ValueError(f"expected 16-bit pcm wav, got {sampwidth * 8}-bit")
        raw = wf.readframes(nframes)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
    return audio, sample_rate, duration


def rms_of_frame(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))


def detect_silence_segments(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float,
    frame_ms: int,
    min_silence_ms: int,
    merge_gap_ms: int,
    keep_lead_ms: int,
    keep_tail_ms: int,
    min_skip_ms: int,
    max_skip_ms: int,
) -> List[SilenceSegment]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    frame_duration = frame_size / float(sample_rate)

    silent_ranges: List[Tuple[float, float]] = []
    current_start: Optional[float] = None

    total_frames = math.ceil(len(samples) / frame_size)
    for i in range(total_frames):
        start_idx = i * frame_size
        end_idx = min(len(samples), start_idx + frame_size)
        frame = samples[start_idx:end_idx]
        t0 = start_idx / float(sample_rate)
        t1 = end_idx / float(sample_rate)
        silent = rms_of_frame(frame) < threshold

        if silent:
            if current_start is None:
                current_start = t0
        else:
            if current_start is not None:
                silent_ranges.append((current_start, t1))
                current_start = None

    if current_start is not None:
        silent_ranges.append((current_start, len(samples) / float(sample_rate)))

    min_silence_sec = min_silence_ms / 1000.0
    merge_gap_sec = merge_gap_ms / 1000.0
    keep_lead_sec = keep_lead_ms / 1000.0
    keep_tail_sec = keep_tail_ms / 1000.0
    min_skip_sec = min_skip_ms / 1000.0
    max_skip_sec = max_skip_ms / 1000.0
    total_duration = len(samples) / float(sample_rate)

    filtered = [(a, b) for a, b in silent_ranges if (b - a) >= min_silence_sec]
    if not filtered:
        return []

    merged: List[Tuple[float, float]] = []
    cur_a, cur_b = filtered[0]
    for a, b in filtered[1:]:
        if a - cur_b <= merge_gap_sec:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))

    out: List[SilenceSegment] = []
    for a, b in merged:
        start = max(0.0, a + keep_lead_sec)
        end = min(total_duration, b - keep_tail_sec)
        length = end - start
        if length < min_skip_sec:
            continue
        if length > max_skip_sec:
            end = start + max_skip_sec
            length = end - start
        if length < min_skip_sec:
            continue
        out.append(SilenceSegment(start=round(start, 3), end=round(end, 3)))

    return out


def sanitize_file_name(name: str) -> str:
    return name.replace("/", "__")


def map_file_path(output_root: Path, base_dir: str, item_id: str, name: str) -> Path:
    safe = sanitize_file_name(name)
    stem = Path(safe).stem
    return output_root / base_dir / f"{stem}.json"


def fetch_archive_audio_items(identifier: str, src: str) -> List[dict]:
    meta_url = f"https://archive.org/metadata/{quote(identifier)}"
    payload = http_get_json(meta_url)
    files = payload.get("files", [])
    items: List[dict] = []

    for idx, f in enumerate(files):
        name = str(f.get("name", ""))
        if not name:
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in DEFAULT_AUDIO_EXTENSIONS:
            continue
        if "source" in f and str(f.get("source", "")).lower() != "original":
            continue

        title = str(f.get("title", "")).strip() or decode_uri_component_safe(Path(name).name)
        url = build_archive_download_url(identifier, name)
        item_id = f"{src}|{name}"

        items.append({
            "id": item_id,
            "name": name,
            "title": title,
            "url": url,
            "src": src,
        })

    items.sort(key=lambda x: x["name"].lower())
    return items


def is_item_already_done(index_data: dict, item_id: str, algo_version: str) -> bool:
    info = index_data.get("items", {}).get(item_id)
    if not info:
        return False
    return info.get("algo_version") == algo_version and info.get("status") == "done"


def write_silence_map_json(
    output_path: Path,
    silence_map: SilenceMap,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(silence_map)
    payload["segments"] = [asdict(seg) for seg in silence_map.segments]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_one_item(
    item: dict,
    output_root: Path,
    base_dir: str,
    threshold: float,
    frame_ms: int,
    min_silence_ms: int,
    merge_gap_ms: int,
    keep_lead_ms: int,
    keep_tail_ms: int,
    min_skip_ms: int,
    max_skip_ms: int,
    algo_version: str,
    sample_rate: int,
) -> Tuple[SilenceMap, Path]:
    with tempfile.TemporaryDirectory(prefix="silence-map-") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        src_audio = tmp_dir / Path(item["name"]).name
        wav_file = tmp_dir / "audio.wav"

        http_download(item["url"], src_audio)
        convert_to_wav(src_audio, wav_file, sample_rate)
        samples, actual_sr, duration = read_wav_mono_pcm16(wav_file)

        segments = detect_silence_segments(
            samples=samples,
            sample_rate=actual_sr,
            threshold=threshold,
            frame_ms=frame_ms,
            min_silence_ms=min_silence_ms,
            merge_gap_ms=merge_gap_ms,
            keep_lead_ms=keep_lead_ms,
            keep_tail_ms=keep_tail_ms,
            min_skip_ms=min_skip_ms,
            max_skip_ms=max_skip_ms,
        )

        silence_map = SilenceMap(
            id=item["id"],
            name=item["name"],
            title=item["title"],
            src=item["src"],
            url=item["url"],
            duration=round(duration, 3),
            sample_rate=actual_sr,
            threshold=threshold,
            frame_ms=frame_ms,
            min_silence_ms=min_silence_ms,
            merge_gap_ms=merge_gap_ms,
            keep_lead_ms=keep_lead_ms,
            keep_tail_ms=keep_tail_ms,
            min_skip_ms=min_skip_ms,
            max_skip_ms=max_skip_ms,
            algo_version=algo_version,
            analyzed_at=utc_now_iso(),
            segments=segments,
        )

        out_path = map_file_path(output_root, base_dir, item["id"], item["name"])
        return silence_map, out_path


def build_candidate_list(
    all_items: List[dict],
    index_data: dict,
    algo_version: str,
    only_new: bool,
) -> List[dict]:
    if not only_new:
        return all_items
    return [item for item in all_items if not is_item_already_done(index_data, item["id"], algo_version)]


def update_index_success(
    index_data: dict,
    item: dict,
    out_path: Path,
    algo_version: str,
    silence_map: SilenceMap,
) -> None:
    index_data.setdefault("items", {})
    index_data["items"][item["id"]] = {
        "name": item["name"],
        "title": item["title"],
        "src": item["src"],
        "path": str(out_path).replace("\\", "/"),
        "algo_version": algo_version,
        "status": "done",
        "duration": silence_map.duration,
        "segment_count": len(silence_map.segments),
        "analyzed_at": silence_map.analyzed_at,
    }


def update_index_error(
    index_data: dict,
    item: dict,
    algo_version: str,
    error_message: str,
) -> None:
    index_data.setdefault("items", {})
    index_data["items"][item["id"]] = {
        "name": item["name"],
        "title": item["title"],
        "src": item["src"],
        "path": None,
        "algo_version": algo_version,
        "status": "error",
        "error": error_message,
        "analyzed_at": utc_now_iso(),
    }


def run_once(args: argparse.Namespace) -> int:
    if not ffmpeg_exists():
        print("ffmpeg bulunamadı. Lütfen ffmpeg kur ve PATH'e ekle.", file=sys.stderr)
        return 2

    if args.source not in SOURCE_CONFIG:
        print(f"Geçersiz source: {args.source}", file=sys.stderr)
        return 2

    source_cfg = SOURCE_CONFIG[args.source]
    output_root = Path(args.output).resolve()
    index_path = output_root / "index.json"
    index_data = load_index(index_path)

    try:
        all_items = fetch_archive_audio_items(source_cfg["identifier"], source_cfg["src"])
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"Metadata okunamadı: {exc}", file=sys.stderr)
        return 1

    candidates = build_candidate_list(
        all_items=all_items,
        index_data=index_data,
        algo_version=args.algo_version,
        only_new=args.only_new,
    )

    print(f"Toplam audio: {len(all_items)}")
    print(f"İşlenecek aday: {len(candidates)}")
    if not candidates:
        save_index(index_path, index_data)
        print("Yeni aday yok.")
        return 0

    ok_count = 0
    err_count = 0

    for batch_no, batch in enumerate(chunked(candidates, args.batch_size), start=1):
        print(f"\nBatch {batch_no}: {len(batch)} dosya")
        for item in batch:
            print(f"  -> {item['name']}")
            try:
                silence_map, out_path = process_one_item(
                    item=item,
                    output_root=output_root,
                    base_dir=source_cfg["base_dir"],
                    threshold=args.threshold,
                    frame_ms=args.frame_ms,
                    min_silence_ms=args.min_silence_ms,
                    merge_gap_ms=args.merge_gap_ms,
                    keep_lead_ms=args.keep_lead_ms,
                    keep_tail_ms=args.keep_tail_ms,
                    min_skip_ms=args.min_skip_ms,
                    max_skip_ms=args.max_skip_ms,
                    algo_version=args.algo_version,
                    sample_rate=args.sample_rate,
                )
                write_silence_map_json(out_path, silence_map)
                update_index_success(index_data, item, out_path.relative_to(output_root), args.algo_version, silence_map)
                ok_count += 1
                print(f"     ok | segment={len(silence_map.segments)} | duration={silence_map.duration:.1f}s")
            except Exception as exc:
                update_index_error(index_data, item, args.algo_version, str(exc))
                err_count += 1
                print(f"     err | {exc}", file=sys.stderr)

        save_index(index_path, index_data)

    print(f"\nBitti. başarılı={ok_count} hata={err_count}")
    return 0 if err_count == 0 else 1


def run_loop(args: argparse.Namespace) -> int:
    print(f"Timer modu başladı. interval={args.interval_sec}s")
    while True:
        code = run_once(args)
        if code not in (0, 1):
            return code
        print(f"Bekleniyor: {args.interval_sec}s\n")
        time.sleep(args.interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper'sız silence map batch builder")
    parser.add_argument("--source", choices=sorted(SOURCE_CONFIG.keys()), required=True)
    parser.add_argument("--output", default="silence-maps")
    parser.add_argument("--only-new", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--min-silence-ms", type=int, default=DEFAULT_MIN_SILENCE_MS)
    parser.add_argument("--merge-gap-ms", type=int, default=DEFAULT_MERGE_GAP_MS)
    parser.add_argument("--keep-lead-ms", type=int, default=DEFAULT_KEEP_LEAD_MS)
    parser.add_argument("--keep-tail-ms", type=int, default=DEFAULT_KEEP_TAIL_MS)
    parser.add_argument("--min-skip-ms", type=int, default=DEFAULT_MIN_SKIP_MS)
    parser.add_argument("--max-skip-ms", type=int, default=DEFAULT_MAX_SKIP_MS)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--algo-version", default=DEFAULT_ALGO_VERSION)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=6 * 60 * 60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.watch:
      return run_loop(args)
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
