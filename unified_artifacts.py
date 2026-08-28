"""On-disk take archive for Ref2VA Director Studio.

Every generated result belongs to a project, a shot, and a take.  The archive
is intentionally independent from the current ComfyUI canvas so a future
timeline UI can reopen a project, show prior takes, and continue after a
restart without guessing from "the latest mp4" in a shared output directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import subprocess
from threading import Lock, RLock
from typing import Any

import numpy as np
from PIL import Image
import torch

import folder_paths
from comfy_api.latest import Types

from .unified_project import normalise_project, safe_project_path_part


ARCHIVE_SCHEMA = 1
_PROJECT_LOCKS_GUARD = Lock()
_PROJECT_LOCKS: dict[str, RLock] = {}


def _project_lock(project_id: str) -> RLock:
    key = safe_project_path_part(project_id)
    with _PROJECT_LOCKS_GUARD:
        return _PROJECT_LOCKS.setdefault(key, RLock())


def _root(project_id: str) -> Path:
    root = Path(folder_paths.get_output_directory()) / "video" / "Ref2VA_Director" / safe_project_path_part(project_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_root(project_id: str) -> Path:
    """Public project root used by preview, export and recovery code."""
    return _root(project_id)


def trim_audio_tail(audio: Any, seconds: float):
    """Return only the audio tail matching a visual continuity window.

    Ref2VA video-reference audio uses ComfyUI's AUDIO dictionary.  Passing a
    full prior soundtrack beside only a few tail frames makes the reference
    durations disagree and can cause the next shot to repeat earlier audio.
    Unknown audio representations are deliberately omitted instead of sending
    a mismatched full clip.
    """
    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if not isinstance(waveform, torch.Tensor) or sample_rate <= 0 or waveform.ndim < 1:
        return None
    wanted = max(1, min(int(waveform.shape[-1]), round(max(0.0, float(seconds)) * sample_rate)))
    trimmed = dict(audio)
    trimmed["waveform"] = waveform[..., -wanted:].contiguous()
    return trimmed


def _shot_root(project_id: str, shot_id: str) -> Path:
    path = _root(project_id) / "shots" / safe_project_path_part(shot_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    finally:
        try:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        except OSError:
            pass


def load_project_snapshot(project_id: str) -> dict[str, Any] | None:
    path = _root(str(project_id)) / "project.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return normalise_project(json.load(handle))


def _attach_archived_takes(project: dict[str, Any]) -> int:
    """Merge completed take.json files into an in-memory project snapshot."""
    project_id = str(project["project_id"])
    recovered = 0
    for shot in project.get("shots") or []:
        take_root = _shot_root(project_id, str(shot["id"])) / "takes"
        known = {str(item.get("take_id") or item.get("id") or "") for item in shot.get("takes") or []}
        shot_recovered = 0
        for metadata_path in sorted(take_root.glob("*/take.json")) if take_root.is_dir() else []:
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                take_id = str(record.get("take_id") or "")
                expected_root = take_root / safe_project_path_part(take_id)
                if not take_id or metadata_path.parent.resolve() != expected_root.resolve() or take_id in known:
                    continue
                if str(record.get("project_id")) != project_id or str(record.get("shot_id")) != str(shot["id"]):
                    continue
                shot.setdefault("takes", []).append(record)
                known.add(take_id)
                recovered += 1
                shot_recovered += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        shot["takes"] = sorted(shot.get("takes") or [], key=lambda item: str(item.get("created_at") or ""))
        if shot["takes"]:
            selected_id = str(shot.get("selected_take_id") or "")
            available_ids = [str(item.get("take_id") or item.get("id") or "") for item in shot["takes"]]
            if shot_recovered or selected_id not in set(available_ids):
                shot["selected_take_id"] = available_ids[-1]
            if shot.get("status") in {"queued", "preparing_models", "sampling", "decoding", "upscaling", "saving", "draft"}:
                shot["status"] = "generated"
    return recovered


def _history_snapshot(project: dict[str, Any], reason: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = _root(str(project["project_id"])) / "history" / f"{stamp}-r{int(project.get('revision', 0)):06d}.json"
    archived = dict(project)
    archived["history_reason"] = str(reason or "auto_save")[:120]
    _atomic_json(path, archived)
    return path


def write_project_snapshot(project: dict[str, Any], *, reason: str = "runner_update", create_history: bool = False) -> Path:
    """Atomically save project state while retaining immutable take files."""
    project_id = str(project["project_id"])
    with _project_lock(project_id):
        current = load_project_snapshot(project_id)
        if create_history and current:
            _history_snapshot(current, reason)
        project.update(normalise_project(project))
        # Immutable take files are the source of truth for completed output.
        # Reattach them before every write so a late stale snapshot cannot
        # hide a successfully generated video from the Director UI.
        _attach_archived_takes(project)
        project["revision"] = max(int(project.get("revision", 0)), int((current or {}).get("revision", 0))) + 1
        project["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = _root(project_id) / "project.json"
        _atomic_json(path, project)
        return path


def save_editor_project(project: dict[str, Any], *, expected_revision: int | None, reason: str = "auto_save") -> dict[str, Any]:
    """Save an editor revision without allowing stale UI take data to win."""
    incoming = normalise_project(project)
    project_id = str(incoming["project_id"])
    with _project_lock(project_id):
        current = load_project_snapshot(project_id)
        if current is not None:
            _attach_archived_takes(current)
        current_revision = int((current or {}).get("revision", 0))
        if current is not None and expected_revision is not None and int(expected_revision) != current_revision:
            raise RuntimeError(f"REVISION_CONFLICT:{current_revision}")
        if current:
            current_shots = {str(item.get("id")): item for item in current.get("shots") or []}
            for shot in incoming.get("shots") or []:
                saved = current_shots.get(str(shot.get("id")))
                if not saved:
                    continue
                # Take records belong to the runner/delete endpoints, never to a
                # possibly stale browser form submission.
                shot["takes"] = saved.get("takes") or []
                shot["selected_take_id"] = saved.get("selected_take_id")
                if saved.get("status") in {"queued", "preparing_models", "sampling", "decoding", "upscaling", "saving", "generated", "failed", "stopped"}:
                    shot["status"] = saved.get("status")
            incoming["jobs"] = current.get("jobs") or incoming.get("jobs") or []
            incoming["active_job_id"] = current.get("active_job_id")
            _history_snapshot(current, reason)
        incoming["revision"] = current_revision + 1
        incoming["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = _root(project_id) / "project.json"
        _atomic_json(path, incoming)
        return incoming


def recover_archived_takes(project_id: str) -> dict[str, Any]:
    """Reattach take.json records that exist on disk but are absent in project.json."""
    with _project_lock(project_id):
        project = load_project_snapshot(project_id)
        if project is None:
            raise FileNotFoundError("项目不存在，无法恢复镜头版本。")
        recovered = _attach_archived_takes(project)
        if recovered:
            project["active_job_id"] = None
            for job in project.get("jobs") or []:
                if job.get("status") in {"queued", "preparing_models", "sampling", "decoding", "upscaling", "saving"}:
                    job["status"] = "recovered"
                    job["current_shot_id"] = None
                    job["updated_at"] = datetime.now(timezone.utc).isoformat()
            _history_snapshot(project, "before_take_recovery")
            project["revision"] = int(project.get("revision", 0)) + 1
            project["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(_root(project_id) / "project.json", project)
        project["recovered_take_count"] = recovered
        return project


def project_storage_summary(project_id: str) -> dict[str, Any]:
    """Return bounded storage statistics for one Director project."""
    project = load_project_snapshot(project_id)
    if project is None:
        raise FileNotFoundError("项目不存在。")
    root = _root(project_id).resolve()
    trash_root = (root / ".trash" / "takes").resolve()
    if root not in trash_root.parents:
        raise ValueError("视频回收区路径越界。")
    files = [path for path in trash_root.rglob("*") if path.is_file()] if trash_root.is_dir() else []
    return {
        "project_id": project_id,
        "trash_take_count": sum(1 for path in files if path.name == "take.json"),
        "trash_file_count": len(files),
        "trash_bytes": sum(path.stat().st_size for path in files),
    }


def purge_project_video_trash(project_id: str) -> dict[str, Any]:
    """Permanently clear only one project's recoverable video-take trash."""
    with _project_lock(project_id):
        project = load_project_snapshot(project_id)
        if project is None:
            raise FileNotFoundError("项目不存在。")
        if project.get("active_job_id"):
            raise RuntimeError("项目正在生成，不能清空视频回收区。")
        root = _root(project_id).resolve()
        trash_root = (root / ".trash" / "takes").resolve()
        if root not in trash_root.parents or trash_root.name != "takes" or trash_root.parent.name != ".trash":
            raise ValueError("视频回收区路径验证失败。")
        summary = project_storage_summary(project_id)
        if trash_root.is_dir():
            for child in list(trash_root.iterdir()):
                resolved = child.resolve()
                if resolved.parent != trash_root:
                    raise ValueError("视频回收区包含越界目标，已停止清理。")
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        trash_root.mkdir(parents=True, exist_ok=True)
        return {
            "project_id": project_id,
            "removed_take_count": summary["trash_take_count"],
            "removed_file_count": summary["trash_file_count"],
            "removed_bytes": summary["trash_bytes"],
            "remaining_file_count": sum(1 for path in trash_root.rglob("*") if path.is_file()),
        }


def list_project_history(project_id: str, limit: int = 30) -> list[dict[str, Any]]:
    history_root = _root(str(project_id)) / "history"
    if not history_root.is_dir():
        return []
    result = []
    for path in sorted(history_root.glob("*.json"), reverse=True)[:max(1, min(100, int(limit)))]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                item = json.load(handle)
            result.append({"history_id": path.stem, "revision": int(item.get("revision", 0)), "updated_at": item.get("updated_at"), "reason": item.get("history_reason", "auto_save"), "name": item.get("name", "")})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return result


def _save_tail_frame(images, destination: Path) -> None:
    if images is None or len(images) <= 0:
        raise ValueError("无法归档空视频：没有可提取的尾帧。")
    frame = images[-1].detach().float().cpu().clamp(0, 1).numpy()
    array = np.rint(frame * 255.0).astype(np.uint8)
    Image.fromarray(array).save(destination, "PNG")


def _video_components(video):
    if video is None:
        return None, None
    components = video.get_components()
    return components, getattr(components, "images", None)


def archive_take(
    *,
    project: dict[str, Any],
    shot: dict[str, Any],
    initial_video=None,
    final_video=None,
    parent_take_id: str | None = None,
    source_take_id: str | None = None,
    fingerprint: str | None = None,
    primary_fingerprint: str | None = None,
    initial_latent=None,
    fps: float = 24.0,
    output_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one take with independent initial/final delivery choices.

    The selected continuity source is never inferred from an output toggle:
    it is recorded explicitly, and a valid initial video is always preferred
    when the shot requests ``continuity.source = initial``.
    """
    if initial_video is None and final_video is None:
        raise ValueError("没有可归档的视频：最初视频和最终视频均为空。")
    project_id = str(project["project_id"])
    take_id = f"take-{uuid.uuid4().hex[:12]}"
    take_dir = _shot_root(project_id, str(shot["id"])) / "takes" / take_id
    take_dir.mkdir(parents=True, exist_ok=False)

    profile = dict(output_profile or {})
    final_parts = ["DreamShot-H3"]
    if profile.get("second_sampling_mode") == "H3 Latent 超分":
        final_parts.append(f"Latent-{profile.get('second_megapixels', 'auto')}MP")
    elif profile.get("enable_final_video"):
        final_parts.append(f"Refine-{profile.get('passes', 1)}pass")
    method = str(profile.get("final_upscale_method") or "关闭")
    if method == "NVIDIA RTX":
        final_parts.append(f"RTX-{profile.get('rtx_scale', 2)}x-{profile.get('rtx_quality', 'HIGH')}")
    elif method == "TE FlashVSR":
        final_parts.append(f"TEFlashVSR-{profile.get('te_flashvsr_scale', 2)}x-{profile.get('te_flashvsr_mode', 'tiny')}-{profile.get('te_flashvsr_quality', 'balanced')}")
    outputs: dict[str, str | None] = {"initial": None, "final": None}
    for label, video in (("initial", initial_video), ("final", final_video)):
        if video is None:
            continue
        filename = "DreamShot-H3-Original.mp4" if label == "initial" else "-".join(final_parts) + ".mp4"
        video.save_to(
            str(take_dir / filename),
            format=Types.VideoContainer("mp4"),
            codec="auto",
            metadata=None,
        )
        outputs[label] = filename
    latent_filename = None
    if initial_latent is not None:
        latent_filename = "initial_latent.pt"
        torch.save(initial_latent, take_dir / latent_filename)

    continuity = shot.get("continuity") if isinstance(shot.get("continuity"), dict) else {}
    requested_source = "final" if continuity.get("source") == "final" else "initial"
    continuity_video = final_video if requested_source == "final" and final_video is not None else initial_video
    if continuity_video is None:
        continuity_video = final_video
    components, images = _video_components(continuity_video)
    if components is None or images is None:
        raise ValueError("归档视频缺少可读取的画面组件。")
    _save_tail_frame(images, take_dir / "tail.png")

    try:
        fps = float(getattr(components, "frame_rate", 24.0) or 24.0)
    except (TypeError, ValueError):
        fps = 24.0
    try:
        frame_count = int(images.shape[0])
    except Exception:
        frame_count = 0
    record = {
        "schema_version": ARCHIVE_SCHEMA,
        "take_id": take_id,
        "project_id": project_id,
        "shot_id": str(shot["id"]),
        "shot_name": str(shot.get("name") or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_take_id": parent_take_id,
        "source_take_id": source_take_id,
        "fingerprint": fingerprint,
        "primary_fingerprint": primary_fingerprint,
        "mode": shot.get("mode"),
        "prompt": shot.get("prompt"),
        "duration_seconds": shot.get("duration_seconds"),
        "continuity": continuity,
        "output_profile": profile,
        "files": {
            **outputs,
            "tail": "tail.png",
            "initial_latent": latent_filename,
            "continuity_source": "final" if continuity_video is final_video and final_video is not None else "initial",
        },
        "media": {"fps": float(fps), "frame_count": frame_count},
    }
    _atomic_json(take_dir / "take.json", record)
    return record


def load_take(project_id: str, shot_id: str, take_id: str) -> dict[str, Any] | None:
    path = _shot_root(project_id, shot_id) / "takes" / safe_project_path_part(take_id) / "take.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def take_path(project_id: str, shot_id: str, take_id: str, file_key: str) -> Path | None:
    record = load_take(project_id, shot_id, take_id)
    if not record:
        return None
    filename = (record.get("files") or {}).get(file_key)
    if not isinstance(filename, str) or not filename:
        source_take_id = str(record.get("source_take_id") or "")
        if source_take_id and file_key in {"initial", "initial_latent"}:
            return take_path(project_id, shot_id, source_take_id, file_key)
        return None
    path = _shot_root(project_id, shot_id) / "takes" / safe_project_path_part(take_id) / filename
    return path if path.is_file() else None


def _repair_local_seam_spikes(
    source: Path, destination: Path, seam_times: list[float], ffmpeg: str,
    *, radius: int = 5,
) -> list[int]:
    """Redistribute abrupt identity/pose changes around known timeline seams.

    Minterpolate adds temporal samples but preserves both sides of an abrupt
    generated-state change. This second pass detects only exceptional changes
    near known joins and replaces a small window with bidirectional optical
    flow. It streams the full video and buffers only each repair window.
    """
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 48.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    search_radius = max(radius + 2, round(fps * 0.35))
    candidates = [max(1, min(frame_count - 1, round(value * fps))) for value in seam_times]
    scores: dict[int, float] = {}
    previous = None
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
        if previous is not None and any(abs(index - candidate) <= search_radius for candidate in candidates):
            scores[index] = float(np.mean(cv2.absdiff(previous, gray)))
        previous = gray
        index += 1
    capture.release()

    peaks: list[int] = []
    for candidate in candidates:
        local = [(frame_index, score) for frame_index, score in scores.items() if abs(frame_index - candidate) <= search_radius]
        if not local:
            continue
        peak_frame, peak_score = max(local, key=lambda item: item[1])
        baseline = float(np.median([score for _, score in local]))
        if peak_score >= max(4.5, baseline * 1.8):
            # Center one frame before the measured adjacent-frame jump so the
            # two stable anchors straddle the complete state transition.
            peaks.append(max(radius, min(frame_count - radius - 1, peak_frame - 1)))
    peaks = sorted(set(peaks))
    if not peaks:
        shutil.copy2(source, destination)
        return []

    windows: list[tuple[int, int]] = []
    for peak in peaks:
        start, end = peak - radius, peak + radius
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))

    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stderr_log = tempfile.TemporaryFile()
    process = subprocess.Popen([
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
        "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(destination),
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_log)
    try:
        index = 0
        window_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if window_index < len(windows) and index == windows[window_index][0]:
                start, end = windows[window_index]
                buffered = [frame]
                for _ in range(start + 1, end + 1):
                    ok, next_frame = capture.read()
                    if not ok:
                        break
                    buffered.append(next_frame)
                if len(buffered) >= 3:
                    first, last = buffered[0], buffered[-1]
                    gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
                    gray_last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
                    forward = cv2.calcOpticalFlowFarneback(gray_first, gray_last, None, 0.5, 5, 25, 5, 7, 1.5, 0)
                    backward = cv2.calcOpticalFlowFarneback(gray_last, gray_first, None, 0.5, 5, 25, 5, 7, 1.5, 0)
                    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
                    denominator = len(buffered) - 1
                    for offset in range(1, denominator):
                        amount = offset / denominator
                        map_forward = np.dstack((grid_x - forward[..., 0] * amount, grid_y - forward[..., 1] * amount)).astype(np.float32)
                        map_backward = np.dstack((grid_x - backward[..., 0] * (1.0 - amount), grid_y - backward[..., 1] * (1.0 - amount))).astype(np.float32)
                        warped_first = cv2.remap(first, map_forward, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                        warped_last = cv2.remap(last, map_backward, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                        buffered[offset] = cv2.addWeighted(warped_first, 1.0 - amount, warped_last, amount, 0.0)
                for buffered_frame in buffered:
                    process.stdin.write(buffered_frame.tobytes())
                index += len(buffered)
                window_index += 1
            else:
                process.stdin.write(frame.tobytes())
                index += 1
        process.stdin.close()
        return_code = process.wait()
    finally:
        capture.release()
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0 or not destination.is_file():
        stderr_log.seek(0)
        detail = stderr_log.read().decode("utf-8", errors="replace")[-900:]
        destination.unlink(missing_ok=True)
        raise ValueError(f"局部接缝修复失败：{detail}")
    stderr_log.close()
    return peaks


def _delivery_video_stats(path: Path) -> dict[str, Any]:
    """Return stable media facts used by the continuity acceptance panel."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    return {"filename": path.name, "fps": round(fps, 3), "frames": frames,
            "duration_seconds": round(frames / fps, 3) if fps > 0 else 0.0,
            "width": width, "height": height}


def _seam_evidence(path: Path, seam_time: float, destination: Path) -> dict[str, Any]:
    """Create a seven-frame seam strip and measure the largest local jump."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    center = max(1, min(frame_count - 2, round(seam_time * fps)))
    search_radius = max(2, round(fps * 0.35))
    start, end = max(1, center - search_radius), min(frame_count - 1, center + search_radius)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    previous, peak_score, peak_frame = None, 0.0, center
    for frame_index in range(start - 1, end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
        if previous is not None:
            score = float(np.mean(cv2.absdiff(previous, gray)))
            if score > peak_score:
                peak_score, peak_frame = score, frame_index
        previous = gray
    images: list[Image.Image] = []
    for offset in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
        sample_frame = max(0, min(frame_count - 1, round(max(0.0, seam_time + offset) * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
        ok, frame = capture.read()
        if not ok:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width = 240
        height = max(1, round(image.height * width / max(1, image.width)))
        images.append(image.resize((width, height), Image.Resampling.LANCZOS))
    capture.release()
    if images:
        strip = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)), "black")
        left = 0
        for image in images:
            strip.paste(image, (left, 0))
            left += image.width
        strip.save(destination, "PNG", optimize=True)
    return {"peak_change": round(peak_score, 3), "peak_time_seconds": round(peak_frame / fps, 3),
            "strip_filename": destination.name if destination.is_file() else None}


def _write_continuity_report(baseline: Path, repaired: Path, seam_times: list[float]) -> dict[str, Any]:
    """Persist compact visual and numeric evidence beside a merged delivery."""
    report: dict[str, Any] = {"baseline": _delivery_video_stats(baseline),
                              "repaired": _delivery_video_stats(repaired), "seams": []}
    for index, seam_time in enumerate(seam_times, start=1):
        before = _seam_evidence(baseline, seam_time, repaired.with_name(f"{repaired.stem}-seam-{index:02d}-24fps.png"))
        after = _seam_evidence(repaired, seam_time, repaired.with_name(f"{repaired.stem}-seam-{index:02d}-48fps.png"))
        improvement = max(0.0, (before["peak_change"] - after["peak_change"]) / before["peak_change"] * 100.0) if before["peak_change"] > 0 else 0.0
        report["seams"].append({"index": index, "time_seconds": round(seam_time, 3),
                                "loop_start_seconds": round(max(0.0, seam_time - 0.4), 3),
                                "loop_end_seconds": round(seam_time + 0.4, 3),
                                "baseline": before, "repaired": after,
                                "improvement_percent": round(improvement, 1)})
    _atomic_json(repaired.with_suffix(".continuity.json"), report)
    return report


def concat_selected_takes(
    project: dict[str, Any], *, source: str = "final_if_available", shot_ids=None,
    destination_name: str = "timeline.mp4", motion_interpolation: bool = False,
) -> Path:
    """Produce one timeline MP4 from the currently selected take of each enabled shot.

    It uses ffmpeg's concat demuxer with stream copy, so it preserves the
    generated media rather than decoding/re-encoding every frame.  A useful
    error is raised when selected clips cannot safely be concatenated.
    """
    clips: list[Path] = []
    trims: list[float] = []
    clip_durations: list[float] = []
    selected_ids = {str(value) for value in shot_ids} if shot_ids is not None else None
    for shot in project.get("shots") or []:
        if not shot.get("enabled", True):
            continue
        if selected_ids is not None and str(shot.get("id")) not in selected_ids:
            continue
        take_id = str(shot.get("selected_take_id") or "")
        if not take_id:
            raise ValueError(f"{shot.get('name') or '镜头'} 尚未选择生成版本，无法合并导出。")
        if source == "per_shot":
            key = "final" if shot.get("merge_source") == "final" else "initial"
        else:
            key = "final" if source == "final_if_available" else "initial"
        path = take_path(project["project_id"], shot["id"], take_id, key)
        if path is None and key == "final":
            path = take_path(project["project_id"], shot["id"], take_id, "initial")
        if path is None:
            raise ValueError(f"{shot.get('name') or '镜头'} 选中的版本文件不存在，无法合并导出。")
        clips.append(path)
        continuity = shot.get("continuity") if isinstance(shot.get("continuity"), dict) else {}
        record = load_take(project["project_id"], shot["id"], take_id) or {}
        media = record.get("media") if isinstance(record.get("media"), dict) else {}
        clip_fps = float(media.get("fps") or shot.get("fps") or project.get("settings", {}).get("fps") or 24.0)
        # H3 uses reference frames as conditioning and does not normally emit
        # them at the start of the generated clip. Trimming context_frames a
        # second time removed about one second of real continuation footage.
        # Keep the opt-in metadata flag for a future backend that does include
        # reference frames in its encoded output.
        includes_context = bool(media.get("includes_reference_context", False))
        trim_frames = int(continuity.get("context_frames", 0) or 0) if includes_context else 0
        trims.append((max(0.0, trim_frames / clip_fps), clip_fps))
        clip_durations.append(max(0.1, float(shot.get("duration_seconds") or 5.0) - trims[-1][0]))
    if not clips:
        raise ValueError("没有启用且可交付的镜头。")
    out_dir = _root(str(project["project_id"])) / "delivery"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{safe_project_path_part(Path(destination_name).stem)}.mp4"
    concat_file = out_dir / ".timeline.concat.txt"
    temp_dir = out_dir / ".timeline_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    def quote(path: Path) -> str:
        return "file '" + str(path.resolve()).replace("'", "'\\''") + "'"
    prepared: list[Path] = []
    try:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = "ffmpeg"
        for index, (clip, trim_info) in enumerate(zip(clips, trims)):
            trim_seconds, clip_fps = trim_info
            part = temp_dir / f"part-{index:03d}.mp4"
            args = [ffmpeg, "-y"]
            if trim_seconds > 0:
                args.extend(["-ss", f"{trim_seconds:.6f}"])
            args.extend(["-i", str(clip), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", f"{clip_fps:g}", "-c:a", "aac", "-ar", "32000", "-ac", "2", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(part)])
            process = subprocess.run(args, capture_output=True, text=True)
            if process.returncode != 0 or not part.is_file():
                detail = (process.stderr or process.stdout or "未知 ffmpeg 错误").strip()[-900:]
                raise ValueError(f"上下文去重失败：{detail}")
            prepared.append(part)
        # Normalize every input to the same CFR/timebase, then use a short
        # dissolve to soften small pose or cadence differences at the boundary.
        target_fps = 24.0
        # Motion-repaired delivery needs enough temporal room for identity and
        # pose differences to evolve instead of completing in one or two
        # displayed frames. Keep the fast path short for action-heavy edits.
        transition_frames = 8 if motion_interpolation else 4
        transition_seconds = transition_frames / target_fps
        actual_durations: list[float] = []
        try:
            import cv2
            for part in prepared:
                capture = cv2.VideoCapture(str(part))
                frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or target_fps)
                capture.release()
                actual_durations.append(max(transition_seconds * 2.0, frame_count / max(1.0, frame_rate)))
        except Exception:
            actual_durations = [max(transition_seconds * 2.0, value) for value in clip_durations]

        def build_filter(use_transition: bool) -> str:
            filters: list[str] = []
            for index in range(len(prepared)):
                filters.append(f"[{index}:v:0]fps={target_fps:g},setsar=1,settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[v{index}]")
                filters.append(f"[{index}:a:0]aresample=32000:async=1:first_pts=0,aformat=sample_fmts=fltp:sample_rates=32000:channel_layouts=stereo,asetpts=PTS-STARTPTS[a{index}]")
            if not use_transition or len(prepared) == 1:
                concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(prepared)))
                filters.append(f"{concat_inputs}concat=n={len(prepared)}:v=1:a=1[outv][outa]")
                return ";".join(filters)
            video_label, audio_label = "v0", "a0"
            cumulative = actual_durations[0]
            for index in range(1, len(prepared)):
                # Finish the dissolve before the previous stream's terminal
                # sample. Without this guard xfade can leave one mixed frame
                # followed by a full-strength continuation frame, producing
                # the single-frame identity snap visible at the seam.
                end_guard = (2.0 / target_fps) if motion_interpolation else 0.0
                offset = max(0.0, cumulative - transition_seconds - end_guard)
                next_video = "outv" if index == len(prepared) - 1 else f"vx{index}"
                next_audio = "outa" if index == len(prepared) - 1 else f"ax{index}"
                filters.append(f"[{video_label}][v{index}]xfade=transition=fade:duration={transition_seconds:.6f}:offset={offset:.6f}[{next_video}]")
                filters.append(f"[{audio_label}][a{index}]acrossfade=d={transition_seconds:.6f}:c1=tri:c2=tri[{next_audio}]")
                video_label, audio_label = next_video, next_audio
                cumulative += actual_durations[index] - transition_seconds - end_guard
            return ";".join(filters)

        args_base = [ffmpeg, "-y"]
        for clip in prepared:
            args_base.extend(["-i", str(clip)])
        encode_args = ["-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", f"{target_fps:g}", "-vsync", "cfr", "-c:a", "aac", "-ar", "32000", "-ac", "2", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(destination)]
        # The 48 FPS path preserves the full duration of every generated clip.
        # Its localized optical-flow pass repairs the seam without overlapping
        # and shortening the two clips as xfade necessarily does.
        process = subprocess.run(args_base + ["-filter_complex", build_filter(not motion_interpolation)] + encode_args, capture_output=True, text=True)
        if process.returncode != 0 or not destination.is_file():
            process = subprocess.run(args_base + ["-filter_complex", build_filter(False)] + encode_args, capture_output=True, text=True)
        if process.returncode != 0 or not destination.is_file():
            detail = (process.stderr or process.stdout or "未知 ffmpeg 错误").strip()[-900:]
            raise ValueError(f"时间轴合并失败。请确认各段的尺寸、帧率和编码一致。{detail}")
        if motion_interpolation:
            baseline = destination.with_name(f"{destination.stem}-24fps.mp4")
            shutil.copy2(destination, baseline)
            repaired = destination.with_name(f".{destination.stem}.motion-repaired.mp4")
            localized = destination.with_name(f".{destination.stem}.localized-repaired.mp4")
            # These clips are explicitly declared continuous. FFmpeg's default
            # scene-cut detector otherwise mistakes a face/pose seam for a cut
            # and bypasses interpolation at the exact frame that needs repair.
            repair_filter = "minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none"
            repair_args = [
                ffmpeg, "-y", "-i", str(destination), "-vf", repair_filter,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
                str(repaired),
            ]
            repair = subprocess.run(repair_args, capture_output=True, text=True)
            if repair.returncode != 0 or not repaired.is_file():
                repaired.unlink(missing_ok=True)
                detail = (repair.stderr or repair.stdout or "未知 ffmpeg 错误").strip()[-900:]
                raise ValueError(f"运动补帧修复失败，未覆盖普通合并结果：{detail}")
            cumulative = actual_durations[0]
            seam_times: list[float] = []
            for index in range(1, len(prepared)):
                seam_times.append(max(0.0, cumulative))
                cumulative += actual_durations[index]
            try:
                _repair_local_seam_spikes(repaired, localized, seam_times, ffmpeg)
                os.replace(localized, destination)
            except Exception:
                localized.unlink(missing_ok=True)
                raise
            finally:
                repaired.unlink(missing_ok=True)
            _write_continuity_report(baseline, destination, seam_times)
        return destination
    finally:
        try:
            concat_file.unlink(missing_ok=True)
        except OSError:
            pass
        for part in temp_dir.glob("part-*.mp4"):
            try: part.unlink()
            except OSError: pass
        try: temp_dir.rmdir()
        except OSError: pass


def load_take_video_context(project_id: str, shot_id: str, take_id: str, *, source: str = "initial", frames: int = 22):
    """Decode only the tail needed by a later single-shot reroll.

    The generated clip itself remains authoritative in the take archive.  This
    helper only creates an IMAGE tensor for Ref2VA's reference-video input;
    it never rewrites or re-encodes the selected take.
    """
    record = load_take(project_id, shot_id, take_id)
    if not record:
        return None
    key = "final" if source == "final" and (record.get("files") or {}).get("final") else "initial"
    path = take_path(project_id, shot_id, take_id, key)
    if path is None:
        return None
    try:
        import cv2
        capture = cv2.VideoCapture(str(path))
        tail: list[np.ndarray] = []
        wanted = max(5, int(frames))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tail.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(tail) > wanted:
                tail.pop(0)
        capture.release()
        if not tail:
            return None
        return torch.from_numpy(np.stack(tail).astype(np.float32) / 255.0).contiguous()
    except Exception:
        return None
