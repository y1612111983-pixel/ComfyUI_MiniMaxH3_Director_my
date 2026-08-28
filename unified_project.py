"""Stable project contract for the Ref2VA Director Studio.

This module deliberately contains no ComfyUI canvas code.  A shot project must
stay readable and valid whether it is edited by the future timeline UI, a
workflow JSON, or a queued run.  Keeping this contract separate prevents a UI
change from silently changing generation or continuity behaviour.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any


PROJECT_VERSION = 3

# The public names are Chinese because they are stored in the user-facing
# project.  The stable keys are what execution code uses.
MODE_KEYS = {
    "t2v": "文生视频",
    "i2v": "图生视频",
    "fl2v": "首尾帧生视频",
    "ref2va": "参考生视频",
    "continuous_ref2va": "连续参考生视频",
}

CONTINUITY_KEYS = {"off", "tail", "motion", "auto_seamless"}
CONTEXT_FRAME_CHOICES = {5, 22, 39}
TIMELINE_ROLES = {"editable_reference", "fixed_guide", "boundary_only"}
TIMELINE_KINDS = {"image", "video", "audio"}
TIMELINE_USAGES = {"conditioning", "edit"}


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def default_shot(number: int) -> dict[str, Any]:
    """A complete first-class shot; all modes share the same schema."""
    return {
        "id": _new_id("shot"),
        "order": int(number),
        "name": f"镜头 {number}",
        "enabled": True,
        "duration_seconds": 5.0,
        "fps": 24.0,
        # A fresh shareable project must run without an unconnected reference
        # socket.  The first shot is therefore independent T2V by default;
        # later shots demonstrate Ref2VA continuity from its generated tail.
        "mode": "continuous_ref2va" if number > 1 else "t2v",
        "prompt": "" if number > 1 else "建立镜头：明确主体、场景、光线与动作起点。",
        # A mode never removes data from the shot.  It only decides what is
        # active, so switching modes cannot break a saved project.
        "assets": {
            "first_frame": None,
            "last_frame": None,
            "images": [],
            "videos": [],
            "audios": [],
        },
        "timeline": {
            "enabled": False,
            "prompt": "",
            "generation_start": 0.0,
            "generation_end": 5.0,
            "snap_seconds": 0.25,
            "clips": [],
        },
        "continuity": {
            "enabled": number > 1,
            "strategy": "auto_seamless" if number > 1 else "off",
            "context_frames": 22,
            # Generation and delivery have intentionally independent sources.
            "source": "initial",
        },
        # Takes are never overwritten.  A later UI may present these as cards.
        "takes": [],
        "selected_take_id": None,
        "status": "draft",
    }


def default_project() -> dict[str, Any]:
    return {
        "schema_version": PROJECT_VERSION,
        "project_id": _new_id("project"),
        "revision": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "name": "梦镜 DreamShot",
        "global_prompt": "",
        "settings": {
            "fps": 24,
            "delivery_source": "final_if_available",
            "default_context_frames": 22,
            "delivery_motion_interpolation": False,
        },
        "shots": [default_shot(1), default_shot(2), default_shot(3)],
        "jobs": [],
        "active_job_id": None,
    }


def _as_dict(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    return value if isinstance(value, dict) else deepcopy(fallback)


def _clean_mode(value: Any, index: int) -> str:
    raw = str(value or "").strip()
    aliases = {
        "文生视频（T2V）": "t2v", "图生视频（I2V）": "i2v",
        "首尾帧生视频（FL2V）": "fl2v", "多参考生视频（Ref2VA）": "ref2va",
        "参考生视频": "ref2va", "连续参考生视频": "continuous_ref2va",
    }
    raw = aliases.get(raw, raw)
    if raw not in MODE_KEYS:
        return "continuous_ref2va" if index > 0 else "ref2va"
    return raw


def _clean_asset_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [deepcopy(item) for item in value if isinstance(item, dict)]


def normalise_project(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Return one safe, forward-compatible project object.

    It intentionally preserves unknown keys so later Director Studio versions
    can read an earlier project without destroying fields they understand.
    """
    base = default_project()
    parsed: Any = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"导演台项目数据不是有效 JSON：{exc.msg}") from exc
    parsed = parsed if isinstance(parsed, dict) else {}
    project = deepcopy(parsed)
    project["schema_version"] = PROJECT_VERSION
    project["project_id"] = str(project.get("project_id") or base["project_id"])
    try:
        project["revision"] = max(0, int(project.get("revision", 0)))
    except (TypeError, ValueError):
        project["revision"] = 0
    project["updated_at"] = str(project.get("updated_at") or base["updated_at"])
    project["name"] = str(project.get("name") or base["name"]).strip()[:120] or base["name"]
    project["global_prompt"] = str(project.get("global_prompt") or "")
    project["global_constraint_prompt"] = str(project.get("global_constraint_prompt") or "")
    global_assets = project.get("global_assets") if isinstance(project.get("global_assets"), dict) else {}
    project["global_assets"] = {
        "images": list(global_assets.get("images") or []),
        "videos": list(global_assets.get("videos") or []),
        "audios": list(global_assets.get("audios") or []),
    }
    project["settings"] = {**base["settings"], **_as_dict(project.get("settings"), {})}
    try:
        project["settings"]["fps"] = max(1, min(120, int(project["settings"]["fps"])))
    except (TypeError, ValueError):
        project["settings"]["fps"] = 24

    source = str(project["settings"].get("delivery_source") or "final_if_available")
    project["settings"]["delivery_source"] = source if source in {"initial", "final_if_available"} else "final_if_available"
    project["settings"]["delivery_motion_interpolation"] = bool(
        project["settings"].get("delivery_motion_interpolation", False)
    )

    rows = project.get("shots")
    if not isinstance(rows, list) or not rows:
        rows = [default_shot(1)]
    normalised: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_shot in enumerate(rows):
        fallback = default_shot(index + 1)
        shot = {**fallback, **_as_dict(raw_shot, {})}
        shot_id = str(shot.get("id") or fallback["id"])
        if shot_id in used_ids:
            shot_id = _new_id("shot")
        used_ids.add(shot_id)
        shot["id"] = shot_id
        shot["order"] = index + 1
        shot["name"] = str(shot.get("name") or f"镜头 {index + 1}").strip()[:80] or f"镜头 {index + 1}"
        shot["enabled"] = bool(shot.get("enabled", True))
        try:
            shot["duration_seconds"] = max(0.2, min(150.0, float(shot.get("duration_seconds", 5.0))))
        except (TypeError, ValueError):
            shot["duration_seconds"] = 5.0
        try:
            shot["fps"] = max(1.0, min(120.0, float(shot.get("fps", 24.0))))
        except (TypeError, ValueError):
            shot["fps"] = 24.0
        shot["mode"] = _clean_mode(shot.get("mode"), index)
        shot["prompt"] = str(shot.get("prompt") or "")
        assets = {**fallback["assets"], **_as_dict(shot.get("assets"), {})}
        for key in ("images", "videos", "audios"):
            assets[key] = _clean_asset_list(assets.get(key))
        for key in ("first_frame", "last_frame"):
            assets[key] = deepcopy(assets.get(key)) if isinstance(assets.get(key), dict) else None
        shot["assets"] = assets
        raw_timeline = _as_dict(shot.get("timeline"), {})
        timeline = {**fallback["timeline"], **raw_timeline}
        timeline["enabled"] = bool(timeline.get("enabled", False))
        timeline["prompt"] = str(timeline.get("prompt") or "")
        duration = shot["duration_seconds"]
        try:
            timeline["generation_start"] = max(0.0, min(duration, float(timeline.get("generation_start", 0.0))))
        except (TypeError, ValueError):
            timeline["generation_start"] = 0.0
        try:
            timeline["generation_end"] = max(timeline["generation_start"] + 0.01, min(duration, float(timeline.get("generation_end", duration))))
        except (TypeError, ValueError):
            timeline["generation_end"] = duration
        try:
            snap = float(timeline.get("snap_seconds", 0.25))
        except (TypeError, ValueError):
            snap = 0.25
        timeline["snap_seconds"] = min((0.0, 0.1, 0.25, 0.5, 1.0), key=lambda item: abs(item - snap))
        clips = []
        for raw_clip in timeline.get("clips") if isinstance(timeline.get("clips"), list) else []:
            if not isinstance(raw_clip, dict) or not isinstance(raw_clip.get("asset"), dict):
                continue
            clip = deepcopy(raw_clip)
            clip["id"] = str(clip.get("id") or _new_id("clip"))
            kind = str(clip.get("kind") or "video")
            clip["kind"] = kind if kind in TIMELINE_KINDS else "video"
            usage = str(clip.get("usage") or "conditioning")
            clip["usage"] = usage if usage in TIMELINE_USAGES else "conditioning"
            role = str(clip.get("role") or "editable_reference")
            clip["role"] = role if role in TIMELINE_ROLES else "editable_reference"
            try:
                # Conditioning clips are scoped to the current H3 shot.  Edit
                # clips form an independent assembly timeline and may extend
                # well beyond a single shot, so do not truncate them to the
                # generation duration when the project is saved/reloaded.
                timeline_limit = 86400.0 if clip["usage"] == "edit" else duration
                clip["start"] = max(0.0, min(timeline_limit, float(clip.get("start", 0.0))))
                remaining = max(0.01, timeline_limit - clip["start"])
                clip["duration"] = max(0.01, min(remaining, float(clip.get("duration", duration))))
                clip["source_in"] = max(0.0, float(clip.get("source_in", 0.0)))
                clip["source_out"] = max(clip["source_in"] + 0.01, float(clip.get("source_out", clip["source_in"] + clip["duration"])))
            except (TypeError, ValueError):
                clip.update({"start": 0.0, "duration": duration, "source_in": 0.0, "source_out": duration})
            clip["audio_enabled"] = bool(clip.get("audio_enabled", True))
            clips.append(clip)
        timeline["clips"] = clips
        shot["timeline"] = timeline
        continuity = {**fallback["continuity"], **_as_dict(shot.get("continuity"), {})}
        continuity["enabled"] = bool(continuity.get("enabled", index > 0)) and index > 0
        strategy = str(continuity.get("strategy") or "off")
        continuity["strategy"] = strategy if continuity["enabled"] and strategy in CONTINUITY_KEYS else "off"
        try:
            frames = int(continuity.get("context_frames", 22))
        except (TypeError, ValueError):
            frames = 22
        continuity["context_frames"] = min(CONTEXT_FRAME_CHOICES, key=lambda item: abs(item - frames))
        continuity["source"] = "final" if continuity.get("source") == "final" else "initial"
        shot["continuity"] = continuity
        shot["takes"] = _clean_asset_list(shot.get("takes"))
        selected = shot.get("selected_take_id")
        shot["selected_take_id"] = str(selected) if selected else None
        shot["status"] = str(shot.get("status") or "draft")
        normalised.append(shot)
    project["shots"] = normalised
    project["jobs"] = [deepcopy(item) for item in (project.get("jobs") or []) if isinstance(item, dict)][-100:]
    active_job_id = project.get("active_job_id")
    project["active_job_id"] = str(active_job_id) if active_job_id else None
    return project


def shot_fingerprint(project: dict[str, Any], index: int, previous_take_id: str | None = None, *, include_refinement: bool = True) -> str:
    """A cache identity that invalidates when the actual shot contract changes."""
    shot = project["shots"][int(index)]
    render = project.get("settings", {}).get("render", {})
    if not include_refinement:
        render = {key: render.get(key) for key in ("aspect_ratio", "megapixels", "primary")}
    shot_payload = {
        "mode": shot["mode"], "prompt": shot["prompt"],
        "duration_seconds": shot["duration_seconds"], "assets": shot["assets"], "timeline": shot.get("timeline", {}),
        "continuity": shot["continuity"], "previous_take_id": previous_take_id,
    }
    strategy = str((shot.get("continuity") or {}).get("strategy") or "off")
    if strategy in {"motion", "auto_seamless"}:
        # Invalidate only motion-continuity takes made before the visual and
        # audio reference windows were length-matched.  Independent shots can
        # still reuse their valid cache.
        shot_payload["continuity_engine"] = (
            "auto-seamless-prompt-v1" if strategy == "auto_seamless" else "tail-audio-window-v2"
        )
    payload = {
        "project_schema": PROJECT_VERSION,
        "global_prompt": project.get("global_prompt", ""),
        "global_constraint_prompt": project.get("global_constraint_prompt", ""),
        "global_assets": project.get("global_assets", {}),
        "fps": project.get("settings", {}).get("fps", 24),
        # The runner persists its actual render contract here: resolution,
        # sampling and whether a final pass exists all affect what a take is.
        "render": render,
        "shot": shot_payload,
    }
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def continuity_route(shot: dict[str, Any], *, has_previous_take: bool) -> dict[str, Any]:
    """Describe how every user-facing mode may inherit the previous shot.

    The route is deliberately an execution *intent*, not a canvas rewiring
    instruction.  This is how modes can be switched without deleting a saved
    first/last frame or any reference asset.

    ``fl2v`` deserves special handling: an explicit user first frame and the
    previous generated tail are both hard start conditions.  They cannot be
    silently merged.  The UI must ask which one wins; until then the route is
    marked ``needs_start_choice`` rather than guessing.
    """
    mode = str(shot.get("mode") or "ref2va")
    continuity = _as_dict(shot.get("continuity"), {})
    assets = _as_dict(shot.get("assets"), {})
    requested = bool(continuity.get("enabled")) and has_previous_take
    strategy = str(continuity.get("strategy") or "off") if requested else "off"
    result = {
        "enabled": strategy != "off",
        "strategy": strategy,
        "use_tail_frame": False,
        "use_context_video": False,
        "use_context_audio": False,
        "conditioning_mode": mode,
        "needs_start_choice": False,
        "message": "独立生成，不引用上一镜头。",
    }
    if not result["enabled"]:
        return result
    if strategy == "tail":
        result.update({
            "use_tail_frame": True,
            "message": "使用上一镜头安全尾帧承接。",
        })
        return result
    if strategy not in {"motion", "auto_seamless"}:
        result.update({"enabled": False, "strategy": "off"})
        return result

    result.update({
        "use_tail_frame": True,
        "use_context_video": True,
        "use_context_audio": True,
        # Context continuation uses Ref2VA conditioning for all five modes.
        # The original selected mode remains in project data for the UI and
        # for endpoint validation; it is never overwritten.
        "conditioning_mode": "continuous_ref2va",
        "inject_continuity_prompt": strategy == "auto_seamless",
        "message": (
            "自动继承上一镜头末尾状态，并使用视频、音频与安全尾帧承接。"
            if strategy == "auto_seamless"
            else "使用上一镜头末尾视频、音频与安全尾帧承接。"
        ),
    })
    if mode == "fl2v" and assets.get("first_frame") is not None:
        policy = str(continuity.get("fl2v_start_policy") or "ask")
        if policy not in {"previous_tail", "explicit_first"}:
            result.update({
                "needs_start_choice": True,
                "message": "首尾帧镜头同时有手动首帧和上一镜头尾帧；运行前需选择谁作为起点。",
            })
        elif policy == "explicit_first":
            result["use_tail_frame"] = False
            result["message"] = "保留手动首帧；上一镜头视频和音频仅作为连续参考。"
        else:
            result["message"] = "以上一镜头尾帧作为首帧；保留手动尾帧作为结束约束。"
    return result


def project_json(project: dict[str, Any]) -> str:
    return json.dumps(normalise_project(project), ensure_ascii=False, separators=(",", ":"))


def safe_project_path_part(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value or ""))
    return value.strip("_")[:80] or "project"


def validate_project_id(value: str) -> str:
    """Return an API-safe project id, rejecting path-like input outright."""
    project_id = str(value or "")
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,80}", project_id):
        raise ValueError("project_id 只能包含字母、数字、下划线和连字符。")
    return project_id
