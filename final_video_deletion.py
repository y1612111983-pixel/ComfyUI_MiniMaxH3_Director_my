"""Recoverable deletion for one selected Director Studio video file."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
from typing import Any


_IDENTIFIER = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]{0,79}$")


def _identifier(value: object, label: str) -> str:
    text = str(value or "")
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(f"{label} 不合法。")
    return text


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def delete_selected_video(output_directory: str | os.PathLike[str], project_id: object, shot_id: object, take_id: object, source: object) -> dict[str, Any]:
    """Move only the selected initial/final video to trash."""
    project_id = _identifier(project_id, "项目标识")
    shot_id = _identifier(shot_id, "镜头标识")
    take_id = _identifier(take_id, "版本标识")
    source = str(source or "")
    if source not in {"initial", "final"}:
        raise ValueError("视频来源必须是 initial 或 final。")
    archive_root = (Path(output_directory) / "video" / "Ref2VA_Director").resolve()
    project_dir = archive_root / project_id
    project_path = project_dir / "project.json"
    take_dir = project_dir / "shots" / shot_id / "takes" / take_id
    record_path = take_dir / "take.json"
    if not _inside(project_dir, archive_root) or not _inside(take_dir, project_dir):
        raise ValueError("删除目标越出 Ref2VA Director 项目输出目录。")
    if not project_path.is_file() or not record_path.is_file():
        raise FileNotFoundError("所选视频的项目或版本记录不存在。")
    with project_path.open("r", encoding="utf-8") as handle:
        project = json.load(handle)
    with record_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if str(project.get("project_id") or "") != project_id:
        raise ValueError("项目记录与请求的项目标识不一致。")
    if any(str(record.get(key) or "") != expected for key, expected in (("project_id", project_id), ("shot_id", shot_id), ("take_id", take_id))):
        raise ValueError("版本记录与项目、镜头或版本标识不一致。")
    shots = project.get("shots") if isinstance(project.get("shots"), list) else []
    shot = next((item for item in shots if isinstance(item, dict) and str(item.get("id") or "") == shot_id), None)
    if shot is None:
        raise FileNotFoundError("项目中不存在指定镜头。")
    takes = shot.get("takes") if isinstance(shot.get("takes"), list) else []
    take_index = next((index for index, item in enumerate(takes) if isinstance(item, dict) and str(item.get("take_id") or "") == take_id), None)
    if take_index is None:
        raise FileNotFoundError("项目中不存在指定视频版本。")
    files = record.get("files") if isinstance(record.get("files"), dict) else {}
    selected_name = files.get(source)
    if not isinstance(selected_name, str) or not selected_name:
        raise FileNotFoundError("所选版本没有可删除的当前视频。")
    selected_path = take_dir / selected_name
    if not _inside(selected_path, take_dir) or not selected_path.is_file() or selected_path.is_symlink():
        raise ValueError("所选视频文件缺失或越出版本目录，已阻止删除。")
    trash_dir = project_dir / ".trash" / "selected-videos" / shot_id / f"{take_id}-{source}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    if not _inside(trash_dir, project_dir):
        raise ValueError("项目回收目录验证失败。")
    trash_dir.mkdir(parents=True, exist_ok=False)
    trash_path = trash_dir / selected_path.name
    shutil.move(str(selected_path), str(trash_path))
    original_record = json.loads(json.dumps(record))
    original_project = json.loads(json.dumps(project))
    try:
        record["files"][source] = None
        takes[take_index].setdefault("files", {})[source] = None
        other_source = "initial" if source == "final" else "final"
        has_other_video = bool(files.get(other_source))
        if has_other_video:
            if str(shot.get("selected_take_id") or "") == take_id and str(shot.get("selected_take_source") or "") == source:
                shot["selected_take_source"] = other_source
            _atomic_json(record_path, record)
        else:
            takes.pop(take_index)
            if str(shot.get("selected_take_id") or "") == take_id:
                replacement = next((item for item in reversed(takes) if isinstance(item, dict) and (item.get("files") or {}).get("initial")), None)
                if replacement is None:
                    replacement = next((item for item in reversed(takes) if isinstance(item, dict) and (item.get("files") or {}).get("final")), None)
                shot["selected_take_id"] = str(replacement.get("take_id")) if replacement else None
                replacement_files = replacement.get("files") if replacement else {}
                shot["selected_take_source"] = "initial" if replacement_files.get("initial") else "final" if replacement else None
            shot["status"] = "generated" if takes else "draft"
            for item in take_dir.iterdir():
                if item.is_symlink() or not _inside(item, take_dir):
                    raise ValueError("单视频版本包含越界链接，已阻止删除。")
            shutil.move(str(take_dir), str(trash_dir / "take-record"))
        _atomic_json(project_path, project)
    except Exception:
        if (trash_dir / "take-record").is_dir() and not take_dir.exists():
            shutil.move(str(trash_dir / "take-record"), str(take_dir))
        if trash_path.is_file() and not selected_path.exists():
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trash_path), str(selected_path))
        _atomic_json(project_path, original_project)
        if record_path.parent.is_dir():
            _atomic_json(record_path, original_record)
        raise
    return {
        "project": project,
        "shot": shot,
        "removed_file": selected_name,
        "removed_source": source,
        "removed_single_video_take": not has_other_video,
        "trash_relative": trash_dir.relative_to(project_dir).as_posix(),
        "recoverable": True,
    }


def delete_upscaled_video(output_directory: str | os.PathLike[str], project_id: object, shot_id: object, take_id: object) -> dict[str, Any]:
    """Compatibility wrapper for the earlier final-only endpoint."""
    return delete_selected_video(output_directory, project_id, shot_id, take_id, "final")
