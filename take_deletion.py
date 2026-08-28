"""Restricted, recoverable deletion for one Director Studio take."""

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
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass


def delete_archived_take(output_directory: str | os.PathLike[str], project_id: object, shot_id: object, take_id: object) -> dict[str, Any]:
    """Move exactly one validated take to the project's recoverable trash.

    Callers provide identifiers, never paths.  The record on disk must agree
    with all three identifiers before any filesystem mutation is attempted.
    """
    project_id = _identifier(project_id, "项目标识")
    shot_id = _identifier(shot_id, "镜头标识")
    take_id = _identifier(take_id, "版本标识")
    archive_root = (Path(output_directory) / "video" / "Ref2VA_Director").resolve()
    project_dir = archive_root / project_id
    project_json = project_dir / "project.json"
    take_dir = project_dir / "shots" / shot_id / "takes" / take_id
    if not _inside(project_dir, archive_root) or not _inside(take_dir, project_dir):
        raise ValueError("删除目标越出 Ref2VA Director 项目输出目录。")
    if not project_json.is_file():
        raise FileNotFoundError("项目记录不存在。")
    if not take_dir.is_dir():
        raise FileNotFoundError("所选视频版本不存在或已经删除。")
    record_path = take_dir / "take.json"
    if not record_path.is_file():
        raise FileNotFoundError("所选版本缺少 take.json，已阻止删除。")
    with project_json.open("r", encoding="utf-8") as handle:
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
    owned_files: list[str] = []
    for item in take_dir.rglob("*"):
        if item.is_symlink() or not _inside(item, take_dir):
            raise ValueError("版本目录包含越界链接，已阻止删除。")
        if item.is_file():
            owned_files.append(item.relative_to(take_dir).as_posix())
    trash_dir = project_dir / ".trash" / "takes" / shot_id / f"{take_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    if not _inside(trash_dir, project_dir):
        raise ValueError("项目回收目录验证失败。")
    trash_dir.parent.mkdir(parents=True, exist_ok=True)
    removed = takes.pop(take_index)
    if str(shot.get("selected_take_id") or "") == take_id:
        replacement = next((item for item in reversed(takes) if isinstance(item, dict) and (item.get("files") or {}).get("initial")), None)
        if replacement is None:
            replacement = next((item for item in reversed(takes) if isinstance(item, dict) and (item.get("files") or {}).get("final")), None)
        shot["selected_take_id"] = str(replacement.get("take_id")) if replacement else None
        replacement_files = replacement.get("files") if replacement else {}
        preferred = str(shot.get("selected_take_source") or "initial")
        shot["selected_take_source"] = preferred if replacement_files.get(preferred) else ("initial" if replacement_files.get("initial") else "final") if replacement else None
    shot["status"] = "generated" if takes else "draft"
    shutil.move(str(take_dir), str(trash_dir))
    try:
        _atomic_json(project_json, project)
    except Exception:
        shutil.move(str(trash_dir), str(take_dir))
        raise
    return {
        "project": project,
        "shot": shot,
        "removed_take": removed,
        "removed_files": sorted(owned_files),
        "trash_relative": trash_dir.relative_to(project_dir).as_posix(),
        "recoverable": True,
    }
