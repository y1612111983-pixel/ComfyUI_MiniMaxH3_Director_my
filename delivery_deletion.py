"""Containment-safe deletion for one merged Director Studio delivery."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid


def _safe_id(value: str, prefix: str) -> str:
    value = str(value or "")
    if not value.startswith(prefix) or any(token in value for token in ("/", "\\", "..")):
        raise ValueError("项目或合并视频标识不合法。")
    return value


def delete_merged_delivery(output_directory, project_id: str, merged_filename: str) -> dict:
    """Move exactly one merged delivery family into recoverable project trash."""
    project_id = _safe_id(project_id, "project-")
    filename = Path(str(merged_filename or "")).name
    if filename != str(merged_filename or "") or not filename.startswith("selected-timeline-") or not filename.endswith(".mp4") or filename.endswith("-24fps.mp4"):
        raise ValueError("合并视频标识不合法。")
    project_root = (Path(output_directory) / "video" / "Ref2VA_Director" / project_id).resolve()
    project_file = project_root / "project.json"
    if not project_file.is_file():
        raise FileNotFoundError("项目不存在。")
    project = json.loads(project_file.read_text(encoding="utf-8"))
    if project.get("active_job_id"):
        raise RuntimeError("项目正在生成，不能删除合并视频。")
    delivery_root = (project_root / "delivery").resolve()
    if delivery_root.parent != project_root or delivery_root.name != "delivery":
        raise ValueError("合并视频目录验证失败。")
    target = (delivery_root / filename).resolve()
    if target.parent != delivery_root or not target.is_file():
        raise FileNotFoundError("合并视频不存在。")
    stem = target.stem
    associated = sorted(path for path in delivery_root.glob(f"{stem}*") if path.is_file() and path.resolve().parent == delivery_root)
    if target not in associated:
        associated.insert(0, target)
    trash = (project_root / ".trash" / "deliveries" / f"delivery-{uuid.uuid4().hex[:12]}").resolve()
    if project_root not in trash.parents or trash.parent.name != "deliveries":
        raise ValueError("合并视频回收路径验证失败。")
    trash.mkdir(parents=True, exist_ok=False)
    moved = []
    try:
        for source in associated:
            destination = trash / source.name
            shutil.move(str(source), str(destination))
            moved.append(source.name)
        manifest = {"project_id": project_id, "merged_filename": filename, "removed_at": datetime.now(timezone.utc).isoformat(), "files": moved}
        (trash / "delivery.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        for moved_name in moved:
            source = trash / moved_name
            if source.exists():
                shutil.move(str(source), str(delivery_root / moved_name))
        shutil.rmtree(trash, ignore_errors=True)
        raise
    return {"project_id": project_id, "merged_filename": filename, "removed_files": moved,
            "trash_relative": str(trash.relative_to(project_root)).replace("\\", "/"), "recoverable": True}
