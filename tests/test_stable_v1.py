from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
import uuid
import torch


NODE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = NODE_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))


def load_modules():
    package_name = "ref2va_stable_v1_testpkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(NODE_ROOT)]
    sys.modules[package_name] = package

    latest = types.ModuleType("comfy_api.latest")
    latest.Types = object()
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules.setdefault("comfy_api", comfy_api)
    sys.modules.setdefault("comfy_api.latest", latest)

    loaded = {}
    for module_name in ("unified_project", "unified_artifacts"):
        qualified = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified, NODE_ROOT / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded["unified_project"], loaded["unified_artifacts"]


def main() -> None:
    project_module, artifacts = load_modules()
    project_id = f"stable-v1-test-{uuid.uuid4().hex}"
    root = artifacts.project_root(project_id)
    try:
        project = project_module.default_project()
        project["project_id"] = project_id
        first = artifacts.save_editor_project(project, expected_revision=None, reason="isolated_create")
        assert first["revision"] == 1

        # Simulate a runner-owned take. A later editor save must not erase it.
        first["shots"][0]["takes"] = [{"id": "take-safe", "video_path": "isolated.mp4"}]
        first["shots"][0]["selected_take_id"] = "take-safe"
        artifacts.write_project_snapshot(first, reason="isolated_runner")
        current = artifacts.load_project_snapshot(project_id)
        assert current is not None

        stale_form = json.loads(json.dumps(current))
        stale_form["shots"][0]["prompt"] = "autosave survives"
        stale_form["shots"][0]["takes"] = []
        stale_form["shots"][0]["selected_take_id"] = None
        saved = artifacts.save_editor_project(stale_form, expected_revision=current["revision"], reason="isolated_edit")
        assert saved["shots"][0]["prompt"] == "autosave survives"
        assert saved["shots"][0]["takes"][0]["id"] == "take-safe"
        assert saved["shots"][0]["selected_take_id"] == "take-safe"
        assert artifacts.list_project_history(project_id)

        recovered_id = "take-recovered"
        recovered_root = root / "shots" / saved["shots"][0]["id"] / "takes" / recovered_id
        recovered_root.mkdir(parents=True, exist_ok=True)
        (recovered_root / "take.json").write_text(json.dumps({
            "take_id": recovered_id,
            "project_id": project_id,
            "shot_id": saved["shots"][0]["id"],
            "created_at": "2099-01-01T00:00:00+00:00",
            "files": {"initial": "initial.mp4"},
        }), encoding="utf-8")
        recovered = artifacts.recover_archived_takes(project_id)
        assert recovered["recovered_take_count"] == 1
        assert recovered["shots"][0]["selected_take_id"] == recovered_id
        assert any(item.get("take_id") == recovered_id for item in recovered["shots"][0]["takes"])

        # A late stale runner/editor snapshot must not hide an immutable take
        # that is already complete on disk.
        masked = json.loads(json.dumps(recovered))
        masked.pop("recovered_take_count", None)
        masked["shots"][0]["takes"] = []
        masked["shots"][0]["selected_take_id"] = None
        masked["shots"][0]["status"] = "sampling"
        artifacts.write_project_snapshot(masked, reason="isolated_late_stale_write")
        remounted = artifacts.load_project_snapshot(project_id)
        assert remounted is not None
        assert remounted["shots"][0]["status"] == "generated"
        assert remounted["shots"][0]["selected_take_id"] == recovered_id
        assert any(item.get("take_id") == recovered_id for item in remounted["shots"][0]["takes"])

        trash_take = root / ".trash" / "takes" / recovered["shots"][0]["id"] / "take-trash-test"
        trash_take.mkdir(parents=True, exist_ok=True)
        (trash_take / "take.json").write_text("{}", encoding="utf-8")
        (trash_take / "initial.mp4").write_bytes(b"isolated-video-test")
        storage = artifacts.project_storage_summary(project_id)
        assert storage["trash_take_count"] == 1 and storage["trash_file_count"] == 2
        purged = artifacts.purge_project_video_trash(project_id)
        assert purged["removed_take_count"] == 1
        assert purged["removed_file_count"] == 2
        assert purged["remaining_file_count"] == 0

        audio = {"waveform": torch.arange(32000, dtype=torch.float32).reshape(1, 1, -1), "sample_rate": 32000}
        audio_tail = artifacts.trim_audio_tail(audio, 22 / 24)
        assert audio_tail is not None
        assert audio_tail["waveform"].shape[-1] == round(32000 * 22 / 24)
        assert audio_tail["waveform"][0, 0, 0] == audio["waveform"][0, 0, -round(32000 * 22 / 24)]

        seamless = project_module.default_shot(2)
        assert seamless["continuity"]["strategy"] == "auto_seamless"
        route = project_module.continuity_route(seamless, has_previous_take=True)
        assert route["enabled"] is True
        assert route["use_tail_frame"] is True
        assert route["use_context_video"] is True
        assert route["use_context_audio"] is True
        assert route["inject_continuity_prompt"] is True

        legacy_motion = project_module.default_shot(2)
        legacy_motion["continuity"]["strategy"] = "motion"
        legacy_route = project_module.continuity_route(legacy_motion, has_previous_take=True)
        assert legacy_route["inject_continuity_prompt"] is False

        try:
            artifacts.save_editor_project(stale_form, expected_revision=current["revision"], reason="stale_edit")
        except RuntimeError as error:
            assert str(error).startswith("REVISION_CONFLICT:")
        else:
            raise AssertionError("stale revision was not rejected")

        for invalid_id in ("../escape", "..\\escape", "C:\\escape", ""):
            try:
                project_module.validate_project_id(invalid_id)
            except ValueError:
                pass
            else:
                raise AssertionError(f"unsafe project id accepted: {invalid_id!r}")

        print(json.dumps({
            "ok": True,
            "project_id": project_id,
            "final_revision": saved["revision"],
            "history_count": len(artifacts.list_project_history(project_id)),
            "take_preserved": True,
            "stale_conflict_rejected": True,
            "unsafe_ids_rejected": True,
            "orphaned_take_recovered": True,
            "late_stale_write_cannot_hide_take": True,
            "continuity_audio_matches_tail_window": True,
            "video_trash_purged": True,
        }, ensure_ascii=False))
    finally:
        resolved = root.resolve()
        expected_parent = (Path(artifacts.folder_paths.get_output_directory()) / "video" / "Ref2VA_Director").resolve()
        if resolved.parent == expected_parent and resolved.name == project_id and resolved.exists():
            shutil.rmtree(resolved)


if __name__ == "__main__":
    main()
