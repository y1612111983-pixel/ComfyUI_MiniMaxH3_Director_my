import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "take_deletion.py"
SPEC = importlib.util.spec_from_file_location("ref2va_take_deletion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
delete_archived_take = MODULE.delete_archived_take


class DeleteArchivedTakeTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        output = Path(temporary.name)
        project_id, shot_id = "project-test", "shot-test"
        project_dir = output / "video" / "Ref2VA_Director" / project_id
        takes_dir = project_dir / "shots" / shot_id / "takes"
        records = []
        for take_id in ("take-first", "take-second"):
            take_dir = takes_dir / take_id
            take_dir.mkdir(parents=True)
            files = {"initial": "initial.mp4", "final": "final.mp4", "tail": "tail.png", "initial_latent": "initial_latent.pt"}
            record = {"take_id": take_id, "project_id": project_id, "shot_id": shot_id, "files": files}
            for filename in files.values():
                (take_dir / filename).write_bytes(take_id.encode())
            (take_dir / "take.json").write_text(json.dumps(record), encoding="utf-8")
            records.append(record)
        delivery = project_dir / "delivery" / "selected-timeline.mp4"
        delivery.parent.mkdir(parents=True)
        delivery.write_bytes(b"merged")
        project = {"project_id": project_id, "shots": [{"id": shot_id, "name": "镜头 1", "takes": records, "selected_take_id": "take-second", "selected_take_source": "initial", "status": "generated"}]}
        (project_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")
        return temporary, output, project_dir, delivery

    def test_legal_target_moves_only_selected_take_and_updates_project(self):
        temporary, output, project_dir, delivery = self.fixture()
        self.addCleanup(temporary.cleanup)
        result = delete_archived_take(output, "project-test", "shot-test", "take-second")
        self.assertTrue(result["recoverable"])
        self.assertFalse((project_dir / "shots" / "shot-test" / "takes" / "take-second").exists())
        self.assertTrue((project_dir / result["trash_relative"] / "initial.mp4").is_file())
        self.assertTrue((project_dir / "shots" / "shot-test" / "takes" / "take-first" / "initial.mp4").is_file())
        self.assertTrue(delivery.is_file())
        project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        shot = project["shots"][0]
        self.assertEqual([item["take_id"] for item in shot["takes"]], ["take-first"])
        self.assertEqual(shot["selected_take_id"], "take-first")
        self.assertEqual(set(result["removed_files"]), {"initial.mp4", "final.mp4", "tail.png", "initial_latent.pt", "take.json"})

    def test_traversal_identifier_is_rejected_without_changes(self):
        temporary, output, project_dir, _ = self.fixture()
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "不合法"):
            delete_archived_take(output, "../project-test", "shot-test", "take-second")
        self.assertTrue((project_dir / "shots" / "shot-test" / "takes" / "take-second").is_dir())

    def test_missing_take_is_rejected_without_project_change(self):
        temporary, output, project_dir, _ = self.fixture()
        self.addCleanup(temporary.cleanup)
        before = (project_dir / "project.json").read_bytes()
        with self.assertRaisesRegex(FileNotFoundError, "不存在"):
            delete_archived_take(output, "project-test", "shot-test", "take-missing")
        self.assertEqual((project_dir / "project.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
