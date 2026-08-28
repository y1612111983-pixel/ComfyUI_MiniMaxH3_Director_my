import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "final_video_deletion.py"
SPEC = importlib.util.spec_from_file_location("ref2va_final_video_deletion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
delete_upscaled_video = MODULE.delete_upscaled_video
delete_selected_video = MODULE.delete_selected_video


class DeleteUpscaledVideoTests(unittest.TestCase):
    def fixture(self, derived=False):
        temporary = tempfile.TemporaryDirectory()
        output = Path(temporary.name)
        project_dir = output / "video" / "Ref2VA_Director" / "project-test"
        takes_dir = project_dir / "shots" / "shot-test" / "takes"
        source = {"take_id": "take-source", "project_id": "project-test", "shot_id": "shot-test", "files": {"initial": "initial.mp4", "final": None}}
        source_dir = takes_dir / "take-source"
        source_dir.mkdir(parents=True)
        (source_dir / "initial.mp4").write_bytes(b"original")
        (source_dir / "take.json").write_text(json.dumps(source), encoding="utf-8")
        target = {"take_id": "take-target", "project_id": "project-test", "shot_id": "shot-test", "source_take_id": "take-source" if derived else None, "files": {"initial": None if derived else "initial.mp4", "final": "final.mp4"}}
        target_dir = takes_dir / "take-target"
        target_dir.mkdir(parents=True)
        if not derived:
            (target_dir / "initial.mp4").write_bytes(b"target-original")
        (target_dir / "final.mp4").write_bytes(b"upscaled")
        (target_dir / "take.json").write_text(json.dumps(target), encoding="utf-8")
        project = {"project_id": "project-test", "shots": [{"id": "shot-test", "takes": [source, target], "selected_take_id": "take-target", "selected_take_source": "final", "status": "generated"}]}
        (project_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")
        return temporary, output, project_dir, target_dir

    def test_removes_only_final_file_from_combined_take(self):
        temporary, output, project_dir, target_dir = self.fixture()
        self.addCleanup(temporary.cleanup)
        result = delete_upscaled_video(output, "project-test", "shot-test", "take-target")
        self.assertFalse((target_dir / "final.mp4").exists())
        self.assertTrue((target_dir / "initial.mp4").is_file())
        self.assertTrue((project_dir / result["trash_relative"] / "final.mp4").is_file())
        shot = result["project"]["shots"][0]
        self.assertEqual(shot["selected_take_id"], "take-target")
        self.assertEqual(shot["selected_take_source"], "initial")
        self.assertIsNone(shot["takes"][1]["files"]["final"])

    def test_removes_final_only_derivative_but_preserves_source_take(self):
        temporary, output, project_dir, target_dir = self.fixture(derived=True)
        self.addCleanup(temporary.cleanup)
        result = delete_upscaled_video(output, "project-test", "shot-test", "take-target")
        self.assertTrue(result["removed_single_video_take"])
        self.assertFalse(target_dir.exists())
        self.assertTrue((project_dir / "shots" / "shot-test" / "takes" / "take-source" / "initial.mp4").is_file())
        shot = result["project"]["shots"][0]
        self.assertEqual([take["take_id"] for take in shot["takes"]], ["take-source"])
        self.assertEqual(shot["selected_take_id"], "take-source")
        self.assertEqual(shot["selected_take_source"], "initial")

    def test_removes_only_initial_file_and_preserves_final(self):
        temporary, output, project_dir, target_dir = self.fixture()
        self.addCleanup(temporary.cleanup)
        result = delete_selected_video(output, "project-test", "shot-test", "take-target", "initial")
        self.assertFalse((target_dir / "initial.mp4").exists())
        self.assertTrue((target_dir / "final.mp4").is_file())
        self.assertTrue((project_dir / result["trash_relative"] / "initial.mp4").is_file())
        shot = result["project"]["shots"][0]
        self.assertEqual(shot["selected_take_id"], "take-target")
        self.assertEqual(shot["selected_take_source"], "final")
        self.assertIsNone(shot["takes"][1]["files"]["initial"])

    def test_rejects_missing_final_without_changes(self):
        temporary, output, project_dir, target_dir = self.fixture()
        self.addCleanup(temporary.cleanup)
        record_path = target_dir / "take.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["files"]["final"] = None
        record_path.write_text(json.dumps(record), encoding="utf-8")
        before = (project_dir / "project.json").read_bytes()
        with self.assertRaisesRegex(FileNotFoundError, "没有可删除"):
            delete_upscaled_video(output, "project-test", "shot-test", "take-target")
        self.assertEqual((project_dir / "project.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
