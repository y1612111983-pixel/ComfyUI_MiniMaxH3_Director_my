import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "delivery_deletion.py"
SPEC = importlib.util.spec_from_file_location("ref2va_delivery_deletion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
delete_merged_delivery = MODULE.delete_merged_delivery


class DeleteMergedDeliveryTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        output = Path(temporary.name)
        project_root = output / "video" / "Ref2VA_Director" / "project-test"
        delivery = project_root / "delivery"
        delivery.mkdir(parents=True)
        (project_root / "project.json").write_text(json.dumps({"project_id": "project-test", "active_job_id": None}), encoding="utf-8")
        stem = "selected-timeline-20260827-120000-abcdef"
        names = [f"{stem}.mp4", f"{stem}-24fps.mp4", f"{stem}.continuity.json", f"{stem}-seam-01-24fps.png", f"{stem}-seam-01-48fps.png"]
        for name in names:
            (delivery / name).write_bytes(name.encode())
        unrelated = delivery / "selected-timeline-keep.mp4"
        unrelated.write_bytes(b"keep")
        return temporary, output, project_root, names, unrelated

    def test_delivery_family_moves_without_touching_source_or_other_delivery(self):
        temporary, output, project_root, names, unrelated = self.fixture()
        self.addCleanup(temporary.cleanup)
        result = delete_merged_delivery(output, "project-test", names[0])
        self.assertTrue(result["recoverable"])
        self.assertEqual(set(result["removed_files"]), set(names))
        self.assertTrue(unrelated.is_file())
        trash = project_root / result["trash_relative"]
        self.assertTrue((trash / "delivery.json").is_file())
        self.assertTrue(all((trash / name).is_file() for name in names))

    def test_traversal_and_baseline_alias_are_rejected_without_changes(self):
        temporary, output, project_root, names, unrelated = self.fixture()
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(ValueError):
            delete_merged_delivery(output, "project-test", "../" + names[0])
        with self.assertRaises(ValueError):
            delete_merged_delivery(output, "project-test", names[1])
        self.assertTrue((project_root / "delivery" / names[0]).is_file())
        self.assertTrue(unrelated.is_file())

    def test_missing_target_and_active_project_are_rejected(self):
        temporary, output, project_root, names, _ = self.fixture()
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(FileNotFoundError):
            delete_merged_delivery(output, "project-test", "selected-timeline-missing.mp4")
        (project_root / "project.json").write_text(json.dumps({"project_id": "project-test", "active_job_id": "job-running"}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            delete_merged_delivery(output, "project-test", names[0])


if __name__ == "__main__":
    unittest.main()
