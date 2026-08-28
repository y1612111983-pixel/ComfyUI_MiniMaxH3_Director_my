import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "image_batching.py"
SPEC = importlib.util.spec_from_file_location("ref2va_image_batching", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
batch_reference_images = MODULE.batch_reference_images


class ReferenceImageBatchingTests(unittest.TestCase):
    def test_different_portrait_and_square_sizes_are_padded_without_resampling(self):
        portrait = torch.full((1, 821, 640, 3), 0.25)
        square = torch.full((1, 1024, 1024, 3), 0.75)
        result = batch_reference_images([portrait, square])
        self.assertEqual(tuple(result.shape), (2, 1024, 1024, 3))
        # The complete source remains unchanged in the center.
        top, left = (1024 - 821) // 2, (1024 - 640) // 2
        self.assertTrue(torch.equal(result[0, top:top + 821, left:left + 640], portrait[0]))
        self.assertTrue(torch.equal(result[1], square[0]))
        # Replicated margins do not introduce black pixels.
        self.assertEqual(float(result[0].min()), 0.25)

    def test_matching_sizes_are_concatenated_unchanged(self):
        first = torch.rand((1, 64, 96, 3))
        second = torch.rand((1, 64, 96, 3))
        result = batch_reference_images([first, second])
        self.assertTrue(torch.equal(result, torch.cat([first, second], dim=0)))

    def test_empty_and_invalid_inputs_have_clear_behavior(self):
        self.assertIsNone(batch_reference_images([]))
        with self.assertRaisesRegex(ValueError, "ComfyUI IMAGE"):
            batch_reference_images([torch.zeros((64, 64, 3))])


if __name__ == "__main__":
    unittest.main()
