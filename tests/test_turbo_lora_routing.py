import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "__init__.py"


def load_default_selector():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name in {"_default_turbo_lora", "_lora_matches_family"}]
    nodes.sort(key=lambda item: item.name != "_lora_matches_family")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_default_turbo_lora"]


class TurboLoraRoutingTests(unittest.TestCase):
    def test_selects_matching_four_step_lora_for_each_family(self):
        choose = load_default_selector()
        loras = [
            "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        ]
        self.assertEqual(choose(loras, "ref2va"), loras[1])
        self.assertEqual(choose(loras, "fl2v"), loras[2])

    def test_does_not_cross_route_or_fall_back_to_eight_step(self):
        choose = load_default_selector()
        self.assertEqual(choose(["minimax_h3_fl2v_turbo_4step.safetensors"], "ref2va"), "")
        self.assertEqual(choose(["minimax_h3_ref2v_turbo_8step.safetensors"], "ref2va"), "")

    def test_backend_allows_manual_cross_family_selection(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('available_loras = set(folder_paths.get_filename_list("loras"))', source)
        self.assertNotIn('不能使用非 Ref2VA LoRA', source)
        self.assertNotIn('不能使用非 FL2V LoRA', source)


if __name__ == "__main__":
    unittest.main()
