from pathlib import Path
import unittest


FRONTEND = Path(__file__).resolve().parents[1] / "web" / "ref2va-unified-director-studio-v190.js"


class FrontendSourceRegressionTests(unittest.TestCase):
    def test_global_resource_card_has_image_video_and_audio_without_widget_scope_leak(self):
        source = FRONTEND.read_text(encoding="utf-8")
        start = source.index("const globalResourceCard =")
        end = source.index("for (const bucket of", start)
        card = source[start:end]
        self.assertIn('document.createElement("img")', card)
        self.assertIn('document.createElement("video")', card)
        self.assertIn('document.createElement("audio")', card)
        self.assertNotIn("widgetName", card)
        self.assertNotIn("hiddenDefaults", card)
        self.assertNotIn("afterChange", card)

    def test_frontend_version_matches_release(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('const FRONTEND_VERSION = "1.10.1";', source)

    def test_mode_specific_four_step_lora_controls_are_present(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('enable_turbo_lora: true', source)
        self.assertIn('ref2va_turbo_lora_name: "minimax_h3_ref2v_turbo_4step', source)
        self.assertIn('fl2v_turbo_lora_name: "minimax_h3_fl2v_turbo_4step', source)
        self.assertIn('["ref2va", "continuous_ref2va"].includes', source)
        self.assertIn('makeFamilyLoraControl("ref2va_turbo_lora_name", ["ref2va", "ref2v"]', source)
        self.assertIn('makeFamilyLoraControl("fl2v_turbo_lora_name", ["fl2va", "fl2v"]', source)

    def test_final_upscale_is_mutually_exclusive(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('final_upscale_method: "关闭"', source)
        self.assertIn('method === "NVIDIA RTX"', source)
        self.assertIn('method === "TE FlashVSR"', source)

    def test_upscaled_video_has_independent_delete_action(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('button("删除当前选中视频"', source)
        self.assertIn('/ref2va-director/delete-selected-video', source)
        self.assertIn("原始视频、其他版本、合并视频和输入素材都会保留", source)
        self.assertIn('rtxEnableWidget.value = false', source)

    def test_lora_stack_seed_mode_and_inherited_second_aspect_are_visible_contracts(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('button("＋ 添加 LoRA"', source)
        self.assertIn('name: String(item.name || "")', source)
        self.assertIn('seed_mode: "种子模式"', source)
        self.assertIn('=== "随机") changeRandomSeed()', source)
        refinement_block = source[source.index("const refinementLabels ="):source.index("const refinementDrawer =")]
        self.assertNotIn('second_aspect_ratio:', refinement_block)

    def test_dreamshot_library_dynamic_compare_and_feedback_contracts(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('brandTitle.textContent="梦镜 DreamShot"', source)
        self.assertIn('link.download=`梦镜 DreamShot+${FRONTEND_VERSION}.json`', source)
        self.assertIn('BUG反馈群 · 1106686971', source)
        self.assertIn('视频库 · 原版', source)
        self.assertIn('原版视频', source)
        self.assertIn('二采 / H3 超分视频', source)
        self.assertIn('RTX / TE 视频放大', source)
        self.assertIn('送去放大', source)
        self.assertIn('送去二采', source)
        self.assertIn('基于此原版继续二采 / 放大', source)
        self.assertIn('existingUpscaleSource', source)
        self.assertIn('option.textContent=`当前镜头 ${index+1}`', source)
        self.assertIn('versionLabel.textContent = "生成视频与版本"', source)
        self.assertIn('let shotSettingsOpen = true', source)
        self.assertIn('let versionOpen = true', source)
        self.assertIn('toolbar = document.createElement("details")', source)
        self.assertIn('version = document.createElement("details")', source)
        self.assertIn('markDrawerSummary(document.createElement("summary"))', source)
        self.assertIn('▶ 播放动态对比', source)
        self.assertIn('source_manifest', source)
        self.assertIn('profileLabel(take,"final")', source)
        self.assertIn('本次合并：${profileLabel(candidate,source)}', source)
        self.assertIn('导出合并清单', source)
        self.assertIn('selections, source: "final_if_available"', source)
        self.assertNotIn('8 帧接缝修复 + 48 FPS', source)
        self.assertIn('const repairMotion = false', source)


if __name__ == "__main__":
    unittest.main()
