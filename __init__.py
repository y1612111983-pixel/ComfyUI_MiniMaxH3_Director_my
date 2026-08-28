"""Compact controls for MiniMax H3 Ref2VA workflows."""
import json
import math
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps
import torch
import nodes
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import latent_preview
import folder_paths
from comfy_api.latest import io, ui, Types
from comfy_api.latest._input_impl import VideoFromFile
from comfy_extras.nodes_audio import LoadAudio, vae_decode_audio
from comfy_extras.nodes_video import CreateVideo, SaveVideo
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3ImageToVideo, MiniMaxH3AddGuide
from custom_nodes.ComfyUI_MiniMaxH3_Director.director.h3_latent_upscale import upscale_h3_video_latent
from custom_nodes.ComfyUI_MiniMaxH3_Director.director.refine_pack import resolution_from_selector
from .unified_project import MODE_KEYS, PROJECT_VERSION, continuity_route, default_project, normalise_project, project_json, safe_project_path_part, shot_fingerprint
from .unified_artifacts import archive_take, concat_selected_takes, list_project_history, load_project_snapshot, load_take, load_take_video_context, project_root, project_storage_summary, purge_project_video_trash, recover_archived_takes, save_editor_project, take_path, trim_audio_tail, write_project_snapshot
from .take_deletion import delete_archived_take
from .final_video_deletion import delete_selected_video, delete_upscaled_video
from .delivery_deletion import delete_merged_delivery
from .image_batching import batch_reference_images

REF2VA_DIRECTOR_BACKEND_VERSION = "1.10.1"

SCHEDULERS = ["beta", "normal", "simple", "sgm_uniform", "karras", "exponential"]
SPACINGS = ["cosine", "linear", "exponential"]
LATENT_UPSCALE_MODELS = folder_paths.get_filename_list("latent_upscale_models")
DEFAULT_LATENT_UPSCALE_MODEL = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
if DEFAULT_LATENT_UPSCALE_MODEL not in LATENT_UPSCALE_MODELS and LATENT_UPSCALE_MODELS:
    DEFAULT_LATENT_UPSCALE_MODEL = LATENT_UPSCALE_MODELS[0]


def _storyboard_shot_index(value):
    """Return a stable positive shot number from a workbench shot name."""
    match = re.search(r"(\d+)", str(value or ""))
    return max(1, int(match.group(1))) if match else 1


def _storyboard_shot_dir(value):
    return f"镜头_{_storyboard_shot_index(value)}"


def _storyboard_tail_path(current_shot):
    """Find the previous shot's deterministic tail-frame cache path."""
    previous_index = _storyboard_shot_index(current_shot) - 1
    if previous_index < 1:
        return None
    return os.path.join(
        folder_paths.get_output_directory(), "video", "连续分镜", f"镜头_{previous_index}", "尾帧.png"
    )


def _image_tensor_from_path(path):
    """Read one RGB image as ComfyUI IMAGE [1,H,W,C] without using VHS."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        values = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(values)[None, ...].contiguous()


def _save_tail_frame(images, path):
    """Persist the last decoded frame for the next shot's automatic handoff."""
    frame = images[-1].detach().float().cpu().clamp(0, 1).numpy()
    array = np.rint(frame * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path, "PNG")

class Ref2VARTXVideoPostprocess(io.ComfyNode):
    """Single-node wrapper for extract -> RTX VSR -> rebuild -> save."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VARTXVideoPostprocess",
            display_name="Ref2VA：RTX 视频放大并保存",
            category="Ref2VA/output",
            is_output_node=True,
            inputs=[
                io.Video.Input("video", display_name="输入视频"),
                io.DynamicCombo.Input("resize_type", options=[
                    io.DynamicCombo.Option("scale by multiplier", [
                        io.Float.Input("scale", default=2.5, min=1.0, max=4.0, step=0.05, display_name="放大倍数")
                    ]),
                    io.DynamicCombo.Option("target dimensions", [
                        io.Int.Input("width", default=1920, min=64, max=8192, step=8, display_name="目标宽度"),
                        io.Int.Input("height", default=1080, min=64, max=8192, step=8, display_name="目标高度"),
                    ]),
                ]),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH", display_name="RTX 质量"),
                io.String.Input("filename_prefix", default="video/ComfyUI_RTX", display_name="保存文件名前缀"),
            ],
            outputs=[io.Video.Output(display_name="放大后视频")],
        )

    @classmethod
    def execute(cls, video, resize_type, quality="HIGH", filename_prefix="video/ComfyUI_RTX"):
        try:
            import nvvfx
        except Exception as exc:
            raise RuntimeError("RTX 视频放大需要已安装并可用的 NVIDIA RTX 节点库。") from exc
        components = video.get_components()
        images = components.images
        b, h, w, c = images.shape
        kind = resize_type.get("resize_type") if isinstance(resize_type, dict) else "scale by multiplier"
        if kind == "target dimensions":
            out_w, out_h = int(resize_type.get("width", 1920)), int(resize_type.get("height", 1080))
        else:
            scale = float(resize_type.get("scale", 2.5))
            out_w, out_h = int(w * scale), int(h * scale)
        out_w, out_h = max(8, round(out_w / 8) * 8), max(8, round(out_h / 8) * 8)
        quality_map = {"LOW": nvvfx.effects.QualityLevel.LOW, "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM, "HIGH": nvvfx.effects.QualityLevel.HIGH, "ULTRA": nvvfx.effects.QualityLevel.ULTRA}
        with nvvfx.VideoSuperRes(quality_map.get(quality, nvvfx.effects.QualityLevel.HIGH)) as sr:
            sr.output_width, sr.output_height = out_w, out_h
            sr.load()
            out = torch.empty((b, out_h, out_w, c), device=images.device, dtype=images.dtype)
            for i in range(b):
                frame = images[i].cuda().permute(2, 0, 1).float().contiguous()
                result = sr.run(frame).image
                out[i] = torch.from_dlpack(result).movedim(0, -1)
        rebuilt = CreateVideo.execute(images=out, audio=components.audio, fps=float(components.frame_rate), bit_depth=video.get_bit_depth())
        rebuilt_video = rebuilt.args[0] if hasattr(rebuilt, "args") else rebuilt[0]
        # Do not call SaveVideo.execute here: its hidden prompt/extra_pnginfo
        # context is only populated when ComfyUI invokes that node directly.
        # Calling it from a wrapper leaves cls.hidden as None on ComfyUI 0.33.x.
        out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), out_w, out_h
        )
        file = f"{filename}_{counter:05}_.mp4"
        full_path = os.path.join(out_dir, file)
        rebuilt_video.save_to(full_path, format=Types.VideoContainer("mp4"), codec="auto", metadata=None)
        return io.NodeOutput(
            rebuilt_video,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )

class Ref2VAVideoPostprocess(io.ComfyNode):
    """Standalone postprocess for an already rendered video."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VAVideoPostprocess", display_name="Ref2VA：视频后处理台（RTX / TE）",
            category="Ref2VA/output", is_output_node=True,
            inputs=[
                io.Video.Input("video", display_name="输入视频"),
                io.Combo.Input("engine", options=["NVIDIA RTX", "TE FlashVSR"], default="TE FlashVSR", display_name="放大引擎"),
                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05, display_name="放大倍数"),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH", display_name="RTX 质量"),
                io.Combo.Input("te_mode", options=["tiny", "tiny-long", "full"], default="tiny", display_name="TE 模式"),
                io.Combo.Input("te_attention", options=["sparse_sage2", "block_sparse_attn", "auto"], default="sparse_sage2", display_name="TE 注意力后端"),
                io.String.Input("filename_prefix", default="video/ComfyUI_Postprocess", display_name="保存文件名前缀"),
            ], outputs=[io.Video.Output(display_name="放大后视频")])

    @classmethod
    def execute(cls, video, engine="TE FlashVSR", scale=2.0, quality="HIGH", te_mode="tiny", te_attention="sparse_sage2", filename_prefix="video/ComfyUI_Postprocess"):
        components = video.get_components()
        images = components.images
        if engine == "NVIDIA RTX":
            try:
                import nvvfx
            except Exception as exc:
                raise RuntimeError("RTX 视频放大需要已安装并可用的 NVIDIA RTX 节点库。") from exc
            b, h, w, c = images.shape
            out_w, out_h = max(8, round(w * float(scale) / 8) * 8), max(8, round(h * float(scale) / 8) * 8)
            levels = {"LOW": nvvfx.effects.QualityLevel.LOW, "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM, "HIGH": nvvfx.effects.QualityLevel.HIGH, "ULTRA": nvvfx.effects.QualityLevel.ULTRA}
            with nvvfx.VideoSuperRes(levels.get(quality, nvvfx.effects.QualityLevel.HIGH)) as sr:
                sr.output_width, sr.output_height = out_w, out_h; sr.load()
                out = torch.empty((b, out_h, out_w, c), device=images.device, dtype=images.dtype)
                for i in range(b):
                    result = sr.run(images[i].cuda().permute(2, 0, 1).float().contiguous()).image
                    out[i] = torch.from_dlpack(result).movedim(0, -1)
        else:
            required = ("TEFlashVSRModelLoader", "TEFlashVSRTuning", "TEFlashVSRRestore")
            missing = [name for name in required if name not in nodes.NODE_CLASS_MAPPINGS]
            if missing: raise RuntimeError("TE FlashVSR 节点未加载：" + ", ".join(missing))
            loader = nodes.NODE_CLASS_MAPPINGS["TEFlashVSRModelLoader"]().load(str("FlashVSR-v1.1"), str(te_mode), "bf16", "auto")[0]
            settings = nodes.NODE_CLASS_MAPPINGS["TEFlashVSRTuning"]().build("balanced", 1.0, "auto", "staged", 2.0, 3.0, 11, 256, 24, 4, str(te_attention))[0]
            out = nodes.NODE_CLASS_MAPPINGS["TEFlashVSRRestore"]().restore(loader, images, int(round(float(scale))), True, 0, settings=settings)[0]
        rebuilt = CreateVideo.execute(images=out, audio=components.audio, fps=float(components.frame_rate), bit_depth=video.get_bit_depth())
        rebuilt_video = rebuilt.args[0] if hasattr(rebuilt, "args") else rebuilt[0]
        out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), out.shape[2], out.shape[1])
        file = f"{filename}_{counter:05}_.mp4"; full_path = os.path.join(out_dir, file)
        rebuilt_video.save_to(full_path, format=Types.VideoContainer("mp4"), codec="auto", metadata=None)
        return io.NodeOutput(rebuilt_video, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

class Ref2VAModelLoader:
    """One panel for the Ref2VA model files and optional Turbo LoRA.

    The prompt intentionally lives in the generation node: it belongs to the
    selected Ref2VA/FL2V mode, not to the model files.
    """
    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        loras = folder_paths.get_filename_list("loras")
        default_lora = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
        if default_lora not in loras and loras:
            default_lora = loras[0]
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"label": "Ref2VA UNet"}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "label": "UNet 数据类型"}),
            "clip_name": (folder_paths.get_filename_list("text_encoders"), {"label": "H3 CLIP"}),
            "clip_type": (["minimax"], {"default": "minimax", "label": "CLIP 类型"}),
            "video_vae_name": (vaes, {"label": "视频 VAE"}),
            "audio_vae_name": (vaes, {"label": "音频 VAE"}),
            "enable_turbo_lora": ("BOOLEAN", {"default": True, "label_on": "开启：使用 Turbo 加速 LoRA", "label_off": "关闭：不加载 Turbo LoRA", "label": "Turbo 加速 LoRA 开关"}),
            "turbo_lora_name": (loras, {"default": default_lora, "label": "Turbo 加速 LoRA 模型"}),
            "turbo_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "label": "Turbo LoRA 强度"}),
        }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "视频 VAE", "音频 VAE")
    FUNCTION = "load_all"
    CATEGORY = "Ref2VA/loaders"

    def load_all(self, unet_name, weight_dtype, clip_name, clip_type, video_vae_name, audio_vae_name, enable_turbo_lora=True, turbo_lora_name=None, turbo_lora_strength=1.0):
        # Reuse ComfyUI's official loaders so dtype handling, device policy and
        # model reload factories remain identical to the original four nodes.
        import nodes
        model = nodes.UNETLoader().load_unet(unet_name, weight_dtype)[0]
        if enable_turbo_lora:
            if not turbo_lora_name:
                raise ValueError("已开启 Turbo 加速 LoRA，但没有选择 LoRA 模型。")
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, turbo_lora_name, float(turbo_lora_strength))[0]
        clip = nodes.CLIPLoader().load_clip(clip_name, clip_type, "default")[0]
        video_vae = nodes.VAELoader().load_vae(video_vae_name)[0]
        audio_vae = nodes.VAELoader().load_vae(audio_vae_name)[0]
        return (model, clip, video_vae, audio_vae)

class Ref2VAGenerationConditioning(io.ComfyNode):
    """Unified T2V / FL2V conditioning node for MiniMax H3."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VAGenerationConditioning",
            display_name="Ref2VA：文生视频 / 首尾帧生视频",
            category="Ref2VA/conditioning",
            inputs=[
                io.Clip.Input("clip", display_name="CLIP"),
                io.Vae.Input("vae", display_name="视频 VAE"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, display_name="提示词"),
                io.Combo.Input("generation_mode", options=["文生视频（T2V）", "首尾帧生视频（FL2V）"], default="文生视频（T2V）", display_name="生成模式"),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32, display_name="宽度"),
                io.Int.Input("height", default=768, min=32, max=8192, step=32, display_name="高度"),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, display_name="帧数 / 时长"),
                io.Image.Input("first_frame", optional=True, display_name="首帧（FL2V）"),
                io.Image.Input("last_frame", optional=True, display_name="尾帧（FL2V）"),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="AV Latent")],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, generation_mode, width, height, length, first_frame=None, last_frame=None):
        if generation_mode == "首尾帧生视频（FL2V）" and first_frame is None and last_frame is None:
            raise ValueError("FL2V 模式至少需要首帧或尾帧中的一张图片。")
        if generation_mode != "首尾帧生视频（FL2V）":
            first_frame = None
            last_frame = None
        return MiniMaxH3ImageToVideo.execute(clip, vae, prompt, int(width), int(height), int(length), first_frame, last_frame)

def _enabled(value):
    if isinstance(value,str):return value.strip().lower() not in {"", "0", "false", "no", "off", "关闭", "跳过"}
    return bool(value)

class Ref2VAImageGate:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"enabled":("BOOLEAN",{"default":True}),"image":("IMAGE",)}}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("image",); FUNCTION="gate"; CATEGORY="Ref2VA/controls"
    def gate(self,enabled,image): return (image if enabled else None,)

class Ref2VAAudioGate:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{"enabled":("BOOLEAN",{"default":True}),"audio":("AUDIO",)}}
    RETURN_TYPES=("AUDIO",); RETURN_NAMES=("audio",); FUNCTION="gate"; CATEGORY="Ref2VA/controls"
    def gate(self,enabled,audio): return (audio if enabled else None,)

class Ref2VARegionSwitchboard:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "enable_images":("BOOLEAN",{"default":True,"label_on":"使用图片区","label_off":"忽略图片区"}),
        "enable_videos":("BOOLEAN",{"default":True,"label_on":"使用视频区","label_off":"忽略视频区"}),
        "enable_audios":("BOOLEAN",{"default":True,"label_on":"使用音频区","label_off":"忽略音频区"}),}}
    RETURN_TYPES=("BOOLEAN","BOOLEAN","BOOLEAN"); RETURN_NAMES=("图片区域开关","视频区域开关","音频区域开关"); FUNCTION="controls"; CATEGORY="Ref2VA/controls"
    def controls(self,enable_images,enable_videos,enable_audios): return (enable_images,enable_videos,enable_audios)

class Ref2VAWithGlobalGates(MiniMaxH3ReferenceToVideo):
    @classmethod
    def define_schema(cls):
        base=MiniMaxH3ReferenceToVideo.define_schema()
        controls=[io.Boolean.Input("enable_images",display_name="图片参考开关",default=True),io.Boolean.Input("enable_videos",display_name="视频参考开关",default=True),io.Boolean.Input("enable_audios",display_name="音频参考开关",default=True)]
        return io.Schema(node_id="Ref2VAWithGlobalGates",display_name="Ref2VA：多参考图 / 视频 / 音频（总开关）",category="model/conditioning/minimax",inputs=base.inputs[:8]+controls+base.inputs[8:],outputs=base.outputs)
    @classmethod
    def execute(cls,clip,vae,audio_vae,prompt,width,height,length,ref_image_size="match",enable_images=True,enable_videos=True,enable_audios=True,ref_images=None,ref_videos=None,ref_video_audios=None,ref_audios=None):
        return MiniMaxH3ReferenceToVideo.execute(clip,vae,audio_vae,prompt,width,height,length,ref_image_size,ref_images=ref_images if enable_images else None,ref_videos=ref_videos if enable_videos else None,ref_video_audios=ref_video_audios if enable_videos else None,ref_audios=ref_audios if enable_audios else None)

class Ref2VAAllInOne(MiniMaxH3ReferenceToVideo):
    """Ref2VA with duration and canvas controls built into the reference node."""
    @classmethod
    def define_schema(cls):
        base=MiniMaxH3ReferenceToVideo.define_schema()
        generation_controls=[
            io.Combo.Input(
                "generation_mode",
                options=[
                    "多参考生视频（Ref2VA）",
                    "文生视频（T2V）",
                    "图生视频（I2V）",
                    "首尾帧生视频（FL2V）",
                ],
                default="多参考生视频（Ref2VA）",
                display_name="生成模式（不拆线）",
            ),
            io.Combo.Input("aspect_ratio",options=["16:9 (宽屏)","9:16 (竖屏)","1:1 (方形)","4:3","3:4"],default="16:9 (宽屏)",display_name="生成比例"),
            io.Float.Input("megapixels",default=1.0,min=0.2,max=4.0,step=0.1,display_name="生成百万像素"),
            io.Float.Input("duration_seconds",default=5.0,min=0.2,max=150.0,step=0.1,display_name="视频时长（秒）"),
            io.Image.Input("first_frame",optional=True,display_name="首帧（FL2V）"),
            io.Image.Input("last_frame",optional=True,display_name="尾帧（FL2V）"),
        ]
        return io.Schema(node_id="Ref2VAAllInOne",display_name="Ref2VA：多参考图 / 视频 / 音频（集成时长与分辨率）",category="model/conditioning/minimax",inputs=base.inputs[:4]+generation_controls+base.inputs[7:],outputs=base.outputs)
    @classmethod
    def execute(cls,clip,vae,audio_vae,prompt,generation_mode,aspect_ratio,megapixels,duration_seconds,first_frame=None,last_frame=None,ref_image_size="match",ref_images=None,ref_videos=None,ref_video_audios=None,ref_audios=None):
        selector=str(aspect_ratio).split(" ",1)[0]
        width,height=resolution_from_selector(selector,float(megapixels))
        raw=max(5,round(float(duration_seconds)*24))
        length=raw+(5-(raw%17))%17
        # Keep every socket permanently present in the graph.  The chosen mode
        # decides which connected inputs are used; switching modes must never
        # remove a cable from the user's canvas.
        mode = str(generation_mode)
        if mode == "多参考生视频（Ref2VA）":
            # In Ref2VA mode this optional input is the previous shot's stored
            # tail frame.  Keep the user's other visual references and insert
            # that tail first so the next shot has a concrete continuity anchor.
            if first_frame is not None:
                ref_images = first_frame if ref_images is None else torch.cat((first_frame, ref_images), dim=0)
            return MiniMaxH3ReferenceToVideo.execute(clip,vae,audio_vae,prompt,width,height,length,ref_image_size,ref_images=ref_images,ref_videos=ref_videos,ref_video_audios=ref_video_audios,ref_audios=ref_audios)
        # Compatibility for workflows saved before v0.3.  The old mixed label
        # had no explicit T2V/I2V distinction, so infer the safe operation from
        # the actually supplied frames rather than rejecting a formerly valid
        # document after an update.
        if mode == "文生视频 / 首尾帧生视频（FL2V）":
            mode = "首尾帧生视频（FL2V）" if last_frame is not None else ("图生视频（I2V）" if first_frame is not None else "文生视频（T2V）")
        if mode == "首尾帧生视频（FL2V）":
            if first_frame is None and last_frame is None:
                raise ValueError("首尾帧生视频（FL2V）至少需要首帧或尾帧中的一张图片。连续分镜自动衔接请使用“多参考生视频（Ref2VA）”。")
            return MiniMaxH3ImageToVideo.execute(clip,vae,prompt,width,height,length,first_frame,last_frame)
        if mode == "图生视频（I2V）":
            if first_frame is None:
                raise ValueError("图生视频（I2V）必须连接首帧图片。")
            return MiniMaxH3ImageToVideo.execute(clip,vae,prompt,width,height,length,first_frame,None)
        if mode == "文生视频（T2V）":
            return MiniMaxH3ImageToVideo.execute(clip,vae,prompt,width,height,length,None,None)
        raise ValueError(f"未知生成模式：{generation_mode}")

class Ref2VAStoryboardPromptPanel(io.ComfyNode):
    """A same-canvas 3-shot planning panel that feeds the active Ref2VA/FL2V prompt.

    It intentionally has no automatic video-to-next-frame loop: ComfyUI cannot
    execute a graph cycle safely.  After a shot is approved, the user uses its
    exported final tail as the next shot's FL2V first frame on the next queue.
    The prompt connection is real, so the selected shot changes the text that
    reaches Ref2VAAllInOne rather than merely changing a label.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VAStoryboardPromptPanel",
            display_name="Ref2VA：连续分镜控制台（3段）",
            category="Ref2VA/storyboard",
            inputs=[
                io.Combo.Input("current_shot", options=["镜头 1：建立", "镜头 2：承接", "镜头 3：收束"], default="镜头 1：建立", display_name="当前要生成的镜头"),
                io.String.Input("global_anchor", multiline=True, default="全局锚点：角色外观、服装、道具、场景、光影和风格在所有镜头中保持一致。", display_name="全局锚点（全部镜头共用）"),
                io.String.Input("shot_1_prompt", multiline=True, default="镜头 1：建立人物、场景与动作起点。", display_name="镜头 1 提示词"),
                io.String.Input("shot_2_prompt", multiline=True, default="镜头 2：承接上一段末帧，只描述新的动作和镜头变化。", display_name="镜头 2 提示词"),
                io.String.Input("shot_3_prompt", multiline=True, default="镜头 3：承接上一段末帧，完成动作并收束画面。", display_name="镜头 3 提示词"),
                io.String.Input("negative_prompt", multiline=True, default="人物变脸、换装、换武器、场景跳变、文字、水印、闪烁、肢体畸形", display_name="连续性负面约束"),
            ],
            outputs=[
                io.String.Output(display_name="当前镜头提示词"),
                io.String.Output(display_name="当前镜头名称"),
            ],
        )

    @classmethod
    def execute(cls, current_shot, global_anchor, shot_1_prompt, shot_2_prompt, shot_3_prompt, negative_prompt):
        shot_text = {
            "镜头 1：建立": shot_1_prompt,
            "镜头 2：承接": shot_2_prompt,
            "镜头 3：收束": shot_3_prompt,
        }.get(current_shot, shot_1_prompt)
        number = "1" if str(current_shot).startswith("镜头 1") else "2" if str(current_shot).startswith("镜头 2") else "3"
        handoff = "本镜头从上一镜头最终视频的尾帧开始，首帧必须承接上一镜头的角色位置、服装、道具、景别与光线。" if number != "1" else "本镜头建立后续镜头必须继承的人物、服装、道具、场景与光线。"
        prompt = f"{global_anchor.strip()}\n\n[当前镜头 {number}]\n{shot_text.strip()}\n\n[连续性要求]\n{handoff}\n\n[负面约束]\n{negative_prompt.strip()}"
        return io.NodeOutput(prompt, current_shot)


class Ref2VAStoryboardVideoArchive(io.ComfyNode):
    """Save each approved storyboard pass into its real, separate shot folder."""
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VAStoryboardVideoArchive",
            display_name="Ref2VA：连续分镜视频归档",
            category="Ref2VA/storyboard",
            is_output_node=True,
            inputs=[
                io.Video.Input("video", display_name="当前镜头视频"),
                io.String.Input("current_shot", default="镜头 1：建立", display_name="当前镜头名称"),
            ],
            outputs=[io.Video.Output(display_name="已归档视频")],
        )

    @classmethod
    def execute(cls, video, current_shot="镜头 1：建立"):
        shot_dir = _storyboard_shot_dir(current_shot)
        components = video.get_components()
        images = components.images
        _, height, width, _ = images.shape
        prefix = f"video/连续分镜/{shot_dir}"
        out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), int(width), int(height)
        )
        file = f"{filename}_{counter:05}_.mp4"
        full_path = os.path.join(out_dir, file)
        video.save_to(full_path, format=Types.VideoContainer("mp4"), codec="auto", metadata=None)
        return io.NodeOutput(
            video,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )


class Ref2VAStoryboardAutoArchive(io.ComfyNode):
    """Choose the real usable shot video, then archive it without manual rewiring.

    A disabled final-video branch returns ``None`` from the combined sampling
    panel.  The selector therefore uses the final video only when it really
    exists; otherwise it preserves the first-pass video for fast continuity
    tests.  This makes the archive/continuation path match the user's actual
    output mode rather than a label or a manually moved cable.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VAStoryboardAutoArchive",
            display_name="Ref2VA：连续分镜自动成片归档",
            category="Ref2VA/storyboard",
            is_output_node=True,
            inputs=[
                io.Video.Input("initial_video", display_name="最初视频（测试模式）"),
                io.Video.Input("final_video", display_name="最终视频（二采/Latent）"),
                io.String.Input("current_shot", default="镜头 1：建立", display_name="当前镜头名称"),
            ],
            outputs=[io.Video.Output(display_name="当前镜头成片（自动选择）")],
        )

    @classmethod
    def execute(cls, initial_video=None, final_video=None, current_shot="镜头 1：建立"):
        # Final is available only when the final-video branch was enabled.
        # Falling back to the first pass is deliberate: it keeps a fast test
        # run usable as the next shot's continuity source.
        selected = final_video if final_video is not None else initial_video
        if selected is None:
            raise ValueError("没有可归档的视频：请至少开启“最初视频”或“最终视频”其中一项。")
        shot_dir = _storyboard_shot_dir(current_shot)
        components = selected.get_components()
        images = components.images
        _, height, width, _ = images.shape
        prefix = f"video/连续分镜/{shot_dir}/原始成片"
        out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), int(width), int(height)
        )
        file = f"{filename}_{counter:05}_.mp4"
        selected.save_to(
            os.path.join(out_dir, file),
            format=Types.VideoContainer("mp4"), codec="auto", metadata=None,
        )
        # A deterministic PNG is intentionally stored alongside the accepted
        # clip. The next shot can load it without a graph cycle or a second
        # manual upload, while a user-selected video still overrides it.
        _save_tail_frame(images, os.path.join(out_dir, "尾帧.png"))
        return io.NodeOutput(
            selected,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )


class Ref2VAStoryboardTailFrameUpload:
    """Automatic previous-tail reader with an optional manual video override.

    The implementation deliberately delegates file decoding to the user's
    installed VideoHelperSuite loader.  It therefore keeps the same upload,
    video preview, frame-rate and resize behaviour while removing the separate
    Select Images node and its manual ``-1`` configuration.
    """
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        video_extensions = {"mp4", "mov", "mkv", "avi", "webm", "m4v", "gif", "webp", "avif"}
        files = []
        if os.path.isdir(input_dir):
            for name in os.listdir(input_dir):
                path = os.path.join(input_dir, name)
                if os.path.isfile(path) and name.rsplit(".", 1)[-1].lower() in video_extensions:
                    files.append(name)
        return {"required": {
            # An explicit empty choice prevents a previously uploaded file from
            # silently becoming the first shot's FL2V frame.
            "video": ([""] + sorted(files), {"default": "", "label": "上一镜头视频"}),
            "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 60, "step": 1, "label": "帧率覆盖（0=原视频）"}),
            "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "label": "自定义宽度（0=原始）"}),
            "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "label": "自定义高度（0=原始）"}),
            "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1000000, "label": "读取帧上限（0=全部）"}),
            "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "label": "跳过开头帧"}),
            "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000000, "label": "每 N 帧读取一次"}),
            "current_shot": ("STRING", {"default": "镜头 1：建立", "label": "当前镜头名称"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("上一镜头最后一帧",)
    FUNCTION = "load_tail_frame"
    CATEGORY = "Ref2VA/storyboard"
    DESCRIPTION = "上传上一镜头视频后自动取最后一帧，直接连接 FL2V 的首帧输入。"

    def load_tail_frame(self, video, force_rate=0, custom_width=0, custom_height=0,
                        frame_load_cap=0, skip_first_frames=0, select_every_nth=1,
                        current_shot="镜头 1：建立"):
        # The first shot deliberately has no preceding clip.  Return an empty
        # optional first-frame value without trying to validate or decode a
        # stale upload left in the widget from an earlier continuation run.
        if _storyboard_shot_index(current_shot) <= 1:
            return (None,)
        # Manual selection has priority, which lets the user intentionally
        # continue from a different take. Without it, use the tail cache made
        # by Ref2VAStoryboardAutoArchive after the previous approved shot.
        if not video:
            tail_path = _storyboard_tail_path(current_shot)
            if tail_path and os.path.isfile(tail_path):
                return (_image_tensor_from_path(tail_path),)
            previous = _storyboard_shot_index(current_shot) - 1
            raise RuntimeError(
                f"镜头 {_storyboard_shot_index(current_shot)} 需要上一镜头尾帧，但未找到镜头 {previous} 的自动归档尾帧。"
                "请先生成并归档上一镜头，或手动选择上一段视频。"
            )
        loader_class = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideo")
        if loader_class is None:
            raise RuntimeError("未找到 VideoHelperSuite 的 VHS_LoadVideo，无法读取上一镜头视频。")
        images, _, _, _ = loader_class().load_video(
            video=video,
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
            format="None",
        )
        if images is None or len(images) == 0:
            raise RuntimeError("上一镜头视频没有可用画面，无法提取最后一帧。")
        return (images[-1:].contiguous(),)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        # The shot selector is part of the result: shot 1 intentionally emits
        # no frame, later shots read the manual clip or automatic tail cache.
        # Include the cache timestamp so newly generated tails cannot be lost
        # to ComfyUI's result cache.
        current_shot = kwargs.get("current_shot", "镜头 1：建立")
        if not video:
            tail_path = _storyboard_tail_path(current_shot)
            tail_stamp = os.path.getmtime(tail_path) if tail_path and os.path.isfile(tail_path) else None
            return ("Ref2VA-tail-frame", current_shot, tail_path, tail_stamp)
        loader_class = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideo")
        if loader_class is not None and hasattr(loader_class, "IS_CHANGED"):
            return ("Ref2VA-tail-frame", current_shot, loader_class.IS_CHANGED(video, **kwargs))
        return ("Ref2VA-tail-frame", current_shot, video)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not video:
            return True
        loader_class = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideo")
        if loader_class is not None and hasattr(loader_class, "VALIDATE_INPUTS"):
            return loader_class.VALIDATE_INPUTS(video)
        return True

class Ref2VADecodeCreateVideo(io.ComfyNode):
    """Decode H3's joint AV latent and immediately package it as a VIDEO."""
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Ref2VADecodeCreateVideo",display_name="Ref2VA：解码并创建视频",category="Ref2VA/output",inputs=[
            io.Latent.Input("av_latent",display_name="采样 Latent"),io.Vae.Input("video_vae",display_name="视频 VAE"),io.Vae.Input("audio_vae",display_name="音频 VAE"),
            io.Float.Input("fps",default=24.0,min=1.0,max=120.0,step=1.0,display_name="帧率"),io.Int.Input("bit_depth",default=8,min=8,max=10,step=2,display_name="位深"),
        ],outputs=[io.Video.Output(display_name="视频")])
    @classmethod
    def execute(cls,av_latent,video_vae,audio_vae,fps=24.0,bit_depth=8):
        samples=av_latent.get("samples") if isinstance(av_latent,dict) else None
        if not getattr(samples,"is_nested",False):raise ValueError("Ref2VA 解码需要 MiniMax H3 的音画联合 Latent。")
        video,audio=tuple(samples.unbind())
        images=video_vae.decode(video)
        if images.ndim==5:images=images.reshape(-1,*images.shape[-3:])
        generated_audio=vae_decode_audio(audio_vae,{"samples":audio})
        created=CreateVideo.execute(images=images,audio=generated_audio,fps=float(fps),bit_depth=int(bit_depth))
        video_output=created.args[0] if hasattr(created,"args") else created[0]
        return io.NodeOutput(video_output)

class Ref2VAPreviewVideo(io.ComfyNode):
    """Pass through a VIDEO and emit an immediate ComfyUI video preview."""
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Ref2VAPreviewVideo",display_name="Ref2VA：实时视频预览",category="Ref2VA/output",is_output_node=True,inputs=[
            io.Video.Input("video",display_name="视频"),
            io.String.Input("filename_prefix",default="video/Ref2VA_Preview",display_name="预览缓存前缀"),
        ],outputs=[io.Video.Output(display_name="视频")])
    @classmethod
    def execute(cls,video,filename_prefix="video/Ref2VA_Preview"):
        # Reuse ComfyUI's native video writer so the result is playable in the
        # node immediately after execution while preserving the VIDEO stream.
        return SaveVideo.execute(video,filename_prefix,"mp4",{"codec":"auto"})

def _refine_sigmas(sigmas,extra,start,spacing):
    if extra<=0:return sigmas
    cpu=sigmas.detach().cpu(); idx=next((i for i,v in enumerate(cpu) if v<=start),-1)
    if idx==-1 or idx>=len(cpu)-1:return sigmas
    a,b=cpu[idx].item(),cpu[-1].item(); t=torch.linspace(0,1,steps=len(cpu)-idx+extra)
    if spacing=="cosine":f=(1-torch.cos(t*math.pi))/2
    elif spacing=="exponential":f=(torch.exp(t*3)-1)/(math.exp(3)-1)
    else:f=t
    return torch.cat([cpu[:idx],a+(b-a)*f]).to(device=sigmas.device,dtype=sigmas.dtype)

def _run_pass(noise,guider,sampler,sigmas,latent,index):
    work=latent.copy(); samples_in=comfy.sample.fix_empty_latent_channels(guider.model_patcher,work["samples"],work.get("downscale_ratio_spacial"),work.get("downscale_ratio_temporal")); work["samples"]=samples_in
    seed=getattr(noise,"seed",None)
    if seed is None: noise_tensor=noise.generate_noise(work); sample_seed=None
    else: noise_tensor=comfy.sample.prepare_noise(samples_in,seed+index,work.get("batch_index")); sample_seed=seed+index
    x0={}; callback=latent_preview.prepare_callback(guider.model_patcher,sigmas.shape[-1]-1,x0)
    samples=guider.sample(noise_tensor,samples_in,sampler,sigmas,denoise_mask=work.get("noise_mask"),callback=callback,disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,seed=sample_seed).to(comfy.model_management.intermediate_device())
    out=work.copy(); out.pop("downscale_ratio_spacial",None); out.pop("downscale_ratio_temporal",None); out["samples"]=samples
    if "x0" not in x0:return out,out
    x=x0["x0"]
    if samples.is_nested and not x.is_nested:x=comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x,[i.shape for i in samples.unbind()]))
    den=out.copy(); den["samples"]=guider.model_patcher.model.process_latent_out(x.cpu()); return out,den

def _refine_only(noise,guider,sampler,latent,s,count):
    rs=s["refine_steps"]; total=max(rs,int(rs/s["refine_denoise"]))
    sigmas=comfy.samplers.calculate_sigmas(guider.model_patcher.get_model_object("model_sampling"),s["refine_scheduler"],total).cpu()[-(rs+1):]
    sigmas=_refine_sigmas(sigmas,s["refine_extra_steps"],s["refine_start_at_sigma"],s["refine_spacing"])
    out=latent; den=latent
    for index in range(1,count+1):out,den=_run_pass(noise,guider,sampler,sigmas,out,index)
    return out,den

def _multi_pass(noise,guider,sampler,main_sigmas,latent,s):
    out,den=_run_pass(noise,guider,sampler,main_sigmas,latent,0)
    if s["passes"]<=1:return out,den
    return _refine_only(noise,guider,sampler,out,s,s["passes"]-1)

def _unpack_node_output(value):
    """Normalise Comfy node outputs without assuming a fixed wrapper type."""
    if hasattr(value,"args") and value.args:return value.args
    if isinstance(value,(tuple,list)):return value
    return (value,)

def _split_h3_av_latent(latent):
    """Separate the H3 joint video/audio latent before video-only H3 upscale."""
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent
    video,audio=_unpack_node_output(LTXVSeparateAVLatent.execute(latent))[:2]
    return video,audio

def _join_h3_av_latent(video_latent,audio_latent,template):
    """Rebuild the joint H3 latent, preserving the sampled audio stream unchanged."""
    output=dict(template)
    output.pop("noise_mask",None)
    video_samples=video_latent.get("samples") if isinstance(video_latent,dict) else video_latent
    audio_samples=audio_latent.get("samples") if isinstance(audio_latent,dict) else audio_latent
    try:
        from comfy_extras.nodes_lt import LTXVConcatAVLatent
        packed=_unpack_node_output(LTXVConcatAVLatent.execute(video_latent,audio_latent))[0]
        if isinstance(packed,dict) and "samples" in packed:return packed
        output["samples"]=packed
        return output
    except Exception:
        import comfy.nested_tensor
        output["samples"]=comfy.nested_tensor.NestedTensor((video_samples,audio_samples))
        return output

class _BasicGuider(comfy.samplers.CFGGuider):
    def set_conds(self,positive):self.inner_set_conds({"positive":positive})
class _SeedNoise:
    def __init__(self,seed):self.seed=seed

class Ref2VASamplingControlPanel:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "noise_seed":("INT",{"default":0,"min":0,"max":0xffffffffffffffff,"control_after_generate":True,"label":"随机种子"}),
        "scheduler":(SCHEDULERS,{"default":"beta","label":"主采样调度器"}),"steps":("INT",{"default":4,"min":1,"max":100,"label":"主采样步数"}),"denoise":("FLOAT",{"default":1.0,"min":0.01,"max":1.0,"step":0.01,"label":"主采样降噪"}),"sampler_name":(list(comfy.samplers.SAMPLER_NAMES),{"default":"euler","label":"采样器"}),
        "main_extra_steps":("INT",{"default":1,"min":0,"max":15,"label":"主采样 Sigma 加步"}),"main_start_at_sigma":("FLOAT",{"default":0.70,"min":0,"max":20,"step":0.01,"label":"主采样 Sigma 阈值"}),"main_spacing":(SPACINGS,{"default":"cosine","label":"主采样 Sigma 曲线"}),
        "second_sampling_mode":(["同分辨率细化","H3 Latent 超分"],{"default":"同分辨率细化","label":"二次采样模式"}),"latent_upscale_model":(LATENT_UPSCALE_MODELS,{"default":DEFAULT_LATENT_UPSCALE_MODEL,"label":"H3 Latent 超分模型"}),"second_aspect_ratio":(["16:9"],{"default":"16:9","label":"二次输出比例"}),"second_megapixels":([1.0,1.5,2.0],{"default":2.0,"label":"二次输出百万像素"}),"upscale_passes":("INT",{"default":1,"min":1,"max":2,"label":"超分后细化次数"}),
        "passes":("INT",{"default":2,"min":1,"max":3,"label":"采样次数"}),"refine_steps":("INT",{"default":3,"min":1,"max":12,"label":"每遍细化步数"}),"refine_denoise":("FLOAT",{"default":0.30,"min":0.01,"max":0.80,"step":0.01,"label":"细化降噪"}),"refine_scheduler":(SCHEDULERS,{"default":"beta","label":"细化调度器"}),"refine_extra_steps":("INT",{"default":1,"min":0,"max":15,"label":"细化 Sigma 加步"}),"refine_start_at_sigma":("FLOAT",{"default":0.60,"min":0,"max":20,"step":0.01,"label":"细化 Sigma 阈值"}),"refine_spacing":(SPACINGS,{"default":"cosine","label":"细化 Sigma 曲线"}),}}
    RETURN_TYPES=("REF2VA_SAMPLING_SETTINGS",); RETURN_NAMES=("采样设置",); FUNCTION="build"; CATEGORY="Ref2VA/sampling"
    def build(self,**settings):return (settings,)

class Ref2VAPrimarySamplingPanel:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "noise_seed":("INT",{"default":0,"min":0,"max":0xffffffffffffffff,"control_after_generate":True,"label":"随机种子"}),
        "scheduler":(SCHEDULERS,{"default":"beta","label":"主采样调度器"}),"steps":("INT",{"default":4,"min":1,"max":100,"label":"主采样步数"}),"denoise":("FLOAT",{"default":1.0,"min":0.01,"max":1.0,"step":0.01,"label":"主采样降噪"}),"sampler_name":(list(comfy.samplers.SAMPLER_NAMES),{"default":"euler","label":"采样器"}),"main_extra_steps":("INT",{"default":1,"min":0,"max":15,"label":"主采样 Sigma 加步"}),"main_start_at_sigma":("FLOAT",{"default":0.70,"min":0,"max":20,"step":0.01,"label":"主采样 Sigma 阈值"}),"main_spacing":(SPACINGS,{"default":"cosine","label":"主采样 Sigma 曲线"}),}}
    RETURN_TYPES=("REF2VA_PRIMARY_SETTINGS",);RETURN_NAMES=("第一次采样设置",);FUNCTION="build";CATEGORY="Ref2VA/sampling"
    def build(self,**settings):return (settings,)

class Ref2VARefinementPanel:
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "second_sampling_mode":(["同分辨率细化","H3 Latent 超分"],{"default":"同分辨率细化","label":"二次采样模式"}),"latent_upscale_model":(LATENT_UPSCALE_MODELS,{"default":DEFAULT_LATENT_UPSCALE_MODEL,"label":"H3 Latent 超分模型"}),"second_aspect_ratio":(["16:9"],{"default":"16:9","label":"二次输出比例"}),"second_megapixels":([1.0,1.5,2.0],{"default":2.0,"label":"二次输出百万像素"}),"upscale_passes":("INT",{"default":1,"min":1,"max":2,"label":"超分后细化次数"}),"passes":("INT",{"default":1,"min":1,"max":3,"label":"二次采样次数"}),"refine_scheduler":(SCHEDULERS,{"default":"beta","label":"细化调度器"}),"refine_steps":("INT",{"default":3,"min":1,"max":12,"label":"每遍细化步数"}),"refine_denoise":("FLOAT",{"default":0.30,"min":0.01,"max":0.80,"step":0.01,"label":"细化降噪"}),"refine_extra_steps":("INT",{"default":1,"min":0,"max":15,"label":"细化 Sigma 加步"}),"refine_start_at_sigma":("FLOAT",{"default":0.60,"min":0,"max":20,"step":0.01,"label":"细化 Sigma 阈值"}),"refine_spacing":(SPACINGS,{"default":"cosine","label":"细化 Sigma 曲线"}),}}
    RETURN_TYPES=("REF2VA_REFINEMENT_SETTINGS",);RETURN_NAMES=("第二次/N次采样设置",);FUNCTION="build";CATEGORY="Ref2VA/sampling"
    def build(self,**settings):
        return (settings,)

class Ref2VASameResolutionRefinementPanel:
    """A real, dedicated template for second-pass same-resolution refinement.

    This deliberately has no latent-upscale inputs.  It replaces the brittle
    frontend-only hide/show template used by the legacy combined panel.
    """
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "passes":("INT",{"default":1,"min":1,"max":3,"label":"二次采样次数"}),
        "refine_scheduler":(SCHEDULERS,{"default":"beta","label":"细化调度器"}),
        "refine_steps":("INT",{"default":3,"min":1,"max":12,"label":"每遍细化步数"}),
        "refine_denoise":("FLOAT",{"default":0.30,"min":0.01,"max":0.80,"step":0.01,"label":"细化降噪"}),
        "refine_extra_steps":("INT",{"default":1,"min":0,"max":15,"label":"细化 Sigma 加步"}),
        "refine_start_at_sigma":("FLOAT",{"default":0.60,"min":0,"max":20,"step":0.01,"label":"细化 Sigma 阈值"}),
        "refine_spacing":(SPACINGS,{"default":"cosine","label":"细化 Sigma 曲线"}),
    }}
    RETURN_TYPES=("REF2VA_REFINEMENT_SETTINGS",); RETURN_NAMES=("同分辨率细化设置",); FUNCTION="build"; CATEGORY="Ref2VA/sampling"
    def build(self, **settings):
        return ({"second_sampling_mode":"同分辨率细化", **settings},)

class Ref2VAH3LatentRefinementPanel:
    """A real, dedicated template for H3 learned latent upscaling."""
    @classmethod
    def INPUT_TYPES(cls): return {"required":{
        "latent_upscale_model":(LATENT_UPSCALE_MODELS,{"default":DEFAULT_LATENT_UPSCALE_MODEL,"label":"H3 Latent 超分模型"}),
        "second_aspect_ratio":(["16:9"],{"default":"16:9","label":"二次输出比例"}),
        "second_megapixels":([1.0,1.5,2.0],{"default":1.5,"label":"二次输出百万像素"}),
        "upscale_passes":("INT",{"default":1,"min":1,"max":2,"label":"超分后细化次数"}),
        "passes":("INT",{"default":1,"min":1,"max":3,"label":"二次采样次数"}),
        "refine_scheduler":(SCHEDULERS,{"default":"beta","label":"细化调度器"}),
        "refine_steps":("INT",{"default":3,"min":1,"max":12,"label":"每遍细化步数"}),
        "refine_denoise":("FLOAT",{"default":0.30,"min":0.01,"max":0.80,"step":0.01,"label":"细化降噪"}),
        "refine_extra_steps":("INT",{"default":1,"min":0,"max":15,"label":"细化 Sigma 加步"}),
        "refine_start_at_sigma":("FLOAT",{"default":0.60,"min":0,"max":20,"step":0.01,"label":"细化 Sigma 阈值"}),
        "refine_spacing":(SPACINGS,{"default":"cosine","label":"细化 Sigma 曲线"}),
    }}
    RETURN_TYPES=("REF2VA_REFINEMENT_SETTINGS",); RETURN_NAMES=("H3 Latent 超分设置",); FUNCTION="build"; CATEGORY="Ref2VA/sampling"
    def build(self, **settings):
        return ({"second_sampling_mode":"H3 Latent 超分", **settings},)

class Ref2VARefinementModeSelector:
    """Stable route selector between the two real refinement templates."""
    @classmethod
    def INPUT_TYPES(cls): return {
        "required":{"second_sampling_mode":(["同分辨率细化","H3 Latent 超分"],{"default":"同分辨率细化","label":"二次采样模式"})},
        "optional":{"same_resolution_settings":("REF2VA_REFINEMENT_SETTINGS",),"latent_settings":("REF2VA_REFINEMENT_SETTINGS",)},
    }
    RETURN_TYPES=("REF2VA_REFINEMENT_SETTINGS",); RETURN_NAMES=("当前二次采样设置",); FUNCTION="select"; CATEGORY="Ref2VA/sampling"
    def select(self, second_sampling_mode, same_resolution_settings=None, latent_settings=None):
        selected = latent_settings if second_sampling_mode == "H3 Latent 超分" else same_resolution_settings
        if selected is None:
            requested = "H3 Latent 超分模板" if second_sampling_mode == "H3 Latent 超分" else "同分辨率细化模板"
            raise ValueError(f"当前选择了{requested}，但该模板没有连接到二次采样模式选择器。")
        return (selected,)

class Ref2VAUnifiedMultiPassSampler:
    @classmethod
    def INPUT_TYPES(cls):return {
        "required":{"model":("MODEL",),"conditioning":("CONDITIONING",),"latent_image":("LATENT",),"primary_settings":("REF2VA_PRIMARY_SETTINGS",)},
        "optional":{"refinement_settings":("REF2VA_REFINEMENT_SETTINGS",)},
    }
    RETURN_TYPES=("LATENT","LATENT","LATENT"); RETURN_NAMES=("最后一次采样","最后一次降噪Latent","第一次采样（对比预览）"); FUNCTION="sample_unified"; CATEGORY="Ref2VA/sampling"; DESCRIPTION="Ref2VA 采样执行器：输出最终采样，以及可单独解码预览的第一次采样。"
    def sample_unified(self,model,conditioning,latent_image,primary_settings,refinement_settings=None):
        # The second-pass panel is deliberately optional: bypassing its group
        # means "first pass only", rather than leaving a required socket empty.
        has_refinement_settings=refinement_settings is not None
        primary_settings=primary_settings or {}
        refinement_settings=refinement_settings or {"passes":1}
        settings={**primary_settings,**refinement_settings}
        settings.setdefault("second_sampling_mode","同分辨率细化")
        settings.setdefault("passes",1)
        settings.setdefault("refine_steps",3)
        settings.setdefault("refine_denoise",0.30)
        settings.setdefault("refine_scheduler","beta")
        settings.setdefault("refine_extra_steps",1)
        settings.setdefault("refine_start_at_sigma",0.60)
        settings.setdefault("refine_spacing","cosine")
        # Safety guard against unstable refinement combinations.  These
        # limits only affect the refinement pass; the first pass is untouched.
        raw_steps=int(settings.get("refine_steps",3)); raw_denoise=float(settings.get("refine_denoise",0.30))
        safe_steps=max(3,min(raw_steps,12)); safe_denoise=max(0.05,min(raw_denoise,0.40))
        if safe_steps!=raw_steps or safe_denoise!=raw_denoise:
            print(f"[Ref2VA] 二次采样参数保护：细化步数 {raw_steps}->{safe_steps}，细化降噪 {raw_denoise:.2f}->{safe_denoise:.2f}")
        settings["refine_steps"]=safe_steps; settings["refine_denoise"]=safe_denoise
        settings["passes"]=max(1,min(int(settings.get("passes",1)),3))
        settings["upscale_passes"]=max(1,min(int(settings.get("upscale_passes",1)),2))
        guider=_BasicGuider(model); guider.set_conds(conditioning); steps,denoise=settings["steps"],settings["denoise"]; total=max(steps,int(steps/denoise))
        sigmas=comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"),settings["scheduler"],total).cpu()[-(steps+1):]; sigmas=_refine_sigmas(sigmas,settings["main_extra_steps"],settings["main_start_at_sigma"],settings["main_spacing"])
        noise=_SeedNoise(settings["noise_seed"]); sampler=comfy.samplers.sampler_object(settings["sampler_name"])
        refine_guider=_BasicGuider(model); refine_guider.set_conds(conditioning)
        first,first_denoised=_run_pass(noise,guider,sampler,sigmas,latent_image,0)
        # The main pass is always independent.  When the switch is off the
        # workflow skips every second-pass step; when it is on, "passes" is the
        # actual number of additional second-pass sampling runs.
        # The Fast Groups Bypasser natively bypasses the refinement panel.  When
        # it is bypassed, no optional refinement settings arrive here, so the
        # first pass is returned directly.
        if not has_refinement_settings:
            return first,first_denoised,first
        if settings.get("second_sampling_mode","同分辨率细化")!="H3 Latent 超分":
            final,final_denoised=_refine_only(noise,refine_guider,sampler,first,settings,settings["passes"])
            return final,final_denoised,first
        # Ref2VA produces a joint NestedTensor (video + audio).  The H3 learned
        # upscaler accepts only its video member, so keep audio aside and join it
        # back before the subsequent denoising pass.
        video_latent,audio_latent=_split_h3_av_latent(first)
        video_samples=video_latent.get("samples")
        if not isinstance(video_samples,torch.Tensor) or video_samples.ndim!=5:
            raise ValueError("H3 Latent 超分未取得有效的视频 latent；请使用 MiniMax H3 Ref2VA 的音画联合 latent。")
        source_w=int(video_samples.shape[-1])*16; source_h=int(video_samples.shape[-2])*16
        target=resolution_from_selector(settings.get("second_aspect_ratio","16:9"),float(settings.get("second_megapixels",2.0)))
        if target is None: raise ValueError("二次 H3 Latent 超分需要明确的输出比例与百万像素。")
        upscaled_video=upscale_h3_video_latent(video_latent,target_width=target[0],target_height=target[1],source_width=source_w,source_height=source_h,model_name=settings.get("latent_upscale_model",DEFAULT_LATENT_UPSCALE_MODEL))
        if isinstance(upscaled_video,dict):upscaled_video.pop("noise_mask",None)
        if isinstance(audio_latent,dict):
            audio_latent=dict(audio_latent)
            audio_latent.pop("noise_mask",None)
        upscaled=_join_h3_av_latent(upscaled_video,audio_latent,first)
        # H3 mode first performs one latent upscale, then executes the requested
        # number of additional sampling passes.  The advanced H3 detail count
        # multiplies those passes when the user deliberately raises it above 1.
        h3_refine_count=int(settings["passes"])*int(settings.get("upscale_passes",1))
        final,final_denoised=_refine_only(noise,refine_guider,sampler,upscaled,settings,h3_refine_count)
        return final,final_denoised,first

    def refine_from_first(self, model, conditioning, first, primary_settings, refinement_settings):
        """Run only refinement/upscale against a persisted first-pass latent."""
        settings = {**(primary_settings or {}), **(refinement_settings or {})}
        settings.setdefault("second_sampling_mode", "同分辨率细化")
        settings.setdefault("passes", 1)
        settings.setdefault("refine_steps", 3)
        settings.setdefault("refine_denoise", 0.30)
        settings.setdefault("refine_scheduler", "beta")
        settings.setdefault("refine_extra_steps", 1)
        settings.setdefault("refine_start_at_sigma", 0.60)
        settings.setdefault("refine_spacing", "cosine")
        settings["refine_steps"] = max(3, min(int(settings["refine_steps"]), 12))
        settings["refine_denoise"] = max(0.05, min(float(settings["refine_denoise"]), 0.40))
        settings["passes"] = max(1, min(int(settings["passes"]), 3))
        settings["upscale_passes"] = max(1, min(int(settings.get("upscale_passes", 1)), 2))
        noise = _SeedNoise(settings["noise_seed"])
        sampler = comfy.samplers.sampler_object(settings["sampler_name"])
        guider = _BasicGuider(model)
        guider.set_conds(conditioning)
        if settings["second_sampling_mode"] != "H3 Latent 超分":
            return _refine_only(noise, guider, sampler, first, settings, settings["passes"])[0]
        video_latent, audio_latent = _split_h3_av_latent(first)
        video_samples = video_latent.get("samples")
        if not isinstance(video_samples, torch.Tensor) or video_samples.ndim != 5:
            raise ValueError("H3 Latent 超分未取得有效的视频 latent；原版可能来自旧缓存，请重新生成一次原版以建立 latent 缓存。")
        source_w, source_h = int(video_samples.shape[-1]) * 16, int(video_samples.shape[-2]) * 16
        target = resolution_from_selector(settings.get("second_aspect_ratio", "16:9"), float(settings.get("second_megapixels", 2.0)))
        upscaled_video = upscale_h3_video_latent(video_latent, target_width=target[0], target_height=target[1], source_width=source_w, source_height=source_h, model_name=settings.get("latent_upscale_model", DEFAULT_LATENT_UPSCALE_MODEL))
        if isinstance(upscaled_video, dict):
            upscaled_video.pop("noise_mask", None)
        if isinstance(audio_latent, dict):
            audio_latent = dict(audio_latent)
            audio_latent.pop("noise_mask", None)
        upscaled = _join_h3_av_latent(upscaled_video, audio_latent, first)
        count = int(settings["passes"]) * int(settings["upscale_passes"])
        return _refine_only(noise, guider, sampler, upscaled, settings, count)[0]

class Ref2VASamplerVideoOutputPanel:
    """One execution panel for an optional initial video and an optional final video.

    The first pass itself is always calculated because it is the input to an
    enabled refinement pass.  ``enable_initial_video`` only controls the
    independent initial-video branch: when off, the first latent is neither decoded
    nor encoded/saved.  The final VIDEO output remains valid in both modes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "primary_settings": ("REF2VA_PRIMARY_SETTINGS",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "enable_initial_video": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启：主采样 + 最初视频",
                    "label_off": "关闭：不生成最初视频",
                    "label": "最初视频开关",
                }),
                "enable_final_video": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启：二采/Latent + 最终视频 + RTX",
                    "label_off": "关闭：不生成最终视频",
                    "label": "最终视频开关",
                }),
            },
            "optional": {"refinement_settings": ("REF2VA_REFINEMENT_SETTINGS",)},
        }

    RETURN_TYPES = ("VIDEO", "VIDEO")
    RETURN_NAMES = ("最初视频（主采样）", "最终视频（接 RTX / 最终保存）")
    FUNCTION = "sample_decode_output"
    CATEGORY = "Ref2VA/sampling"
    DESCRIPTION = "合并主采样、二采/Latent 与视频解码。最初视频和最终视频各有独立开关；关闭最终视频会一起绕过二采/Latent、最终解码、RTX 与最终保存。"

    def sample_decode_output(self, model, conditioning, latent_image, primary_settings,
                             video_vae, audio_vae, enable_initial_video=True,
                             enable_final_video=True,
                             refinement_settings=None):
        # These are deliberately fixed inside the combined output panel.  Older
        # ComfyUI canvas serializations can restore numeric widget values as NaN
        # after an input-schema update; H3's standard 24 fps / 8-bit packaging
        # remains valid and avoids exposing a broken visual control.
        fps, bit_depth = 24.0, 8
        if not bool(enable_initial_video) and not bool(enable_final_video):
            # Both outputs are disabled: do not start sampling, refinement or
            # decoding merely because this panel is an output node.
            return (None, None)
        # The final-video switch is the single master switch for the full
        # second-pass branch.  Off means no refinement settings reach the
        # sampler, therefore no same-resolution/H3-Latent pass is executed.
        final_latent, _, first_latent = Ref2VAUnifiedMultiPassSampler().sample_unified(
            model, conditioning, latent_image, primary_settings,
            refinement_settings if bool(enable_final_video) else None,
        )
        initial_video = None
        if bool(enable_initial_video):
            initial_video = Ref2VADecodeCreateVideo.execute(first_latent, video_vae, audio_vae, fps, bit_depth)[0]
        if not bool(enable_final_video):
            # The frontend mutes the existing RTX/final-save branch together
            # with this toggle. Returning None prevents accidental final decode
            # should a stale external link survive a hand-edited workflow.
            return (initial_video, None)
        final_video = Ref2VADecodeCreateVideo.execute(final_latent, video_vae, audio_vae, fps, bit_depth)[0]
        return (initial_video, final_video)

class Ref2VAMultiPassSampler:
    @classmethod
    def INPUT_TYPES(cls):return {"required":{"noise":("NOISE",),"guider":("GUIDER",),"sampler":("SAMPLER",),"sigmas":("SIGMAS",),"latent_image":("LATENT",),"passes":("INT",{"default":2,"min":1,"max":3}),"refine_steps":("INT",{"default":3,"min":1,"max":12}),"refine_denoise":("FLOAT",{"default":0.30,"min":0.01,"max":0.8}),"refine_scheduler":(SCHEDULERS,{"default":"beta"}),"refine_extra_steps":("INT",{"default":1,"min":0,"max":15}),"refine_start_at_sigma":("FLOAT",{"default":0.6,"min":0,"max":20}),"refine_spacing":(SPACINGS,{"default":"cosine"})}}
    RETURN_TYPES=("LATENT","LATENT"); RETURN_NAMES=("output","denoised_output"); FUNCTION="sample"; CATEGORY="Ref2VA/sampling"
    def sample(self,noise,guider,sampler,sigmas,latent_image,**settings):return _multi_pass(noise,guider,sampler,sigmas,latent_image,settings)

class Ref2VASequenceWorkbenchV01(io.ComfyNode):
    """Independent v0.1 shot queue for the user's Ref2VA / FL2V workflow.

    This node is deliberately only a controller in v0.1.  It provides a real
    selected-shot prompt and duration to the existing generation path, while
    the next iteration will add the separate serial queue executor.  Keeping
    control and generation separate protects the user's stable workflow.
    """
    DEFAULT_TIMELINE = {
        "project": "连续镜头项目",
        "active": "shot-1",
        "shots": [
            {"id": "shot-1", "name": "镜头 1", "duration": 5.0, "enabled": True,
             "prompt": "建立镜头：明确人物、场景、光线与动作起点。"},
            {"id": "shot-2", "name": "镜头 2", "duration": 5.0, "enabled": True,
             "prompt": "承接上一镜头尾帧：只描述后续动作、镜头运动或情绪变化。"},
            {"id": "shot-3", "name": "镜头 3", "duration": 5.0, "enabled": True,
             "prompt": "承接上一镜头尾帧：完成动作并收束画面。"},
        ],
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VASequenceWorkbenchV01",
            display_name="Ref2VA：连续镜头工作台 v0.1",
            category="Ref2VA/storyboard",
            inputs=[
                io.String.Input("timeline_data", multiline=True,
                    default=json.dumps(cls.DEFAULT_TIMELINE, ensure_ascii=False),
                    display_name="镜头队列数据"),
            ],
            outputs=[
                io.String.Output(display_name="当前镜头提示词"),
                io.Float.Output(display_name="当前镜头时长（秒）"),
                io.String.Output(display_name="当前镜头名称"),
                io.Boolean.Output(display_name="需要上一镜头尾帧"),
                io.String.Output(display_name="镜头队列状态"),
            ],
        )

    @classmethod
    def execute(cls, timeline_data):
        try:
            data = json.loads(timeline_data or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("连续镜头工作台数据无效。请使用工作台内的新增、删除和编辑操作，不要手改隐藏数据。") from exc
        shots = data.get("shots") if isinstance(data, dict) else None
        if not isinstance(shots, list) or not shots:
            raise ValueError("连续镜头工作台至少需要保留一个镜头。")
        active_id = str(data.get("active") or shots[0].get("id") or "shot-1")
        active_index = next((i for i, shot in enumerate(shots) if str(shot.get("id")) == active_id), 0)
        active = shots[active_index]
        name = str(active.get("name") or f"镜头 {active_index + 1}").strip() or f"镜头 {active_index + 1}"
        prompt = str(active.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{name} 还没有填写提示词。")
        try:
            duration = float(active.get("duration", 5.0))
        except (TypeError, ValueError):
            duration = 5.0
        duration = max(0.2, min(duration, 150.0))
        handoff = active_index > 0
        enabled_count = sum(1 for shot in shots if bool(shot.get("enabled", True)))
        status = f"{data.get('project') or '连续镜头项目'}｜当前：{name}｜共 {len(shots)} 段，启用 {enabled_count} 段｜{'需要上一镜头尾帧' if handoff else '首镜头：不使用上一镜头尾帧'}"
        return io.NodeOutput(prompt, duration, name, handoff, status)


class Ref2VASequenceWorkbenchV03(Ref2VASequenceWorkbenchV01):
    """v0.3 compact continuity workbench.

    The execution contract stays identical to v0.1 so it can be swapped into
    the copied workflow without breaking its real prompt, duration, archive or
    tail-frame links.  The separate node id gives v0.3 a fresh, fixed-size DOM
    panel instead of inheriting any oversized saved v0.1 canvas geometry.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VASequenceWorkbenchV03",
            display_name="Ref2VA：连续镜头工作台 v0.3",
            category="Ref2VA/storyboard",
            # The v0.3 editor owns the large prompt fields in its DOM panel.
            # Keep this hidden serialization carrier single-line; ComfyUI's
            # native multiline STRING widget may reserve a tall blank node
            # body before an extension has mounted its panel.
            inputs=[io.String.Input("timeline_data", multiline=False,
                default=json.dumps(cls.DEFAULT_TIMELINE, ensure_ascii=False),
                display_name="镜头队列数据")],
            outputs=[
                io.String.Output(display_name="当前镜头提示词"),
                io.Float.Output(display_name="当前镜头时长（秒）"),
                io.String.Output(display_name="当前镜头名称"),
                io.Boolean.Output(display_name="需要上一镜头尾帧"),
                io.String.Output(display_name="镜头队列状态"),
            ],
        )


class Ref2VAUnifiedDirectorStudio(io.ComfyNode):
    """The single project/timeline control point for the next Director Studio.

    The node is intentionally useful before the serial runner exists: it
    drives the user's existing generation graph through real outputs, while its
    project data preserves all mode-specific assets and continuity choices.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Ref2VAUnifiedDirectorStudio",
            display_name="梦镜 DreamShot",
            category="Ref2VA/director",
            inputs=[io.String.Input(
                "project_data", multiline=False, default=project_json(default_project()),
                display_name="导演台项目数据",
            )],
            outputs=[
                io.String.Output(display_name="当前镜头提示词"),
                io.Float.Output(display_name="当前镜头时长（秒）"),
                io.String.Output(display_name="当前镜头名称"),
                io.String.Output(display_name="当前镜头模式"),
                io.Boolean.Output(display_name="启用连续上下文"),
                io.String.Output(display_name="规范化项目数据"),
            ],
        )

    @classmethod
    def execute(cls, project_data):
        project = normalise_project(project_data)
        shots = project["shots"]
        requested = str(project.get("active_shot_id") or "")
        shot = next((item for item in shots if item["id"] == requested), None)
        if shot is None:
            shot = next((item for item in shots if item.get("enabled")), shots[0])
            project["active_shot_id"] = shot["id"]
        position = shots.index(shot)
        route = continuity_route(shot, has_previous_take=position > 0)
        prompt = "\n\n".join(part for part in (
            str(project.get("global_prompt") or "").strip(),
            str(shot.get("prompt") or "").strip(),
        ) if part).strip()
        return io.NodeOutput(
            prompt,
            float(shot["duration_seconds"]),
            shot["name"],
            shot["mode"],
            bool(route["enabled"]),
            project_json(project),
        )


class Ref2VAUnifiedDirectorRunner:
    """First executable Director Studio runner.

    A single queue runs enabled shots in order.  Each successful shot gets an
    immutable take archive before the following shot obtains its continuity
    context.  This makes "run all" deterministic rather than a set of
    independent jobs racing over whichever MP4 happened to be written last.
    """
    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models")
        ref_default = next((name for name in unets if "ref2va" in name.lower()), unets[0] if unets else "")
        fl_default = next((name for name in unets if "fl2va" in name.lower()), ref_default)
        vaes = folder_paths.get_filename_list("vae")
        loras = folder_paths.get_filename_list("loras")
        ref_turbo_default = _default_turbo_lora(loras, "ref2va")
        fl_turbo_default = _default_turbo_lora(loras, "fl2v")
        return {
            "required": {
                "ref2va_unet_name": (unets, {"default": ref_default, "label": "参考/连续镜头 UNet（Ref2VA）"}),
                "fl2va_unet_name": (unets, {"default": fl_default, "label": "文生/图生/首尾帧 UNet（FL2VA）"}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {"label": "H3 CLIP"}),
                "clip_type": (["minimax"], {"default": "minimax"}),
                "video_vae_name": (vaes, {"label": "视频 VAE"}),
                "audio_vae_name": (vaes, {"label": "音频 VAE"}),
                "enable_turbo_lora": ("BOOLEAN", {"default": True, "label_on": "按镜头模式自动加载 4 步 Turbo LoRA", "label_off": "不加载 Turbo LoRA"}),
                "turbo_lora_name": (loras, {"default": fl_turbo_default or ref_turbo_default or (loras[0] if loras else ""), "label": "旧版 Turbo LoRA（兼容）"}),
                "turbo_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_stack_json": ("STRING", {"default": "[]", "multiline": False, "label": "附加 LoRA Stack"}),
                "project_data": ("STRING", {"multiline": False, "default": project_json(default_project())}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1}),
                "seed_mode": (["随机", "固定"], {"default": "随机", "label": "种子模式"}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
                "scheduler": (SCHEDULERS, {"default": "beta"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "sampler_name": (list(comfy.samplers.SAMPLER_NAMES), {"default": "euler"}),
                "main_extra_steps": ("INT", {"default": 1, "min": 0, "max": 15}),
                "main_start_at_sigma": ("FLOAT", {"default": 0.70, "min": 0, "max": 20, "step": 0.01}),
                "main_spacing": (SPACINGS, {"default": "cosine"}),
                "run_scope": (["选中镜头", "仅当前镜头", "已启用全部镜头"], {"default": "仅当前镜头"}),
                "enable_final_video": ("BOOLEAN", {"default": False, "label_on": "生成最终/二采版本", "label_off": "仅生成原始版本"}),
                "second_sampling_mode": (["同分辨率多次采样", "H3 Latent 超分"], {"default": "H3 Latent 超分"}),
                "latent_upscale_model": (LATENT_UPSCALE_MODELS, {"default": DEFAULT_LATENT_UPSCALE_MODEL}),
                "second_aspect_ratio": (["16:9"], {"default": "16:9"}),
                "second_megapixels": ([1.0, 1.5, 2.0], {"default": 1.5}),
                "upscale_passes": ("INT", {"default": 1, "min": 1, "max": 2}),
                "passes": ("INT", {"default": 1, "min": 1, "max": 3}),
                "refine_scheduler": (SCHEDULERS, {"default": "beta"}),
                "refine_steps": ("INT", {"default": 3, "min": 1, "max": 12}),
                "refine_denoise": ("FLOAT", {"default": 0.30, "min": 0.01, "max": 0.80, "step": 0.01}),
                "refine_extra_steps": ("INT", {"default": 1, "min": 0, "max": 15}),
                "refine_start_at_sigma": ("FLOAT", {"default": 0.60, "min": 0, "max": 20, "step": 0.01}),
                "refine_spacing": (SPACINGS, {"default": "cosine"}),
                "enable_rtx_upscale": ("BOOLEAN", {"default": False, "label_on": "开启 RTX 最终放大", "label_off": "关闭 RTX 最终放大"}),
                "rtx_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "rtx_quality": (["LOW", "MEDIUM", "HIGH", "ULTRA"], {"default": "HIGH"}),
                "rtx_filename_prefix": ("STRING", {"default": "video/H3_Final_RTX"}),
                # Appended after all 1.7.2 widgets so saved positional widget values remain compatible.
                "ref2va_turbo_lora_name": (loras, {"default": ref_turbo_default, "label": "Ref2VA 模式 4 步 LoRA"}),
                "fl2v_turbo_lora_name": (loras, {"default": fl_turbo_default, "label": "T2V / I2V / FL2V 模式 4 步 LoRA"}),
                # Appended after the 1.7.4 widgets to preserve positional workflow compatibility.
                "final_upscale_method": (["关闭", "NVIDIA RTX", "TE FlashVSR"], {"default": "关闭", "label": "最终视频放大方式"}),
                "te_flashvsr_model": (["FlashVSR-v1.1"], {"default": "FlashVSR-v1.1", "label": "TE FlashVSR 模型"}),
                "te_flashvsr_mode": (["tiny", "tiny-long", "full"], {"default": "tiny", "label": "TE FlashVSR 推理模式"}),
                "te_flashvsr_precision": (["bf16", "fp16"], {"default": "bf16", "label": "TE FlashVSR 精度"}),
                "te_flashvsr_scale": ([2, 4], {"default": 2, "label": "TE FlashVSR 放大倍数"}),
                "te_flashvsr_quality": (["detail", "balanced", "throughput"], {"default": "balanced", "label": "TE FlashVSR 质量模式"}),
                "te_flashvsr_spatial": (["auto", "full_frame", "adaptive_tiles"], {"default": "auto", "label": "TE FlashVSR 空间策略"}),
                "te_flashvsr_memory": (["auto", "resident", "staged"], {"default": "auto", "label": "TE FlashVSR 显存策略"}),
                "te_flashvsr_attention": (["sparse_sage2", "block_sparse_attn", "auto"], {"default": "sparse_sage2", "label": "TE FlashVSR 注意力后端"}),
                "te_flashvsr_color_fix": ("BOOLEAN", {"default": True, "label_on": "开启颜色修正", "label_off": "关闭颜色修正"}),
            },
        }

    # v1 deliberately exposes exactly one input wire and one result wire.
    # The project UI owns all per-shot images, videos, audio and continuity;
    # legacy sockets remain on the old v0.x nodes instead of cluttering the
    # shareable Director Studio canvas.
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Ref2VA/director"

    @staticmethod
    def _length(seconds, fps=24.0):
        raw = max(5, round(float(seconds) * float(fps)))
        return raw + (5 - raw % 17) % 17

    @staticmethod
    def _video_parts(video):
        if video is None:
            return None, None
        parts = video.get_components()
        return getattr(parts, "images", None), getattr(parts, "audio", None)

    @staticmethod
    def _asset_path(asset):
        """Load an image deliberately attached to one project shot.

        The browser stores only an input-relative filename.  Resolving it here
        keeps the project portable and, importantly, prevents a hand-edited
        project JSON from reading arbitrary files outside ComfyUI's input
        directory.
        """
        if not isinstance(asset, dict) or not asset.get("filename"):
            return None
        base = os.path.realpath(folder_paths.get_input_directory())
        relative = os.path.normpath(os.path.join(str(asset.get("subfolder") or ""), str(asset["filename"])))
        candidate = os.path.realpath(os.path.join(base, relative))
        if os.path.commonpath((base, candidate)) != base:
            raise ValueError("镜头素材路径无效：只能读取 ComfyUI input 目录内的文件。")
        if not os.path.isfile(candidate):
            raise ValueError(f"镜头素材不存在：{asset['filename']}")
        return candidate

    @classmethod
    def _shot_image(cls, asset):
        path = cls._asset_path(asset)
        return _image_tensor_from_path(path) if path else None

    @classmethod
    def _shot_images(cls, assets):
        values = [cls._shot_image(item) for item in (assets or [])]
        return batch_reference_images(values)

    @classmethod
    def _shot_media_refs(cls, assets):
        videos, video_audios, audios = {}, {}, {}
        for index, asset in enumerate(assets or []):
            path = cls._asset_path(asset)
            if not path:
                continue
            video = VideoFromFile(str(path))
            images, audio = cls._video_parts(video)
            if images is not None:
                videos[f"ref_video_{len(videos)}"] = images
            if audio is not None and bool(asset.get("audio_enabled", True)):
                video_audios[f"ref_video_audio_{len(video_audios)}"] = audio
        return videos, video_audios, audios

    @classmethod
    def _shot_audio_refs(cls, assets):
        result = {}
        for asset in assets or []:
            path = cls._asset_path(asset)
            if not path:
                continue
            relative = os.path.relpath(path, folder_paths.get_input_directory()).replace("\\", "/")
            result[f"ref_audio_{len(result)}"] = LoadAudio.execute(relative)[0]
        return result

    @staticmethod
    def _timeline_plan(shot):
        """Resolve the enabled cyan generation range into ordered H3 inputs."""
        timeline = shot.get("timeline") if isinstance(shot.get("timeline"), dict) else {}
        if not timeline.get("enabled"):
            return None
        start = float(timeline.get("generation_start", 0.0) or 0.0)
        end = float(timeline.get("generation_end", shot.get("duration_seconds", 5.0)) or shot.get("duration_seconds", 5.0))
        clips = []
        for clip in timeline.get("clips") or []:
            if not isinstance(clip, dict) or not isinstance(clip.get("asset"), dict):
                continue
            clip_start = float(clip.get("start", 0.0) or 0.0)
            clip_end = clip_start + float(clip.get("duration", 0.0) or 0.0)
            if clip_end <= start or clip_start >= end:
                continue
            clips.append(clip)
        clips.sort(key=lambda item: (float(item.get("start", 0.0) or 0.0), str(item.get("id") or "")))
        return {"start": start, "end": end, "clips": clips}

    @classmethod
    def _timeline_assets(cls, shot):
        plan = cls._timeline_plan(shot)
        if plan is None:
            return None, []
        assets = {"first_frame": None, "last_frame": None, "images": [], "videos": [], "audios": []}
        guides = []
        for clip in plan["clips"]:
            if clip.get("usage") == "edit":
                continue
            kind, role, asset = clip.get("kind"), clip.get("role", "editable_reference"), clip["asset"]
            if role == "editable_reference":
                bucket = {"image": "images", "video": "videos", "audio": "audios"}.get(kind)
                if bucket:
                    item = dict(asset)
                    item["audio_enabled"] = bool(clip.get("audio_enabled", True))
                    assets[bucket].append(item)
            else:
                guides.append(clip)
        return assets, guides

    @classmethod
    def _apply_timeline_guides(cls, conditioning, latent, guides, fps, video_vae, audio_vae, range_start=0.0):
        for clip in guides:
            kind = clip.get("kind")
            start_frame = max(0, int(round((float(clip.get("start", 0.0)) - range_start) * fps)))
            image = audio = None
            if kind == "image":
                image = cls._shot_image(clip.get("asset"))
            elif kind == "video":
                path = cls._asset_path(clip.get("asset"))
                frames, soundtrack = cls._video_parts(VideoFromFile(str(path)))
                if clip.get("role") == "boundary_only" and frames is not None and len(frames):
                    conditioning = MiniMaxH3AddGuide.execute(conditioning, latent, start_frame, video_vae, audio_vae, frames[:1], None)[0]
                    end_frame = max(start_frame, int(round((float(clip.get("start", 0.0)) + float(clip.get("duration", 0.0)) - range_start) * fps)) - 1)
                    conditioning = MiniMaxH3AddGuide.execute(conditioning, latent, end_frame, video_vae, audio_vae, frames[-1:], None)[0]
                    continue
                image, audio = frames, soundtrack if clip.get("audio_enabled", True) else None
            elif kind == "audio":
                path = cls._asset_path(clip.get("asset"))
                relative = os.path.relpath(path, folder_paths.get_input_directory()).replace("\\", "/")
                audio = LoadAudio.execute(relative)[0]
            conditioning = MiniMaxH3AddGuide.execute(conditioning, latent, start_frame, video_vae, audio_vae, image, audio)[0]
        return conditioning

    @staticmethod
    def _cached_take(project, shot, fingerprint):
        """Return archived VIDEO objects only for an exact current fingerprint."""
        for record in reversed(shot.get("takes") or []):
            if record.get("fingerprint") != fingerprint:
                continue
            take_id = str(record.get("take_id") or "")
            initial_path = take_path(project["project_id"], shot["id"], take_id, "initial")
            if initial_path is None:
                continue
            final_path = take_path(project["project_id"], shot["id"], take_id, "final")
            return record, VideoFromFile(str(initial_path)), VideoFromFile(str(final_path)) if final_path else None
        return None

    @staticmethod
    def _cached_primary_take(project, shot, primary_fingerprint):
        for record in reversed(shot.get("takes") or []):
            if record.get("primary_fingerprint") != primary_fingerprint:
                continue
            take_id = str(record.get("take_id") or "")
            initial_path = take_path(project["project_id"], shot["id"], take_id, "initial")
            latent_path = take_path(project["project_id"], shot["id"], take_id, "initial_latent")
            if initial_path is None or latent_path is None:
                continue
            return record, VideoFromFile(str(initial_path)), torch.load(str(latent_path), map_location="cpu", weights_only=False)
        return None

    def run(self, ref2va_unet_name, fl2va_unet_name, weight_dtype, clip_name, clip_type,
            video_vae_name, audio_vae_name, enable_turbo_lora, turbo_lora_name, turbo_lora_strength, lora_stack_json,
            project_data, aspect_ratio="16:9", megapixels=1.0,
            seed_mode="随机", noise_seed=0, scheduler="beta", steps=4, denoise=1.0, sampler_name="euler",
            main_extra_steps=1, main_start_at_sigma=0.70, main_spacing="cosine", run_scope="仅当前镜头", enable_final_video=False,
            second_sampling_mode="H3 Latent 超分", latent_upscale_model=DEFAULT_LATENT_UPSCALE_MODEL,
            second_aspect_ratio="16:9", second_megapixels=1.5, upscale_passes=1, passes=1,
            refine_scheduler="beta", refine_steps=3, refine_denoise=0.30,
            refine_extra_steps=1, refine_start_at_sigma=0.60, refine_spacing="cosine",
            enable_rtx_upscale=False, rtx_scale=2.0, rtx_quality="HIGH", rtx_filename_prefix="video/H3_Final_RTX",
            refinement_settings=None, first_frame=None, last_frame=None,
            reference_images=None, reference_video=None,
            reference_video_audio=None, reference_audio=None,
            ref2va_turbo_lora_name=None, fl2v_turbo_lora_name=None,
            final_upscale_method="关闭", te_flashvsr_model="FlashVSR-v1.1", te_flashvsr_mode="tiny",
            te_flashvsr_precision="bf16", te_flashvsr_scale=2, te_flashvsr_quality="balanced",
            te_flashvsr_spatial="auto", te_flashvsr_memory="auto",
            te_flashvsr_attention="sparse_sage2", te_flashvsr_color_fix=True):
        # The portable build defaults to NoPreviews. Enable lightweight latent
        # image previews so the Director can inspect sampling progress.
        latent_preview.set_preview_method("auto")
        system = Ref2VADirectorSystem().load_system(
            ref2va_unet_name, fl2va_unet_name, weight_dtype, clip_name, clip_type,
            video_vae_name, audio_vae_name, enable_turbo_lora,
            turbo_lora_name, turbo_lora_strength, lora_stack_json,
            ref2va_turbo_lora_name=ref2va_turbo_lora_name,
            fl2v_turbo_lora_name=fl2v_turbo_lora_name,
        )[0]
        ref2va_model, fl2va_model = system.get("ref2va_model"), system.get("fl2va_model")
        clip = system.get("clip")
        video_vae, audio_vae = system.get("video_vae"), system.get("audio_vae")
        if any(item is None for item in (ref2va_model, fl2va_model, clip, video_vae, audio_vae)):
            raise ValueError("导演系统加载器没有提供完整的 Ref2VA/FL2VA 模型、CLIP、视频 VAE 和音频 VAE。")
        primary_settings = {
            "seed_mode": "固定" if seed_mode == "固定" else "随机",
            "noise_seed": int(noise_seed), "scheduler": scheduler, "steps": int(steps),
            "denoise": float(denoise), "sampler_name": sampler_name,
            "main_extra_steps": int(main_extra_steps),
            "main_start_at_sigma": float(main_start_at_sigma), "main_spacing": main_spacing,
        }
        refinement_settings = {
            "second_sampling_mode": "H3 Latent 超分" if second_sampling_mode == "H3 Latent 超分" else "同分辨率细化",
            "latent_upscale_model": latent_upscale_model,
            "second_aspect_ratio": str(aspect_ratio), "second_megapixels": float(second_megapixels),
            "upscale_passes": int(upscale_passes), "passes": int(passes),
            "refine_scheduler": refine_scheduler, "refine_steps": int(refine_steps),
            "refine_denoise": float(refine_denoise), "refine_extra_steps": int(refine_extra_steps),
            "refine_start_at_sigma": float(refine_start_at_sigma), "refine_spacing": refine_spacing,
        }
        project = normalise_project(project_data)
        resolved_upscale_method = str(final_upscale_method or "关闭")
        # Migrate 1.7.4 and older workflows: their only final-upscale setting
        # was enable_rtx_upscale. A newly selected method always wins.
        if resolved_upscale_method == "关闭" and bool(enable_rtx_upscale):
            resolved_upscale_method = "NVIDIA RTX"
        project["settings"]["render"] = {
            "aspect_ratio": str(aspect_ratio), "megapixels": float(megapixels),
            "lora_stack": json.loads(lora_stack_json or "[]"),
            "primary": primary_settings, "enable_final_video": bool(enable_final_video),
            "refinement": refinement_settings if bool(enable_final_video) else None,
            "rtx": {
                "enabled": resolved_upscale_method == "NVIDIA RTX", "scale": float(rtx_scale),
                "quality": str(rtx_quality), "filename_prefix": str(rtx_filename_prefix),
            },
            "final_upscale": {
                "method": resolved_upscale_method,
                "te_flashvsr": {
                    "model": str(te_flashvsr_model), "mode": str(te_flashvsr_mode),
                    "precision": str(te_flashvsr_precision), "scale": int(te_flashvsr_scale),
                    "quality": str(te_flashvsr_quality), "spatial": str(te_flashvsr_spatial),
                    "memory": str(te_flashvsr_memory), "attention": str(te_flashvsr_attention),
                    "color_fix": bool(te_flashvsr_color_fix),
                },
            },
        }
        output_profile = {
            "enable_final_video": bool(enable_final_video),
            "second_sampling_mode": refinement_settings["second_sampling_mode"],
            "second_megapixels": float(second_megapixels),
            "passes": int(passes),
            "final_upscale_method": resolved_upscale_method,
            "rtx_scale": float(rtx_scale), "rtx_quality": str(rtx_quality),
            "te_flashvsr_scale": int(te_flashvsr_scale), "te_flashvsr_mode": str(te_flashvsr_mode),
            "te_flashvsr_quality": str(te_flashvsr_quality),
        }
        write_project_snapshot(project)
        width, height = resolution_from_selector(str(aspect_ratio), float(megapixels))
        reports = [f"项目：{project['name']}（{project['project_id']}）", f"画布：{width}×{height}"]
        te_flashvsr_state = {}

        def apply_final_upscale(video, shot_index):
            if resolved_upscale_method == "关闭":
                return video
            if video is None:
                raise ValueError(f"镜头 {shot_index + 1} 没有可供最终放大的视频。")
            if resolved_upscale_method == "NVIDIA RTX":
                result = Ref2VARTXVideoPostprocess.execute(
                    video,
                    {"resize_type": "scale by multiplier", "scale": float(rtx_scale)},
                    str(rtx_quality), str(rtx_filename_prefix),
                )
                reports.append(f"镜头 {shot_index + 1} 完成 NVIDIA RTX 最终放大：×{float(rtx_scale):g}，质量 {rtx_quality}。")
                return result.args[0] if hasattr(result, "args") else result[0]
            if resolved_upscale_method != "TE FlashVSR":
                raise ValueError(f"未知最终视频放大方式：{resolved_upscale_method}")
            required_nodes = ("TEFlashVSRModelLoader", "TEFlashVSRTuning", "TEFlashVSRRestore")
            missing = [name for name in required_nodes if name not in nodes.NODE_CLASS_MAPPINGS]
            if missing:
                raise RuntimeError("TE FlashVSR 节点尚未加载，请安装 TE-Speed-FlashVSR 后完全重启 ComfyUI。缺少：" + "、".join(missing))
            if not te_flashvsr_state:
                loader_cls = nodes.NODE_CLASS_MAPPINGS["TEFlashVSRModelLoader"]
                tuning_cls = nodes.NODE_CLASS_MAPPINGS["TEFlashVSRTuning"]
                loaded = loader_cls().load(str(te_flashvsr_model), str(te_flashvsr_mode), str(te_flashvsr_precision), "auto")
                te_flashvsr_state["model"] = loaded[0]
                settings = tuning_cls().create(
                    quality_profile=str(te_flashvsr_quality), intensity=1.0,
                    spatial_strategy=str(te_flashvsr_spatial), memory_policy=str(te_flashvsr_memory),
                    attention_budget=2.0, kv_retention=3.0, local_radius=11,
                    max_tile_edge=256, blend_overlap=24, preprocess_batch=4,
                    attention_backend=str(te_flashvsr_attention),
                )
                te_flashvsr_state["settings"] = settings[0]
            components = video.get_components()
            restore_cls = nodes.NODE_CLASS_MAPPINGS["TEFlashVSRRestore"]
            restored = restore_cls().restore(
                te_flashvsr_state["model"], components.images, int(te_flashvsr_scale),
                bool(te_flashvsr_color_fix), int(noise_seed), settings=te_flashvsr_state["settings"],
            )[0]
            rebuilt = CreateVideo.execute(
                images=restored, audio=components.audio,
                fps=float(components.frame_rate), bit_depth=video.get_bit_depth(),
            )
            reports.append(
                f"镜头 {shot_index + 1} 完成 TE FlashVSR 最终超分：{te_flashvsr_mode} / "
                f"{te_flashvsr_quality} / ×{int(te_flashvsr_scale)} / {te_flashvsr_attention}。"
            )
            return rebuilt.args[0] if hasattr(rebuilt, "args") else rebuilt[0]
        # Keep both sources in memory.  Continuity is deliberately independent
        # from delivery: the following shot can choose the raw first pass for
        # stable motion, or a selected final pass when the user explicitly
        # asks for the refined result to become the continuity source.
        previous_initial_frames = previous_initial_audio = None
        previous_final_frames = previous_final_audio = None
        previous_frames = previous_audio = None
        previous_take_id = None
        last_initial = last_final = None
        requested_id = str(project.get("active_shot_id") or "")
        active_index = next((i for i, item in enumerate(project["shots"]) if item["id"] == requested_id), 0)
        if run_scope == "仅当前镜头":
            indices = [active_index]
        elif run_scope == "选中镜头":
            selected_ids = {str(item) for item in project.get("selected_shot_ids", [])}
            indices = [i for i, item in enumerate(project["shots"]) if str(item.get("id")) in selected_ids]
            if not indices:
                raise ValueError("选择模式下尚未选择任何镜头。")
            # Batch generation is an additive operation. Do not unexpectedly
            # resample a selected shot that already has a usable take merely
            # because its current fingerprint changed; use the per-shot
            # “重跑当前镜头” action when replacement is intended.
            skipped = [i for i in indices if project["shots"][i].get("takes") and project["shots"][i].get("selected_take_id")]
            indices = [i for i in indices if i not in skipped]
            if not indices:
                raise ValueError("所选镜头都已有可用视频；如需重新采样，请点击对应镜头的“重跑当前镜头”。")
        else:
            indices = [i for i, item in enumerate(project["shots"]) if item.get("enabled", True)]
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = {
            "job_id": job_id,
            "status": "queued",
            "scope": str(run_scope),
            "shot_ids": [project["shots"][i]["id"] for i in indices],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "parameters_snapshot": json.loads(json.dumps({
                "project_id": project["project_id"],
                "global_prompt": project.get("global_prompt", ""),
                "global_constraint_prompt": project.get("global_constraint_prompt", ""),
                "global_assets": project.get("global_assets", {}),
                "settings": project.get("settings", {}),
                "shots": [project["shots"][i] for i in indices],
            }, ensure_ascii=False)),
            "completed_shot_ids": [],
            "current_shot_id": None,
        }
        project.setdefault("jobs", []).append(job)
        project["jobs"] = project["jobs"][-100:]
        project["active_job_id"] = job_id
        for i in indices:
            project["shots"][i]["status"] = "queued"
        write_project_snapshot(project, reason="job_queued", create_history=True)
        for index in indices:
            shot = project["shots"][index]
            if not shot.get("enabled", True) and run_scope == "已启用全部镜头":
                reports.append(f"镜头 {index + 1} 跳过：未启用")
                continue
            shot["status"] = "sampling"
            job["status"] = "sampling"
            job["current_shot_id"] = shot["id"]
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_project_snapshot(project, reason="job_progress")
            requested_source = shot.get("continuity", {}).get("source", "initial")
            if requested_source == "final" and previous_final_frames is not None:
                previous_frames, previous_audio = previous_final_frames, previous_final_audio
            else:
                previous_frames, previous_audio = previous_initial_frames, previous_initial_audio
            if previous_frames is None and index > 0 and shot.get("continuity", {}).get("enabled"):
                previous = project["shots"][index - 1]
                selected_take = previous.get("selected_take_id")
                if selected_take:
                    source = shot.get("continuity", {}).get("source", "initial")
                    previous_frames = load_take_video_context(project["project_id"], previous["id"], selected_take, source=source, frames=int(shot.get("continuity", {}).get("context_frames", 22)))
                    previous_take_id = selected_take if previous_frames is not None else None
                    if previous_frames is not None:
                        reports.append(f"镜头 {index + 1} 从已确认的镜头 {index} 版本 {selected_take} 读取连续上下文。")
            route = continuity_route(shot, has_previous_take=previous_frames is not None)
            if route.get("needs_start_choice"):
                raise ValueError(f"{shot['name']} 同时有手动首帧与上一镜头尾帧；请在导演台选择本镜头的起点规则后再运行。")
            expected_fingerprint = shot_fingerprint(project, index, previous_take_id)
            primary_fingerprint = shot_fingerprint(project, index, previous_take_id, include_refinement=False)
            cached = self._cached_take(project, shot, expected_fingerprint)
            if cached is not None:
                record, initial, final = cached
                shot["selected_take_id"] = record["take_id"]
                shot["status"] = "cached"
                previous_initial_frames, previous_initial_audio = self._video_parts(initial)
                previous_final_frames, previous_final_audio = self._video_parts(final)
                previous_frames, previous_audio = previous_initial_frames, previous_initial_audio
                previous_take_id = record["take_id"]
                last_initial, last_final = initial, final
                reports.append(f"镜头 {index + 1} 复用缓存：{shot['name']} → {record['take_id']}（所有影响画面的字段未变）。")
                continue
            prompt = "\n\n".join(text for text in (project.get("global_prompt", "").strip(), project.get("global_constraint_prompt", "").strip(), shot.get("prompt", "").strip()) if text).strip()
            timeline_prompt = str((shot.get("timeline") or {}).get("prompt") or "").strip()
            if (shot.get("timeline") or {}).get("enabled") and timeline_prompt:
                prompt = "\n\n".join(text for text in (prompt, "[时间线补充指令]\n" + timeline_prompt) if text).strip()
            if not prompt:
                raise ValueError(f"{shot['name']} 尚未填写提示词。")
            shot_fps = float(shot.get("fps", 24.0) or 24.0)
            length = self._length(shot["duration_seconds"], shot_fps)
            mode = shot["mode"]
            assets = shot.get("assets") or {}
            timeline_assets, timeline_guides = self._timeline_assets(shot)
            if timeline_assets is not None:
                assets = {**assets, **timeline_assets}
            global_assets = project.get("global_assets") or {}
            # Project-level references are prepended so every shot receives
            # the same shared visual/audio context without replacing local assets.
            assets = {
                "first_frame": assets.get("first_frame"), "last_frame": assets.get("last_frame"),
                "images": list(global_assets.get("images") or []) + list(assets.get("images") or []),
                "videos": list(global_assets.get("videos") or []) + list(assets.get("videos") or []),
                "audios": list(global_assets.get("audios") or []) + list(assets.get("audios") or []),
            }
            shot_first = self._shot_image(assets.get("first_frame"))
            shot_last = self._shot_image(assets.get("last_frame"))
            shot_refs = self._shot_images(assets.get("images"))
            # v1 assets live in the project.  Keeping them in project data is
            # what makes a copied three-node workflow self-contained and
            # avoids mode switches disconnecting canvas wires.
            ref_images = {"ref_image_0": shot_refs} if shot_refs is not None else {}
            shot_videos, shot_video_audios, shot_audios = self._shot_media_refs(assets.get("videos"))
            ref_videos = shot_videos
            ref_video_audios = shot_video_audios
            ref_audios = self._shot_audio_refs(assets.get("audios"))
            inherited_tail = None
            if route.get("enabled") and previous_frames is not None:
                frames, audio = previous_frames, previous_audio
                if frames is None or len(frames) < 1:
                    raise ValueError(f"{shot['name']} 找不到上一镜头的可用画面，无法连续生成。")
                inherited_tail = frames[-1:].contiguous()
                if route.get("use_context_video"):
                    context_count = int(shot.get("continuity", {}).get("context_frames", 22))
                    context = frames[-min(len(frames), context_count):].contiguous()
                    # Reference ordering is explicit: context is Video 1;
                    # user-supplied shared reference becomes Video 2.
                    ref_videos = {"ref_video_0": context, **{f"ref_video_{i + 1}": value for i, value in enumerate(ref_videos.values())}}
                    if audio is not None:
                        previous_fps = float(project["shots"][index - 1].get("fps") or project.get("settings", {}).get("fps") or 24.0)
                        context_audio = trim_audio_tail(audio, len(context) / max(1.0, previous_fps))
                        if context_audio is not None:
                            ref_video_audios = {"ref_video_audio_0": context_audio, **{f"ref_video_audio_{i + 1}": value for i, value in enumerate(ref_video_audios.values())}}
                    if route.get("inject_continuity_prompt"):
                        continuity_prompt = (
                            "[自动无缝接力]\n"
                            "从 <Video 1> 的最终状态直接继续，不重新建立镜头或重新开始动作。"
                            "严格继承主体身份与外观、空间位置、姿态、动作阶段、运动方向与速度，"
                            "并继承摄影机位置、景别、焦距、运镜方向、场景结构、光线和环境动态。"
                            "保持自然运动惯性和平滑明暗过渡；不得跳位、冻结、重复动作起点、"
                            "突然变速、改变空间关系或产生画面闪变。"
                        )
                    else:
                        continuity_prompt = "连续镜头要求：承接 <Video 1> 的角色位置、动作惯性、镜头方向、光线与声音节奏，然后自然展开本镜头的新动作。"
                    prompt = continuity_prompt + "\n\n" + prompt
                elif route.get("use_tail_frame"):
                    ref_images = {"ref_image_0": inherited_tail, **{f"ref_image_{i + 1}": value for i, value in enumerate(ref_images.values())}}
            # Independent T2V/I2V/FL2V uses FL2VA.  As soon as a shot needs
            # a previous clip or tail as a reference, it must use Ref2VA so
            # the reference-conditioning payload is understood by the UNet.
            uses_reference_continuity = bool(route.get("enabled") and (ref_images or ref_videos or ref_audios))
            model = ref2va_model if mode in {"ref2va", "continuous_ref2va"} or uses_reference_continuity else fl2va_model
            # ImageToVideo uses its own hard-keyframe conditioning layout.
            # It cannot be safely spliced with ReferenceToVideo conditioning:
            # reference payload token rows vary with image/video dimensions.
            # For I2V/FL2V continuity we therefore make the selected prior
            # tail the actual start keyframe (the model-native, deterministic
            # route).  Reference/continuous modes retain the richer tail-video
            # + audio path below.
            if mode == "t2v" and not route.get("enabled"):
                conditioning, latent = MiniMaxH3ImageToVideo.execute(clip, video_vae, prompt, width, height, length, None, None)
            elif mode in {"i2v", "fl2v"}:
                if mode == "i2v":
                    use_previous = bool(route.get("enabled") and inherited_tail is not None and shot.get("continuity", {}).get("i2v_start_policy", "previous_tail") == "previous_tail")
                    start, end = (inherited_tail if use_previous else shot_first), None
                    if start is None:
                        raise ValueError(f"{shot['name']} 为图生视频，必须连接首帧，或开启上一镜头承接。")
                else:
                    start = inherited_tail if route.get("enabled") and shot.get("continuity", {}).get("fl2v_start_policy") == "previous_tail" else shot_first
                    end = shot_last
                    if start is None and end is None:
                        raise ValueError(f"{shot['name']} 为首帧 / 尾帧参考视频，至少需要首帧或尾帧中的一张图片。")
                conditioning, latent = MiniMaxH3ImageToVideo.execute(clip, video_vae, prompt, width, height, length, start, end)
            else:
                if route.get("enabled") and inherited_tail is not None and not ref_images:
                    ref_images = {"ref_image_0": inherited_tail}
                conditioning, latent = MiniMaxH3ReferenceToVideo.execute(
                    clip, video_vae, audio_vae, prompt, width, height, length, "match",
                    ref_images=ref_images or None, ref_videos=ref_videos or None,
                    ref_video_audios=ref_video_audios or None, ref_audios=ref_audios or None,
                )
            if timeline_guides:
                plan = self._timeline_plan(shot) or {"start": 0.0}
                conditioning = self._apply_timeline_guides(
                    conditioning, latent, timeline_guides, shot_fps, video_vae, audio_vae, plan["start"]
                )
            primary_cached = self._cached_primary_take(project, shot, primary_fingerprint) if bool(enable_final_video) else None
            sampler = Ref2VAUnifiedMultiPassSampler()
            if primary_cached is not None:
                source_record, initial, first_latent = primary_cached
                final_latent = sampler.refine_from_first(model, conditioning, first_latent, primary_settings, refinement_settings)
                final = Ref2VADecodeCreateVideo.execute(final_latent, video_vae, audio_vae, shot_fps, 8)[0]
                final = apply_final_upscale(final, index)
                record = archive_take(
                    project=project, shot=shot, initial_video=None, final_video=final,
                    parent_take_id=previous_take_id, source_take_id=source_record["take_id"],
                    fingerprint=expected_fingerprint, primary_fingerprint=primary_fingerprint, fps=shot_fps,
                    output_profile=output_profile,
                )
                reports.append(f"镜头 {index + 1} 复用原版 latent，仅执行二采/超分：{source_record['take_id']} → {record['take_id']}。")
            else:
                final_latent, _, first_latent = sampler.sample_unified(
                    model, conditioning, latent, primary_settings,
                    refinement_settings if bool(enable_final_video) else None,
                )
                initial = Ref2VADecodeCreateVideo.execute(first_latent, video_vae, audio_vae, shot_fps, 8)[0]
                final = Ref2VADecodeCreateVideo.execute(final_latent, video_vae, audio_vae, shot_fps, 8)[0] if bool(enable_final_video) else None
                if resolved_upscale_method != "关闭":
                    final = apply_final_upscale(final if final is not None else initial, index)
                record = archive_take(
                    project=project, shot=shot, initial_video=initial, final_video=final,
                    parent_take_id=previous_take_id, fingerprint=expected_fingerprint,
                    primary_fingerprint=primary_fingerprint, initial_latent=first_latent, fps=shot_fps,
                    output_profile=output_profile,
                )
            shot["takes"].append(record)
            shot["selected_take_id"] = record["take_id"]
            shot["status"] = "generated"
            job["completed_shot_ids"].append(shot["id"])
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            # Persist immediately, not only after the whole queue finishes.
            # If a later segment is stopped or errors, the completed cards
            # remain available for preview, continuation and cache reuse.
            write_project_snapshot(project)
            previous_initial_frames, previous_initial_audio = self._video_parts(initial)
            previous_final_frames, previous_final_audio = self._video_parts(final)
            previous_frames, previous_audio = previous_initial_frames, previous_initial_audio
            previous_take_id = record["take_id"]
            last_initial, last_final = initial, final
            reports.append(f"镜头 {index + 1} 完成：{shot['name']} → {record['take_id']}（{route['message']}）")
        job["status"] = "completed"
        job["current_shot_id"] = None
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        project["active_job_id"] = None
        write_project_snapshot(project, reason="job_completed")
        if last_initial is None:
            raise ValueError("当前项目没有启用任何镜头。")
        return ()


def _default_turbo_lora(loras, family):
    """Select only a four-step Turbo LoRA that belongs to the requested model family."""
    candidates = [
        name for name in loras
        if _lora_matches_family(name, family)
        and ("4step" in name.lower() or "4_step" in name.lower() or "4-step" in name.lower())
    ]
    return candidates[0] if candidates else ""


def _lora_matches_family(name, family):
    lowered = str(name or "").lower()
    family_tokens = ("ref2va", "ref2v") if family == "ref2va" else ("fl2va", "fl2v")
    return any(token in lowered for token in family_tokens)


class Ref2VADirectorSystem:
    """One-wire model pack for a clean shareable Director Studio canvas."""
    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models")
        ref_default = next((name for name in unets if "ref2va" in name.lower()), unets[0] if unets else "")
        fl_default = next((name for name in unets if "fl2va" in name.lower()), ref_default)
        vaes = folder_paths.get_filename_list("vae")
        loras = folder_paths.get_filename_list("loras")
        ref_turbo_default = _default_turbo_lora(loras, "ref2va")
        fl_turbo_default = _default_turbo_lora(loras, "fl2v")
        return {"required": {
            "ref2va_unet_name": (unets, {"default": ref_default, "label": "参考/连续镜头 UNet（Ref2VA）"}),
            "fl2va_unet_name": (unets, {"default": fl_default, "label": "文生/图生/首尾帧 UNet（FL2VA）"}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
            "clip_name": (folder_paths.get_filename_list("text_encoders"), {"label": "H3 CLIP"}),
            "clip_type": (["minimax"], {"default": "minimax"}),
            "video_vae_name": (vaes, {"label": "视频 VAE"}),
            "audio_vae_name": (vaes, {"label": "音频 VAE"}),
            "enable_turbo_lora": ("BOOLEAN", {"default": True, "label_on": "按镜头模式自动加载 4 步 Turbo LoRA", "label_off": "不加载 Turbo LoRA"}),
            "turbo_lora_name": (loras, {"default": fl_turbo_default or ref_turbo_default or (loras[0] if loras else ""), "label": "旧版 Turbo LoRA（兼容）"}),
            "turbo_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            "lora_stack_json": ("STRING", {"default": "[]", "multiline": False, "label": "附加 LoRA Stack"}),
            "ref2va_turbo_lora_name": (loras, {"default": ref_turbo_default, "label": "Ref2VA 模式 4 步 LoRA"}),
            "fl2v_turbo_lora_name": (loras, {"default": fl_turbo_default, "label": "T2V / I2V / FL2V 模式 4 步 LoRA"}),
        }}

    RETURN_TYPES = ("REF2VA_DIRECTOR_SYSTEM",)
    RETURN_NAMES = ("导演系统",)
    FUNCTION = "load_system"
    CATEGORY = "Ref2VA/director"
    DESCRIPTION = "统一加载 Ref2VA 导演台所需的 UNet、CLIP、视频 VAE、音频 VAE 和 Turbo LoRA。"

    def load_system(self, ref2va_unet_name, fl2va_unet_name, weight_dtype, clip_name, clip_type,
                    video_vae_name, audio_vae_name, enable_turbo_lora=False,
                    turbo_lora_name=None, turbo_lora_strength=1.0, lora_stack_json="[]",
                    ref2va_turbo_lora_name=None, fl2v_turbo_lora_name=None):
        if not ref2va_unet_name or not fl2va_unet_name:
            raise ValueError("导演系统需要同时选择 Ref2VA 与 FL2VA 两个 UNet。")
        ref_model = nodes.UNETLoader().load_unet(ref2va_unet_name, weight_dtype)[0]
        fl_model = nodes.UNETLoader().load_unet(fl2va_unet_name, weight_dtype)[0]
        if enable_turbo_lora:
            ref_lora = str(ref2va_turbo_lora_name or "").strip()
            fl_lora = str(fl2v_turbo_lora_name or "").strip()
            if not ref_lora or not fl_lora:
                raise ValueError("已启用按模式自动 Turbo LoRA，但 Ref2VA 或 FL2V 的 4 步 LoRA 没有选择完整。")
            available_loras = set(folder_paths.get_filename_list("loras"))
            if ref_lora not in available_loras:
                raise ValueError(f"Ref2VA 模式 Turbo LoRA 文件不存在：{ref_lora}")
            if fl_lora not in available_loras:
                raise ValueError(f"T2V / I2V / FL2V 模式 Turbo LoRA 文件不存在：{fl_lora}")
            ref_model = nodes.LoraLoaderModelOnly().load_lora_model_only(ref_model, ref_lora, float(turbo_lora_strength))[0]
            fl_model = nodes.LoraLoaderModelOnly().load_lora_model_only(fl_model, fl_lora, float(turbo_lora_strength))[0]
        try:
            lora_stack = json.loads(lora_stack_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"附加 LoRA Stack 数据无效：{exc}") from exc
        if not isinstance(lora_stack, list):
            raise ValueError("附加 LoRA Stack 必须是列表。")
        available_loras = set(folder_paths.get_filename_list("loras"))
        for index, item in enumerate(lora_stack, start=1):
            if not isinstance(item, dict) or not item.get("name"):
                raise ValueError(f"附加 LoRA 第 {index} 项缺少模型文件。")
            name = str(item["name"])
            if name not in available_loras:
                raise ValueError(f"附加 LoRA 文件不存在：{name}")
            strength = float(item.get("strength", 1.0))
            if not -100.0 <= strength <= 100.0:
                raise ValueError(f"附加 LoRA 第 {index} 项权重超出 -100 到 100。")
            ref_model = nodes.LoraLoaderModelOnly().load_lora_model_only(ref_model, name, strength)[0]
            fl_model = nodes.LoraLoaderModelOnly().load_lora_model_only(fl_model, name, strength)[0]
        clip = nodes.CLIPLoader().load_clip(clip_name, clip_type, "default")[0]
        video_vae = nodes.VAELoader().load_vae(video_vae_name)[0]
        audio_vae = nodes.VAELoader().load_vae(audio_vae_name)[0]
        return ({"ref2va_model": ref_model, "fl2va_model": fl_model, "clip": clip, "video_vae": video_vae, "audio_vae": audio_vae},)


class Ref2VADirectorDelivery:
    """The second and final visible connection in a shareable Director workflow."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "director_result": ("REF2VA_DIRECTOR_RESULT",),
            "delivery_scope": (["最后生成镜头", "当前选中镜头", "全部已启用镜头合并"], {"default": "全部已启用镜头合并"}),
            "delivery_source": (["最终视频优先", "仅原始视频"], {"default": "最终视频优先"}),
            "filename_prefix": ("STRING", {"default": "video/Ref2VA_Director/当前交付", "multiline": False}),
        }}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("当前交付视频",)
    FUNCTION = "deliver"
    OUTPUT_NODE = True
    CATEGORY = "Ref2VA/director"

    def deliver(self, director_result, delivery_scope="全部已启用镜头合并", delivery_source="最终视频优先", filename_prefix="video/Ref2VA_Director/当前交付"):
        if not isinstance(director_result, dict):
            raise ValueError("交付台需要连接统一导演台逐段执行器的“导演台结果”。")
        project = director_result.get("project") or {}
        initial = director_result.get("initial_video")
        final = director_result.get("final_video")
        source = "final_if_available" if delivery_source == "最终视频优先" else "initial"
        source_label = "最终视频优先" if source == "final_if_available" else "仅原始视频"
        selected_path = None
        if delivery_scope == "全部已启用镜头合并":
            selected_path = concat_selected_takes(project, source=source)
            selected = VideoFromFile(str(selected_path))
            source_label = f"全部已启用镜头合并（{source_label}）"
        elif delivery_scope == "当前选中镜头":
            shot_id = str(project.get("active_shot_id") or "")
            shot = next((item for item in project.get("shots") or [] if item.get("id") == shot_id), None)
            if shot is None or not shot.get("selected_take_id"):
                raise ValueError("当前镜头尚未选择可交付版本。")
            key = "final" if source == "final_if_available" else "initial"
            selected_path = take_path(project["project_id"], shot["id"], shot["selected_take_id"], key)
            if selected_path is None and key == "final":
                selected_path = take_path(project["project_id"], shot["id"], shot["selected_take_id"], "initial")
            if selected_path is None:
                raise ValueError("当前镜头选中的版本文件不存在。")
            selected = VideoFromFile(str(selected_path))
            source_label = f"当前镜头：{shot.get('name') or shot_id}（{source_label}）"
        else:
            shot_id = str(director_result.get("last_shot_id") or "")
            shot = next((item for item in project.get("shots") or [] if item.get("id") == shot_id), None)
            if shot is not None and shot.get("selected_take_id"):
                key = "final" if source == "final_if_available" else "initial"
                selected_path = take_path(project["project_id"], shot["id"], shot["selected_take_id"], key)
                if selected_path is None and key == "final":
                    selected_path = take_path(project["project_id"], shot["id"], shot["selected_take_id"], "initial")
            selected = VideoFromFile(str(selected_path)) if selected_path is not None else (initial if source == "initial" else (final or initial))
            source_label = "最后生成镜头：" + source_label
        if selected is None:
            raise ValueError("导演台结果中没有可交付的视频。")
        if selected_path is None:
            return (selected,)
        output_root = os.path.realpath(folder_paths.get_output_directory())
        real_path = os.path.realpath(str(selected_path))
        if os.path.commonpath((output_root, real_path)) != output_root:
            raise ValueError("交付视频不在 ComfyUI output 目录内，无法创建安全预览。")
        relative = os.path.relpath(real_path, output_root)
        subfolder, filename = os.path.split(relative)
        return io.NodeOutput(
            selected,
            ui=ui.PreviewVideo([ui.SavedResult(filename, subfolder.replace("\\", "/"), io.FolderType.output)]),
        )


def _install_unified_director_routes():
    """Small read-only route used by the timeline to refresh saved take cards."""
    try:
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:
        # Direct Python imports and ComfyUI's early plugin scan have no live
        # PromptServer yet.  The normal server import path installs this later.
        return
    if getattr(routes, "_ref2va_director_routes", False):
        return

    def valid_project_id(value):
        value = str(value or "")
        return bool(value) and value == safe_project_path_part(value) and value.startswith("project-")

    def pending_upscale_directory():
        path = os.path.realpath(os.path.join(folder_paths.get_temp_directory(), "ref2va_existing_upscale"))
        os.makedirs(path, exist_ok=True)
        return path

    @routes.get("/ref2va-director/version")
    async def get_director_version(request):
        return web.json_response({
            "ok": True,
            "backend_version": REF2VA_DIRECTOR_BACKEND_VERSION,
            "project_schema_version": PROJECT_VERSION,
            "capabilities": ["project_save", "revision_conflict", "history", "job_snapshot", "safe_take_delete", "safe_delivery_delete", "project_storage", "purge_video_trash", "smooth_merge", "auto_seamless_continuity", "motion_interpolated_delivery", "continuity_acceptance_report"],
        })

    @routes.post("/ref2va-director/render-edit-timeline")
    async def render_edit_timeline(request):
        """Render externally imported/generated videos as an edit, never as H3 references."""
        try:
            import imageio_ffmpeg
            body = await request.json()
            clips = [item for item in (body.get("clips") or []) if isinstance(item, dict) and item.get("usage") == "edit" and item.get("kind") == "video"]
            clips.sort(key=lambda item: (float(item.get("start", 0) or 0), str(item.get("id") or "")))
            if not clips:
                return web.json_response({"ok": False, "error": "编辑时间线中没有视频片段。"}, status=400)
            work = os.path.realpath(os.path.join(folder_paths.get_temp_directory(), "ref2va_timeline_edits", uuid.uuid4().hex))
            os.makedirs(work, exist_ok=True)
            roots = {"input": os.path.realpath(folder_paths.get_input_directory()), "output": os.path.realpath(folder_paths.get_output_directory())}
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            segments = []
            for index, clip in enumerate(clips):
                asset = clip.get("asset") or {}
                asset_type = str(asset.get("type") or "input")
                root = roots.get(asset_type)
                if root is None:
                    raise ValueError("编辑视频只允许来自 input 或 output。")
                source = os.path.realpath(os.path.join(root, str(asset.get("subfolder") or ""), str(asset.get("filename") or "")))
                if os.path.commonpath((root, source)) != root or not os.path.isfile(source):
                    raise ValueError(f"找不到编辑视频：{asset.get('filename') or '未命名'}")
                source_in = max(0.0, float(clip.get("source_in", 0) or 0))
                duration = max(0.05, float(clip.get("duration", 0) or 0))
                segment = os.path.join(work, f"segment-{index:04d}.mp4")
                command = [ffmpeg, "-y", "-ss", f"{source_in:.3f}", "-i", source, "-t", f"{duration:.3f}", "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=24", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", segment]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
                if completed.returncode != 0:
                    raise RuntimeError((completed.stderr or "视频片段转码失败")[-1200:])
                segments.append(segment)
            concat_file = os.path.join(work, "concat.txt")
            with open(concat_file, "w", encoding="utf-8") as handle:
                for path in segments:
                    handle.write("file '" + path.replace("\\", "/").replace("'", "'\\''") + "'\n")
            output = os.path.join(work, "edited-preview.mp4")
            completed = subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output], capture_output=True, text=True, timeout=600)
            if completed.returncode != 0 or not os.path.isfile(output):
                raise RuntimeError((completed.stderr or "编辑视频合并失败")[-1200:])
            relative = os.path.relpath(output, folder_paths.get_temp_directory()).replace("\\", "/")
            subfolder, filename = relative.rsplit("/", 1)
            return web.json_response({"ok": True, "asset": {"filename": filename, "subfolder": subfolder, "type": "temp"}, "message": f"已渲染 {len(segments)} 个剪辑片段。"})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @routes.post("/ref2va-director/upscale-uploaded")
    async def upscale_uploaded_video(request):
        """Run postprocess against an uploaded file or a verified project take."""
        try:
            body = await request.json()
            name = str(body.get("name") or "")
            subfolder = str(body.get("subfolder") or "")
            stored_project_id = str(body.get("project_id") or "")
            if stored_project_id:
                if not valid_project_id(stored_project_id):
                    return web.json_response({"ok": False, "error": "项目视频标识无效。"}, status=400)
                stored_project = load_project_snapshot(stored_project_id)
                if not isinstance(stored_project, dict):
                    return web.json_response({"ok": False, "error": "找不到该项目，无法读取原版视频。"}, status=404)
                shot_id = str(body.get("shot_id") or "")
                take_id = str(body.get("take_id") or "")
                source_kind = str(body.get("source") or "initial")
                if source_kind not in {"initial", "final"}:
                    return web.json_response({"ok": False, "error": "项目视频版本无效。"}, status=400)
                stored_shot = next((item for item in stored_project.get("shots") or [] if str(item.get("id")) == shot_id), None)
                stored_take = next((item for item in (stored_shot or {}).get("takes") or [] if str(item.get("take_id")) == take_id), None)
                if stored_take is None or not stored_take.get("files", {}).get(source_kind):
                    return web.json_response({"ok": False, "error": "找不到所选的项目视频版本。"}, status=404)
                source_path = take_path(stored_project_id, shot_id, take_id, source_kind)
                if source_path is None or not os.path.isfile(source_path):
                    return web.json_response({"ok": False, "error": "所选项目视频文件不存在。"}, status=404)
                source = str(source_path)
            else:
                if not name or any(part in name for part in ("..", "/", "\\")) or any(part in subfolder for part in ("..",)):
                    return web.json_response({"ok": False, "error": "输入视频标识无效。"}, status=400)
                input_root = os.path.realpath(folder_paths.get_input_directory())
                source = os.path.realpath(os.path.join(input_root, subfolder, name))
                if os.path.commonpath((input_root, source)) != input_root or not os.path.isfile(source):
                    return web.json_response({"ok": False, "error": "找不到已上传的视频。"}, status=404)
            engine = str(body.get("engine") or "TE FlashVSR")
            scale = max(1, min(4, int(round(float(body.get("scale") or 2)))))
            video = VideoFromFile(source)
            # ComfyUI's global progress hook normally receives these values
            # from a queued prompt. This endpoint executes an already-uploaded
            # video directly, so initialise an isolated progress identity for
            # TE FlashVSR before it emits its first progress update.
            prompt_server = PromptServer.instance
            if not getattr(prompt_server, "last_prompt_id", None):
                prompt_server.last_prompt_id = f"ref2va-existing-upscale-{uuid.uuid4().hex}"
            if not getattr(prompt_server, "last_node_id", None):
                prompt_server.last_node_id = "ref2va-existing-upscale"
            if engine == "NVIDIA RTX":
                result = Ref2VARTXVideoPostprocess.execute(video, {"resize_type": "scale by multiplier", "scale": float(scale)}, str(body.get("quality") or "HIGH"), "video/Existing_RTX")[0]
            else:
                required = ("TEFlashVSRModelLoader", "TEFlashVSRTuning", "TEFlashVSRRestore")
                missing = [key for key in required if key not in nodes.NODE_CLASS_MAPPINGS]
                if missing:
                    raise RuntimeError("TE FlashVSR 节点未加载：" + ", ".join(missing))
                loader = nodes.NODE_CLASS_MAPPINGS[required[0]]().load("FlashVSR-v1.1", str(body.get("mode") or "tiny"), str(body.get("precision") or "bf16"), "auto")[0]
                settings = nodes.NODE_CLASS_MAPPINGS[required[1]]().create(quality_profile=str(body.get("quality") or "balanced"), intensity=1.0, spatial_strategy=str(body.get("spatial") or "auto"), memory_policy=str(body.get("memory") or "staged"), attention_budget=2.0, kv_retention=3.0, local_radius=11, max_tile_edge=256, blend_overlap=24, preprocess_batch=4, attention_backend=str(body.get("attention") or "auto"))[0]
                components = video.get_components()
                restored = nodes.NODE_CLASS_MAPPINGS[required[2]]().restore(
                    loader, components.images, scale, bool(body.get("color_fix", True)), 0,
                    settings=settings,
                )[0]
                rebuilt = CreateVideo.execute(
                    images=restored, audio=components.audio,
                    fps=float(components.frame_rate), bit_depth=video.get_bit_depth(),
                )
                result = rebuilt.args[0] if hasattr(rebuilt, "args") else rebuilt[0]
            components = result.get_components()
            images = components.images
            _, height, width, _ = images.shape
            out_dir = pending_upscale_directory()
            output_name = f"Existing_Upscale_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(out_dir, output_name)
            result.save_to(output_path, format=Types.VideoContainer("mp4"), codec="auto", metadata=None)
            return web.json_response({"ok": True, "filename": output_name, "subfolder": "ref2va_existing_upscale", "type": "temp"})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/ref2va-director/delete-pending-upscale")
    async def delete_pending_upscale(request):
        try:
            body = await request.json()
            filename = str(body.get("filename") or "")
            if not filename or filename != os.path.basename(filename) or not filename.startswith("Existing_Upscale_") or not filename.lower().endswith(".mp4"):
                return web.json_response({"ok": False, "error": "临时放大文件标识无效。"}, status=400)
            root = pending_upscale_directory()
            path = os.path.realpath(os.path.join(root, filename))
            if os.path.commonpath((root, path)) != root:
                return web.json_response({"ok": False, "error": "临时放大文件路径无效。"}, status=400)
            if os.path.isfile(path):
                os.remove(path)
            return web.json_response({"ok": True})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"删除临时放大文件失败：{exc}"}, status=500)

    @routes.get("/ref2va-director/project/{project_id}")
    async def get_director_project(request):
        project_id = request.match_info.get("project_id", "")
        try:
            if not valid_project_id(project_id):
                return web.json_response({"ok": False, "error": "项目标识无效。"}, status=400)
            data = recover_archived_takes(project_id)
            return web.json_response({"ok": True, "project": data})
        except FileNotFoundError:
            return web.json_response({"ok": False, "error": "项目尚未保存。"}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @routes.post("/ref2va-director/project/save")
    async def save_director_project(request):
        try:
            body = await request.json()
            project = body.get("project")
            if not isinstance(project, dict) or not valid_project_id(project.get("project_id")):
                return web.json_response({"ok": False, "error": "项目数据或项目标识无效。"}, status=400)
            expected = body.get("expected_revision")
            expected = int(expected) if expected is not None else None
            saved = save_editor_project(project, expected_revision=expected, reason=str(body.get("reason") or "auto_save"))
            return web.json_response({"ok": True, "project": saved, "revision": saved["revision"], "updated_at": saved["updated_at"]})
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("REVISION_CONFLICT:"):
                current_revision = int(message.split(":", 1)[1])
                current = load_project_snapshot(str((body.get("project") or {}).get("project_id") or ""))
                return web.json_response({"ok": False, "error": "项目已在其他页面或后台更新，请先合并或重新加载。", "code": "revision_conflict", "current_revision": current_revision, "project": current}, status=409)
            return web.json_response({"ok": False, "error": message}, status=400)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"ok": False, "error": f"项目保存请求无效：{exc}"}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"项目保存失败：{exc}"}, status=500)

    @routes.get("/ref2va-director/project/{project_id}/history")
    async def get_director_project_history(request):
        project_id = request.match_info.get("project_id", "")
        if not valid_project_id(project_id):
            return web.json_response({"ok": False, "error": "项目标识无效。"}, status=400)
        return web.json_response({"ok": True, "history": list_project_history(project_id, 50)})

    @routes.get("/ref2va-director/project/{project_id}/storage")
    async def get_director_project_storage(request):
        project_id = request.match_info.get("project_id", "")
        try:
            if not valid_project_id(project_id):
                return web.json_response({"ok": False, "error": "项目标识无效。"}, status=400)
            return web.json_response({"ok": True, "storage": project_storage_summary(project_id)})
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"读取项目存储失败：{exc}"}, status=500)


    @routes.post("/ref2va-director/project/purge-video-trash")
    async def purge_director_project_video_trash(request):
        try:
            body = await request.json()
            project_id = str(body.get("project_id") or "")
            if not valid_project_id(project_id):
                return web.json_response({"ok": False, "error": "项目标识无效。"}, status=400)
            result = purge_project_video_trash(project_id)
            return web.json_response({"ok": True, "message": "当前项目的视频回收区已永久清空。", "result": result})
        except RuntimeError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"清空视频回收区失败：{exc}"}, status=500)

    @routes.post("/ref2va-director/merge-selected")
    async def merge_selected_director_shots(request):
        try:
            body = await request.json()
            project_id = str(body.get("project_id") or "")
            shot_ids = [str(value) for value in (body.get("shot_ids") or []) if value]
            if not project_id or not shot_ids:
                return web.json_response({"ok": False, "error": "请至少选择一个已生成镜头。"}, status=400)
            path = project_root(project_id) / "project.json"
            if not path.is_file():
                return web.json_response({"ok": False, "error": "项目尚未生成任何镜头。"}, status=404)
            with path.open("r", encoding="utf-8") as handle:
                project = json.load(handle)
            selections = {str(item.get("shot_id")): item for item in (body.get("selections") or []) if isinstance(item, dict) and item.get("shot_id")}
            if selections:
                for shot in project.get("shots", []):
                    selection = selections.get(str(shot.get("id") or ""))
                    if not selection:
                        continue
                    take_id = str(selection.get("take_id") or "")
                    record = load_take(project_id, str(shot.get("id") or ""), take_id)
                    requested_source = "final" if selection.get("source") == "final" else "initial"
                    if not record or not (record.get("files") or {}).get(requested_source):
                        return web.json_response({"ok": False, "error": f"{shot.get('name') or '镜头'} 选择的合并版本不存在。"}, status=400)
                    shot["selected_take_id"] = take_id
                    shot["merge_source"] = requested_source
                source = "per_shot"
            else:
                source = "initial" if body.get("source") == "initial" else "final_if_available"
            motion_interpolation = bool(body.get("motion_interpolation", False))
            source_manifest = []
            for shot in sorted((item for item in project.get("shots", []) if str(item.get("id")) in set(shot_ids)), key=lambda item: int(item.get("order", 0))):
                take_id = str(shot.get("selected_take_id") or "")
                record = load_take(project_id, str(shot.get("id") or ""), take_id) or {}
                files = record.get("files") if isinstance(record.get("files"), dict) else {}
                selected_key = ("final" if shot.get("merge_source") == "final" else "initial") if source == "per_shot" else ("initial" if source == "initial" or not files.get("final") else "final")
                source_manifest.append({
                    "shot_id": str(shot.get("id") or ""), "shot_name": str(shot.get("name") or ""),
                    "take_id": take_id, "source": selected_key,
                    "filename": files.get(selected_key), "output_profile": record.get("output_profile") or {},
                })
            delivery_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            merged = concat_selected_takes(
                project, source=source, shot_ids=shot_ids,
                destination_name=f"selected-timeline-{delivery_id}.mp4",
                motion_interpolation=motion_interpolation,
            )
            relative = os.path.relpath(str(merged), folder_paths.get_output_directory())
            subfolder, filename = os.path.split(relative)
            baseline = merged.with_name(f"{merged.stem}-24fps.mp4") if motion_interpolation else None
            report_path = merged.with_suffix(".continuity.json")
            continuity_report = None
            if report_path.is_file():
                with report_path.open("r", encoding="utf-8") as handle:
                    continuity_report = json.load(handle)
            return web.json_response({
                "ok": True, "filename": filename, "subfolder": subfolder.replace("\\", "/"),
                "type": "output", "motion_interpolation": motion_interpolation,
                "fps": 48 if motion_interpolation else 24,
                "baseline_filename": baseline.name if baseline and baseline.is_file() else None,
                "continuity_report": continuity_report,
                "source_manifest": source_manifest,
            })
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.post("/ref2va-director/unload-models")
    async def unload_director_models(request):
        try:
            running, queued = PromptServer.instance.prompt_queue.get_current_queue()
            if running or queued:
                return web.json_response({"ok": False, "error": "当前仍有生成任务正在运行或等待，不能卸载模型。"}, status=409)
            before = comfy.model_management.get_free_memory(comfy.model_management.get_torch_device())
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(True)
            after = comfy.model_management.get_free_memory(comfy.model_management.get_torch_device())
            return web.json_response({"ok": True, "free_before": before, "free_after": after, "message": "模型已卸载，显存缓存已清理。"})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"卸载模型失败：{exc}"}, status=500)

    @routes.post("/ref2va-director/delete-delivery")
    async def delete_director_delivery(request):
        try:
            body = await request.json()
            project_id = str(body.get("project_id") or "")
            merged_filename = str(body.get("merged_filename") or "")
            if not project_id or not merged_filename:
                return web.json_response({"ok": False, "error": "缺少项目或合并视频标识。"}, status=400)
            running, queued = PromptServer.instance.prompt_queue.get_current_queue()
            if running or queued:
                return web.json_response({"ok": False, "error": "当前仍有生成任务正在运行或等待，不能删除合并视频。"}, status=409)
            result = delete_merged_delivery(folder_paths.get_output_directory(), project_id, merged_filename)
            return web.json_response({"ok": True, "result": result, "recoverable": True,
                                      "message": "本次合并视频、验收报告和接缝帧条已移入项目回收区。"})
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except RuntimeError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"删除合并视频失败：{exc}"}, status=500)

    @routes.post("/ref2va-director/delete-take")
    async def delete_director_take(request):
        try:
            body = await request.json()
            project_id = str(body.get("project_id") or "")
            shot_id = str(body.get("shot_id") or "")
            take_id = str(body.get("take_id") or "")
            if not project_id or not shot_id or not take_id:
                return web.json_response({"ok": False, "error": "缺少项目、镜头或版本标识。"}, status=400)
            running, queued = PromptServer.instance.prompt_queue.get_current_queue()
            if running or queued:
                return web.json_response({"ok": False, "error": "当前仍有生成任务正在运行或等待，不能删除视频版本。"}, status=409)
            result = delete_archived_take(folder_paths.get_output_directory(), project_id, shot_id, take_id)
            return web.json_response({
                "ok": True,
                "project": result["project"],
                "shot": result["shot"],
                "removed_files": result["removed_files"],
                "recoverable": result["recoverable"],
                "message": "视频版本已移入项目回收区。",
            })
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"删除视频版本失败：{exc}"}, status=500)

    @routes.post("/ref2va-director/delete-selected-video")
    async def delete_director_selected_video(request):
        try:
            body = await request.json()
            project_id = str(body.get("project_id") or "")
            shot_id = str(body.get("shot_id") or "")
            take_id = str(body.get("take_id") or "")
            source = str(body.get("source") or "")
            if not project_id or not shot_id or not take_id:
                return web.json_response({"ok": False, "error": "缺少项目、镜头或版本标识。"}, status=400)
            running, queued = PromptServer.instance.prompt_queue.get_current_queue()
            if running or queued:
                return web.json_response({"ok": False, "error": "当前仍有生成任务正在运行或等待，不能删除超分视频。"}, status=409)
            result = delete_selected_video(folder_paths.get_output_directory(), project_id, shot_id, take_id, source)
            return web.json_response({
                "ok": True,
                "project": result["project"],
                "shot": result["shot"],
                "removed_file": result["removed_file"],
                "recoverable": result["recoverable"],
                "message": "当前选中视频已移入项目回收区，同版本的另一条视频已保留。",
            })
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"删除当前选中视频失败：{exc}"}, status=500)

    routes._ref2va_director_routes = True


_install_unified_director_routes()

NODE_CLASS_MAPPINGS={"Ref2VAImageGate":Ref2VAImageGate,"Ref2VAAudioGate":Ref2VAAudioGate,"Ref2VARegionSwitchboard":Ref2VARegionSwitchboard,"Ref2VAWithGlobalGates":Ref2VAWithGlobalGates,"Ref2VAAllInOne":Ref2VAAllInOne,"Ref2VAStoryboardPromptPanel":Ref2VAStoryboardPromptPanel,"Ref2VAStoryboardVideoArchive":Ref2VAStoryboardVideoArchive,"Ref2VAStoryboardAutoArchive":Ref2VAStoryboardAutoArchive,"Ref2VAStoryboardTailFrameUpload":Ref2VAStoryboardTailFrameUpload,"Ref2VADecodeCreateVideo":Ref2VADecodeCreateVideo,"Ref2VAPreviewVideo":Ref2VAPreviewVideo,"Ref2VARTXVideoPostprocess":Ref2VARTXVideoPostprocess,"Ref2VAModelLoader":Ref2VAModelLoader,"Ref2VAGenerationConditioning":Ref2VAGenerationConditioning,"Ref2VAMultiPassSampler":Ref2VAMultiPassSampler,"Ref2VASamplingControlPanel":Ref2VASamplingControlPanel,"Ref2VAPrimarySamplingPanel":Ref2VAPrimarySamplingPanel,"Ref2VARefinementPanel":Ref2VARefinementPanel,"Ref2VASameResolutionRefinementPanel":Ref2VASameResolutionRefinementPanel,"Ref2VAH3LatentRefinementPanel":Ref2VAH3LatentRefinementPanel,"Ref2VARefinementModeSelector":Ref2VARefinementModeSelector,"Ref2VAUnifiedMultiPassSampler":Ref2VAUnifiedMultiPassSampler,"Ref2VASamplerVideoOutputPanel":Ref2VASamplerVideoOutputPanel,"Ref2VASequenceWorkbenchV01":Ref2VASequenceWorkbenchV01,"Ref2VASequenceWorkbenchV03":Ref2VASequenceWorkbenchV03,"Ref2VAUnifiedDirectorStudio":Ref2VAUnifiedDirectorStudio,"Ref2VAUnifiedDirectorRunner":Ref2VAUnifiedDirectorRunner}
NODE_DISPLAY_NAME_MAPPINGS={"Ref2VAImageGate":"Ref2VA 图片/视频参考开关","Ref2VAAudioGate":"Ref2VA 音频参考开关","Ref2VARegionSwitchboard":"Ref2VA 参考区域开关（图片 / 视频 / 音频）","Ref2VAWithGlobalGates":"Ref2VA：多参考图 / 视频 / 音频（总开关）","Ref2VAAllInOne":"Ref2VA：多参考图 / 视频 / 音频（集成时长与分辨率）","Ref2VAStoryboardPromptPanel":"Ref2VA：连续分镜控制台（3段）","Ref2VAStoryboardVideoArchive":"Ref2VA：连续分镜视频归档","Ref2VAStoryboardAutoArchive":"Ref2VA：连续分镜自动成片归档","Ref2VAStoryboardTailFrameUpload":"Ref2VA：上传上一镜头并自动取尾帧","Ref2VADecodeCreateVideo":"Ref2VA：解码并创建视频","Ref2VAPreviewVideo":"Ref2VA：实时视频预览","Ref2VARTXVideoPostprocess":"Ref2VA：RTX 视频放大并保存","Ref2VAModelLoader":"Ref2VA：模型加载面板（UNet / CLIP / 视频VAE / 音频VAE / Turbo LoRA）","Ref2VAGenerationConditioning":"Ref2VA：文生视频 / 首尾帧生视频","Ref2VAMultiPassSampler":"Ref2VA 多遍细化采样（兼容旧工作流）","Ref2VASamplingControlPanel":"Ref2VA 采样控制面板（旧版）","Ref2VAPrimarySamplingPanel":"Ref2VA 第一次采样控制面板","Ref2VARefinementPanel":"Ref2VA 第二次 / N次采样控制面板（旧版）","Ref2VASameResolutionRefinementPanel":"Ref2VA 二次采样模板：同分辨率细化","Ref2VAH3LatentRefinementPanel":"Ref2VA 二次采样模板：H3 Latent 超分","Ref2VARefinementModeSelector":"Ref2VA 二次采样模式选择器","Ref2VAUnifiedMultiPassSampler":"Ref2VA 采样执行器","Ref2VASamplerVideoOutputPanel":"Ref2VA：最初视频 / 最终视频输出控制板","Ref2VASequenceWorkbenchV01":"Ref2VA：连续镜头工作台 v0.1","Ref2VASequenceWorkbenchV03":"Ref2VA：连续镜头工作台 v0.3"}
NODE_CLASS_MAPPINGS["Ref2VADirectorSystem"] = Ref2VADirectorSystem
NODE_CLASS_MAPPINGS["Ref2VADirectorDelivery"] = Ref2VADirectorDelivery
NODE_DISPLAY_NAME_MAPPINGS["Ref2VAUnifiedDirectorStudio"] = "梦镜 DreamShot"
NODE_DISPLAY_NAME_MAPPINGS["Ref2VAUnifiedDirectorRunner"] = "梦镜 DreamShot"
NODE_DISPLAY_NAME_MAPPINGS["Ref2VADirectorSystem"] = "Ref2VA：导演系统加载器"
NODE_DISPLAY_NAME_MAPPINGS["Ref2VADirectorDelivery"] = "Ref2VA：导演台交付与预览"
WEB_DIRECTORY="./web"

NODE_CLASS_MAPPINGS["Ref2VAVideoPostprocess"] = Ref2VAVideoPostprocess
NODE_DISPLAY_NAME_MAPPINGS["Ref2VAVideoPostprocess"] = "Ref2VA：视频后处理台（RTX / TE）"



