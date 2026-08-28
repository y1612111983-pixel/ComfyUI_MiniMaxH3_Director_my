import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// An original compact Director Studio UI: one project rail + one editor.
// It intentionally does not imitate either installed H3 Director interface.
const TYPE = "Ref2VAUnifiedDirectorRunner";
const FRONTEND_VERSION = "1.10.1";
const PROJECT_SCHEMA_VERSION = 3;
const MODES = [
  ["t2v", "文生视频"], ["i2v", "图生视频"], ["fl2v", "首帧 / 尾帧参考视频"],
  ["ref2va", "参考生视频"], ["continuous_ref2va", "连续参考生视频"],
];
const modeLabel = (key) => MODES.find(([id]) => id === key)?.[1] || "参考生视频";
const STATUS_LABELS = { draft: "待生成", queued: "等待队列", preparing_models: "准备模型", sampling: "正在采样", decoding: "正在解码", upscaling: "正在超分", saving: "正在保存", generated: "已完成", cached: "已复用缓存", failed: "生成失败", stopped: "已终止", stale: "参数已修改" };
const defaultFollowingMode = (firstMode) => firstMode === "t2v" ? "continuous_ref2va" : (MODES.some(([id]) => id === firstMode) ? firstMode : "continuous_ref2va");
const makeId = () => `shot-${Math.random().toString(36).slice(2, 14)}`;
const fallback = () => ({
  schema_version: PROJECT_SCHEMA_VERSION, project_id: `project-${Math.random().toString(36).slice(2, 14)}`, revision: 0, updated_at: new Date().toISOString(),
  name: "梦镜 DreamShot", global_prompt: "", global_constraint_prompt: "", global_assets: { images: [], videos: [], audios: [] }, settings: { fps: 24, delivery_source: "final_if_available", default_context_frames: 22, delivery_motion_interpolation: false, video_library: [] },
  shots: [1, 2, 3].map((n) => ({ id: makeId(), order: n, name: `镜头 ${n}`, enabled: true, duration_seconds: 5, fps: 24, mode: n === 1 ? "t2v" : "continuous_ref2va", mode_inherited: n > 1, prompt: n === 1 ? "建立镜头：明确主体、场景、光线与动作起点。" : "承接上一镜头，描述新的动作、运镜或情绪变化。", assets: { first_frame: null, last_frame: null, images: [], videos: [], audios: [] }, timeline: { enabled: false, prompt: "", generation_start: 0, generation_end: 5, snap_seconds: .25, clips: [] }, continuity: { enabled: n > 1, strategy: n > 1 ? "auto_seamless" : "off", context_frames: 22, source: "initial" }, takes: [], selected_take_id: null, status: "draft" })), jobs: [], active_job_id: null,
});
const parse = (value) => { try { const x = JSON.parse(value || ""); return x && Array.isArray(x.shots) ? x : fallback(); } catch { return fallback(); } };
const outputViewUrl = (projectId, shotId, takeId, filename) => {
  const subfolder = `video/Ref2VA_Director/${projectId}/shots/${shotId}/takes/${takeId}`;
  return `/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}&type=output`;
};
const storedViewUrl = (asset) => `/view?filename=${encodeURIComponent(asset?.filename || "")}&subfolder=${encodeURIComponent(asset?.subfolder || "")}&type=${encodeURIComponent(asset?.type || "output")}${asset?.cacheKey ? `&v=${encodeURIComponent(asset.cacheKey)}` : ""}`;

app.registerExtension({
  name: "Ref2VA.UnifiedDirectorStudio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TYPE) return;
    const oldCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = oldCreated?.apply(this, arguments);
      if (this.ref2vaDirectorStudioMounted) return result;
      this.ref2vaDirectorStudioMounted = true;
      // The studio owns its preview panel; suppress ComfyUI's native output
      // media strip below the node to avoid the duplicate floating image.
      this.hideOutput = true;
      const dataWidget = this.widgets?.find((w) => w.name === "project_data");
      if (!dataWidget) return result;
      // The Runner has many implementation parameters, but they belong in the
      // compact Director Studio—not as a second native node UI above it.
      // Keeping both caused the stacked panel and automatic shrink shown in
      // the user's screenshot.  v1 keeps its clean public socket contract;
      // these hidden widgets are parameters controlled by the studio UI.
      for (const widget of this.widgets || []) {
        widget.type = "converted-widget";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
      }
      // Older saved workflows serialized the hidden implementation widgets as
      // unconnected input sockets.  New ComfyUI validates those sockets as
      // required and blocks the queue.  The values already live in widgets;
      // remove only their duplicate sockets while retaining genuine sockets.
      const hiddenWidgetNames = new Set((this.widgets || []).map((widget) => widget.name));
      this.inputs = (this.inputs || []).filter((input) => !hiddenWidgetNames.has(input.name));
      // The converted hidden widget is not guaranteed to survive every
      // workspace/page remount.  Keep two serialized mirrors and choose the
      // newest valid project instead of silently falling back to a fresh
      // three-shot/5-second project.
      this.properties = this.properties || {};
      const validProject = (value) => {
        try { const item = typeof value === "string" ? JSON.parse(value) : value; return item && Array.isArray(item.shots) ? item : null; }
        catch { return null; }
      };
      const nodeStorageKey = `ref2va-director-state-${String(this.id)}`;
      const storedNodeProject = validProject(this.properties.ref2va_project_data);
      const storedBrowserProject = (() => { try { return validProject(localStorage.getItem(nodeStorageKey)); } catch { return null; } })();
      const widgetProject = validProject(dataWidget.value);
      const candidates = [widgetProject, storedNodeProject, storedBrowserProject].filter(Boolean);
      let project = candidates.sort((a, b) => Number(b.ui_saved_at || 0) - Number(a.ui_saved_at || 0))[0] || fallback();
      const hiddenDefaults = {
        ref2va_unet_name: "minimax_h3_ref2va_int8_convrot.safetensors",
        fl2va_unet_name: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        weight_dtype: "default", clip_name: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        clip_type: "minimax", video_vae_name: "minimax_h3_video_vae_fp16.safetensors",
        audio_vae_name: "minimax_h3_audio_vae_fp32.safetensors", enable_turbo_lora: true,
        turbo_lora_name: "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        ref2va_turbo_lora_name: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        fl2v_turbo_lora_name: "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        final_upscale_method: "关闭", te_flashvsr_model: "FlashVSR-v1.1",
        te_flashvsr_mode: "tiny", te_flashvsr_precision: "bf16", te_flashvsr_scale: 2,
        te_flashvsr_quality: "balanced", te_flashvsr_spatial: "auto",
        te_flashvsr_memory: "auto", te_flashvsr_attention: "sparse_sage2",
        te_flashvsr_color_fix: true,
        turbo_lora_strength: 1.0,
        lora_stack_json: "[]",
        aspect_ratio: "16:9", megapixels: 1.0, seed_mode: "随机", noise_seed: 0,
        scheduler: "beta", steps: 4, denoise: 1.0, sampler_name: "euler",
        main_extra_steps: 1, main_start_at_sigma: 0.70, main_spacing: "cosine",
        run_scope: "仅当前镜头", enable_final_video: false,
        second_sampling_mode: "H3 Latent 超分", latent_upscale_model: "",
        second_aspect_ratio: "16:9", second_megapixels: 1.5,
        upscale_passes: 1, passes: 1, refine_scheduler: "beta",
        refine_steps: 3, refine_denoise: 0.30, refine_extra_steps: 1,
        refine_start_at_sigma: 0.60, refine_spacing: "cosine",
        enable_rtx_upscale: false, rtx_scale: 2.0, rtx_quality: "HIGH",
        rtx_filename_prefix: "video/H3_Final_RTX",
      };
      const normalizeHiddenWidgets = () => {
        for (const [name, fallbackValue] of Object.entries(hiddenDefaults)) {
          const widget = this.widgets?.find((item) => item.name === name);
          if (!widget) continue;
          const choices = Array.isArray(widget.options?.values) ? widget.options.values : null;
          if (choices) {
            if (!choices.includes(widget.value)) widget.value = choices.includes(fallbackValue) ? fallbackValue : choices[0];
          } else if (typeof fallbackValue === "number") {
            const value = Number(widget.value);
            const min = Number(widget.options?.min);
            const max = Number(widget.options?.max);
            widget.value = Number.isFinite(value) ? Math.min(Number.isFinite(max) ? max : value, Math.max(Number.isFinite(min) ? min : value, value)) : fallbackValue;
          } else if (typeof fallbackValue === "boolean") {
            if (typeof widget.value !== "boolean") widget.value = fallbackValue;
          }
        }
      };
      normalizeHiddenWidgets();
      const renumberShots = () => {
        project.shots.forEach((shot, index) => {
          shot.order = index + 1;
          shot.name = `镜头 ${index + 1}`;
        });
      };
      renumberShots();
      let selected = String(project.active_shot_id || project.shots[0]?.id || "");
      const selectedShotIds = new Set(Array.isArray(project.selected_shot_ids) ? project.selected_shot_ids : []);
      let mergedPreview = null;
      let mergedBaselinePreview = null;
      let mergedSourceManifest = [];
      // A project-origin video selected from the library.  It is kept separate
      // from uploads so sending an original take to postprocess never turns it
      // into a reference asset or re-uploads it through input/.
      let existingUpscaleSource = null;
      let existingUpscaleOpen = false;
      let continuityReport = null;
      let globalDrawerOpen = false;
      let shotSettingsOpen = true;
      let versionOpen = true;
      let shotResourceOpen = false;
      let shotTimelineOpen = false;
      let selectedTimelineClipId = null;
      let editPreviewAsset = null;
      let livePreviewUrl = "";
      let liveCompletedPreview = null;
      let liveProgressValue = 0;
      let liveProgressMax = 0;
      let liveRunning = false;
      let liveShotId = null;
      const completedLiveShotIds = new Set();
      let livePreviewImage = null;
      let liveProgressFill = null;
      let liveStatusText = null;
      let liveStopButton = null;
      let saveStatusElement = null;
      let versionStatusElement = null;
      let mergeButton = null;
      let mergeProgressPanel = null;
      let mergeProgressText = null;
      let mergeElapsedText = null;
      let mergeProgressMover = null;
      let mergeRunning = false;
      let mergeStartedAt = 0;
      let mergeTimer = 0;
      let mergeMessage = "";
      let mergeMessageColor = "#9fc8e7";
      let saveTimer = 0;
      let saveInFlight = false;
      let saveQueued = false;
      let localEditSerial = 0;
      let saveState = "本地已恢复，等待后台同步";
      let saveStateColor = "#f2c56b";
      let backendVersion = null;
      let backendSchemaVersion = null;
      let conflictProject = null;
      const profileLabel = (take, source) => {
        if (source === "initial") return "H3 原始生成";
        const p = take?.output_profile || {};
        const parts = [];
        if (p.second_sampling_mode === "H3 Latent 超分") parts.push(`H3 Latent ${p.second_megapixels || "自动"}MP`);
        else if (p.enable_final_video) parts.push(`细化 ${p.passes || 1} 次`);
        if (p.final_upscale_method === "NVIDIA RTX") parts.push(`RTX ${p.rtx_scale || 2}x ${p.rtx_quality || "HIGH"}`);
        if (p.final_upscale_method === "TE FlashVSR") parts.push(`TE FlashVSR ${p.te_flashvsr_scale || 2}x ${p.te_flashvsr_mode || "tiny"} ${p.te_flashvsr_quality || "balanced"}`);
        return parts.join(" + ") || "最终处理视频";
      };
      const root = document.createElement("div");
      root.style.cssText = "box-sizing:border-box;width:100%;min-width:880px;background:#0d1a29;border:1px solid #356088;border-radius:10px;color:#dceeff;font-family:system-ui,'Microsoft YaHei',sans-serif;padding:12px 12px 24px;display:flex;flex-direction:column;gap:10px;overflow:hidden;";
      if (!document.getElementById("ref2va-director-drawer-style")) {
        const drawerStyle = document.createElement("style"); drawerStyle.id = "ref2va-director-drawer-style";
        drawerStyle.textContent = `
          summary.ref2va-drawer-summary { display:flex !important;align-items:center;gap:9px;list-style:none; }
          summary.ref2va-drawer-summary::-webkit-details-marker { display:none; }
          summary.ref2va-drawer-summary::before { content:"";flex:0 0 auto;width:0;height:0;border-top:7px solid transparent;border-bottom:7px solid transparent;border-left:10px solid #72c5ff;transform-origin:4px 7px;transition:transform .12s ease; }
          details[open] > summary.ref2va-drawer-summary::before { transform:rotate(90deg); }
          @keyframes ref2va-merge-progress { from { transform:translateX(-110%); } to { transform:translateX(310%); } }
        `;
        document.head.append(drawerStyle);
      }
      const markDrawerSummary = (summary) => { summary.classList.add("ref2va-drawer-summary"); return summary; };
      let fitToContent = () => {};
      const priorResize = this.onResize;
      this.onResize = function (size) {
        if (Array.isArray(size) || ArrayBuffer.isView(size)) size[0] = Math.max(900, Number(size[0] || 0));
        if (this.size) this.size[0] = Math.max(900, Number(this.size[0] || 0));
        root.style.width = `${Math.max(880, Number(this.size?.[0] || size?.[0] || 900) - 22)}px`;
        const result = priorResize?.apply(this, arguments);
        requestAnimationFrame(fitToContent);
        return result;
      };
      const mount = this.addDOMWidget("ref2va_director_studio", "div", root, { serialize: false, hideOnZoom: false });
      root.style.width = `${Math.max(880, Number(this.size?.[0] || 900) - 22)}px`;
      // Migrate already-open v1 workflows to the unified single-node layout.
      // Only directly connected legacy companion nodes are removed; unrelated
      // loaders or delivery nodes elsewhere on the canvas are left untouched.
      const legacyTypes = new Set(["Ref2VADirectorSystem", "Ref2VADirectorDelivery"]);
      const nodeTypeName = (node) => String(node?.type || node?.comfyClass || node?.constructor?.comfyClass || "");
      const removeLegacyCompanions = () => {
        const graph = this.graph;
        if (!graph) return false;
        const connectedLegacy = new Set();
        const linkIds = new Set();
        for (const input of this.inputs || []) if (input?.link != null) linkIds.add(input.link);
        for (const output of this.outputs || []) for (const linkId of output?.links || []) if (linkId != null) linkIds.add(linkId);
        for (const linkId of linkIds) {
          const link = graph.links?.[linkId] || graph.links?.get?.(linkId);
          if (!link) continue;
          const origin = graph.getNodeById?.(link.origin_id);
          const target = graph.getNodeById?.(link.target_id);
          if (origin && origin !== this && legacyTypes.has(nodeTypeName(origin))) connectedLegacy.add(origin);
          if (target && target !== this && legacyTypes.has(nodeTypeName(target))) connectedLegacy.add(target);
        }
        for (const legacyNode of connectedLegacy) graph.remove?.(legacyNode);
        for (let index = (this.inputs?.length || 0) - 1; index >= 0; index--) {
          if (this.inputs[index]?.name === "system") this.removeInput?.(index);
        }
        for (let index = (this.outputs?.length || 0) - 1; index >= 0; index--) {
          if (["director_result", "导演结果"].includes(this.outputs[index]?.name)) this.removeOutput?.(index);
        }
        if (connectedLegacy.size || linkIds.size) {
          graph.setDirtyCanvas?.(true, true);
          return true;
        }
        return false;
      };
      // Connections from serialized workflows may be restored just after
      // onNodeCreated, so run the narrow migration again after graph loading.
      setTimeout(removeLegacyCompanions, 0);
      setTimeout(removeLegacyCompanions, 250);
      setTimeout(removeLegacyCompanions, 1000);
      // Measure the actual children, not root.scrollHeight.  ComfyUI stretches
      // the DOM host to the current node height, so using scrollHeight here
      // creates a feedback loop that makes the node taller on every render.
      const contentHeight = () => {
        const children = [...root.children];
        if (!children.length) return 520;
        return Math.max(520, Math.ceil(Math.max(...children.map((child) => child.offsetTop + child.offsetHeight)) + 24));
      };
      mount.computeSize = () => [Math.max(900, Number(this.size?.[0] || 900)), contentHeight()];
      let fitting = false;
      fitToContent = () => {
        if (fitting) return;
        const width = Math.max(900, Number(this.size?.[0] || 0));
        const requiredHeight = contentHeight() + 36;
        if (Number(this.size?.[1] || 0) === requiredHeight && Number(this.size?.[0] || 0) >= 900) return;
        fitting = true;
        this.setSize?.([width, requiredHeight]);
        fitting = false;
      };
      const updateStatusBar = () => {
        if (saveStatusElement) { saveStatusElement.textContent = saveState; saveStatusElement.style.color = saveStateColor; }
        if (versionStatusElement) {
          const backend = backendVersion || "未连接";
          versionStatusElement.textContent = `前端 ${FRONTEND_VERSION} · 后台 ${backend} · 项目结构 ${PROJECT_SCHEMA_VERSION}`;
          versionStatusElement.style.color = backendVersion === FRONTEND_VERSION && Number(backendSchemaVersion) === PROJECT_SCHEMA_VERSION ? "#63e3a2" : "#ffb36b";
        }
      };
      const setSaveState = (message, color = "#9fc8e7") => { saveState = message; saveStateColor = color; updateStatusBar(); };
      const mergeSaveAcknowledgement = (backendProject) => {
        if (!backendProject?.shots) return;
        // Never replace `project` here. Textareas and controls created by the
        // current render hold references to its shot objects. Replacing the
        // root after an earlier save response would make subsequent typing go
        // into detached objects and disappear on the next shot switch.
        project.revision = Number(backendProject.revision || project.revision || 0);
        project.updated_at = backendProject.updated_at || project.updated_at;
        project.jobs = backendProject.jobs || project.jobs || [];
        project.active_job_id = backendProject.active_job_id || null;
        const localById = new Map((project.shots || []).map((item) => [String(item.id), item]));
        for (const incoming of backendProject.shots) {
          const local = localById.get(String(incoming.id));
          if (!local) continue;
          // These fields are owned by generation and deletion endpoints. All
          // editable fields deliberately remain on the live local object.
          local.takes = incoming.takes || [];
          local.selected_take_id = incoming.selected_take_id || null;
          const staleActiveStatus = (completedLiveShotIds.has(String(local.id)) || (!liveRunning && !project.active_job_id)) && ["queued", "preparing_models", "sampling", "decoding", "upscaling", "saving"].includes(String(incoming.status || ""));
          local.status = staleActiveStatus && local.takes?.length ? "generated" : (incoming.status || local.status);
        }
      };
      const persistBackend = async (reason = "auto_save") => {
        if (liveRunning) {
          saveQueued = true;
          setSaveState("生成中：修改已保存在本地，生成完成后同步后台", "#f2c56b");
          return;
        }
        if (saveInFlight) { saveQueued = true; return; }
        saveInFlight = true; saveQueued = false; setSaveState("正在保存…", "#80c8ff");
        try {
          const response = await api.fetchApi("/ref2va-director/project/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ project, expected_revision: Number(project.revision || 0), reason }) });
          let body = null; try { body = await response.json(); } catch (_) { /* handled below */ }
          if (response.status === 409 && body?.code === "revision_conflict") {
            conflictProject = body.project || null;
            setSaveState(`保存冲突：后台已有更新 r${body.current_revision ?? "?"}，请重新加载`, "#ff8f8f");
            return;
          }
          if (!response.ok || !body?.ok) {
            const oldBackend = response.status === 404 || response.status === 405;
            throw new Error(oldBackend ? "当前后台尚未加载稳定版保存接口，需要重启 ComfyUI" : (body?.error || `HTTP ${response.status}`));
          }
          mergeSaveAcknowledgement(body.project);
          dataWidget.value = JSON.stringify(project);
          this.properties.ref2va_project_data = dataWidget.value;
          try { localStorage.setItem(nodeStorageKey, dataWidget.value); } catch (_) {}
          setSaveState(`已保存 · r${project.revision}`, "#63e3a2");
        } catch (error) {
          setSaveState(`保存失败：${error?.message || error}`, "#ff8f8f");
        } finally {
          saveInFlight = false;
          if (saveQueued) { saveQueued = false; persistBackend("queued_change"); }
        }
      };
      const scheduleBackendSave = (reason = "auto_save") => {
        window.clearTimeout(saveTimer);
        if (liveRunning) {
          saveQueued = true;
          setSaveState("生成中：修改已保存在本地，生成完成后同步后台", "#f2c56b");
          return;
        }
        setSaveState("有未保存修改", "#f2c56b");
        saveTimer = window.setTimeout(() => persistBackend(reason), 450);
      };
      const save = (reason = "auto_save", syncBackend = true) => {
        if (syncBackend) localEditSerial += 1;
        project.active_shot_id = selected;
        project.selected_shot_ids = [...selectedShotIds];
        project.shots.forEach((s, i) => { s.order = i + 1; });
        project.schema_version = PROJECT_SCHEMA_VERSION;
        project.ui_saved_at = Date.now();
        const serialized = JSON.stringify(project);
        dataWidget.value = serialized;
        dataWidget.options = { ...(dataWidget.options || {}), serialize: true };
        dataWidget.serializeValue = async () => serialized;
        this.properties.ref2va_project_data = serialized;
        try { localStorage.setItem(nodeStorageKey, serialized); } catch (_) { /* node property remains the durable backup */ }
        if (syncBackend) scheduleBackendSave(reason);
      };
      const checkBackendVersion = async () => {
        try {
          const response = await api.fetchApi("/ref2va-director/version");
          const body = await response.json();
          if (!response.ok || !body?.ok) throw new Error(body?.error || `HTTP ${response.status}`);
          backendVersion = String(body.backend_version || "未知");
          backendSchemaVersion = Number(body.project_schema_version || 0);
        } catch (_) {
          backendVersion = null; backendSchemaVersion = null;
        }
        updateStatusBar();
      };
      const loadAuthoritativeProject = async () => {
        const editSerialAtRequest = localEditSerial;
        try {
          const response = await api.fetchApi(`/ref2va-director/project/${encodeURIComponent(project.project_id)}`);
          const body = await response.json();
          if (response.status === 404) { persistBackend("initial_project_save"); return; }
          if (!response.ok || !body?.ok || !body.project?.shots) throw new Error(body?.error || `HTTP ${response.status}`);
          const backendProject = body.project;
          if (localEditSerial !== editSerialAtRequest) {
            conflictProject = backendProject;
            setSaveState("后台项目读取期间检测到新的本地输入，已保护当前内容不被覆盖", "#ffb36b");
            render(); return;
          }
          const localRevision = Number(project.revision || 0);
          const remoteRevision = Number(backendProject.revision || 0);
          if (localRevision > remoteRevision) {
            conflictProject = backendProject;
            setSaveState(`发现本地恢复副本 r${localRevision} 与后台 r${remoteRevision} 不同，请确认后台版本`, "#ffb36b");
            render(); return;
          }
          project = backendProject;
          selected = String(project.active_shot_id || project.shots[0]?.id || "");
          selectedShotIds.clear(); (project.selected_shot_ids || []).forEach((id) => selectedShotIds.add(id));
          save("backend_restore", false);
          setSaveState(`已从后台恢复 · r${project.revision || 0}`, "#63e3a2");
          render();
        } catch (error) {
          setSaveState(`后台恢复失败：${error?.message || error}`, "#ff8f8f");
        }
      };
      const aspectCss = (value) => {
        const match = String(value || "16:9").match(/^\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)\s*$/);
        return match && Number(match[1]) > 0 && Number(match[2]) > 0 ? `${Number(match[1])} / ${Number(match[2])}` : "16 / 9";
      };
      const configuredAspect = () => aspectCss(this.widgets?.find((item) => item.name === "aspect_ratio")?.value || hiddenDefaults.aspect_ratio);
      const followConfiguredAspect = (element) => {
        element.dataset.ref2vaConfiguredAspect = "true";
        element.style.aspectRatio = configuredAspect();
        return element;
      };
      const followMediaAspect = (media, target = media) => {
        const apply = () => {
          const width = Number(media.videoWidth || media.naturalWidth || 0);
          const height = Number(media.videoHeight || media.naturalHeight || 0);
          if (width > 0 && height > 0) {
            target.dataset.ref2vaConfiguredAspect = "false";
            target.style.aspectRatio = `${width} / ${height}`;
            requestAnimationFrame(fitToContent);
          }
        };
        media.addEventListener(media.tagName === "IMG" ? "load" : "loadedmetadata", apply);
        if (media.complete || media.readyState >= 1) apply();
      };
      const updateConfiguredAspects = () => {
        const primaryAspect = this.widgets?.find((item) => item.name === "aspect_ratio")?.value || "16:9";
        const legacySecondAspect = this.widgets?.find((item) => item.name === "second_aspect_ratio");
        if (legacySecondAspect) { legacySecondAspect.value = primaryAspect; legacySecondAspect.callback?.(primaryAspect); }
        root.querySelectorAll('[data-ref2va-configured-aspect="true"]').forEach((element) => { element.style.aspectRatio = configuredAspect(); });
        requestAnimationFrame(fitToContent);
      };
      const shot = () => project.shots.find((s) => s.id === selected) || project.shots[0];
      const el = (tag, css = "") => { const x = document.createElement(tag); x.style.cssText = css; return x; };
      const button = (text, action, color = "#187d58") => { const b = el("button", `background:${color};border:1px solid #4dbf95;color:#fff;border-radius:6px;padding:7px 10px;cursor:pointer;font-weight:600;`); b.textContent = text; b.onclick = action; return b; };
      const updateMergeProgress = () => {
        if (mergeButton) {
          mergeButton.disabled = mergeRunning;
          mergeButton.textContent = mergeRunning ? "正在合并…" : "合并所选";
          mergeButton.style.cursor = mergeRunning ? "wait" : "pointer";
          mergeButton.style.opacity = mergeRunning ? ".65" : "1";
        }
        if (mergeProgressPanel) mergeProgressPanel.style.display = (mergeRunning || mergeMessage) ? "flex" : "none";
        if (mergeProgressMover) {
          mergeProgressMover.style.animation = mergeRunning ? "ref2va-merge-progress 1.25s ease-in-out infinite" : "none";
          mergeProgressMover.style.transform = mergeRunning ? "" : "none";
          mergeProgressMover.style.width = mergeRunning ? "34%" : "100%";
          mergeProgressMover.style.background = mergeRunning ? "#2cb9ff" : mergeMessageColor;
        }
        if (mergeProgressText) {
          mergeProgressText.textContent = mergeRunning ? "正在后台合并并修复接缝，请保持页面打开" : mergeMessage;
          mergeProgressText.style.color = mergeRunning ? "#dceeff" : mergeMessageColor;
        }
        if (mergeElapsedText) {
          if (mergeRunning) {
            const elapsed = Math.max(0, Math.floor((Date.now() - mergeStartedAt) / 1000));
            const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const seconds = String(elapsed % 60).padStart(2, "0");
            mergeElapsedText.textContent = `已用时 ${minutes}:${seconds}`;
          } else mergeElapsedText.textContent = "";
        }
      };
      const formatBytes = (bytes) => {
        const value = Math.max(0, Number(bytes || 0));
        if (value < 1024) return `${value} B`;
        if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
        if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(2)} MB`;
        return `${(value / 1024 ** 3).toFixed(2)} GB`;
      };
      const render = () => {
        root.replaceChildren();
        const statusBar = el("div", "display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;padding:7px 10px;border:1px solid #31516e;border-radius:7px;background:#091522;font-size:11px;");
        versionStatusElement = el("span", "color:#ffb36b;font-weight:700;");
        saveStatusElement = el("span", `color:${saveStateColor};margin-left:auto;`);
        const historyButton = button("项目历史", async () => {
          try {
            const response = await api.fetchApi(`/ref2va-director/project/${encodeURIComponent(project.project_id)}/history`);
            const body = await response.json();
            if (!response.ok || !body?.ok) throw new Error(body?.error || `HTTP ${response.status}`);
            const rows = (body.history || []).slice(0, 20).map((item) => `r${item.revision} · ${item.updated_at || "未知时间"} · ${item.reason || "自动保存"}`);
            window.alert(rows.length ? `最近项目恢复点：\n\n${rows.join("\n")}` : "当前项目还没有历史恢复点。首次保存后会自动创建。 ");
          } catch (error) { window.alert(`读取项目历史失败：${error?.message || error}`); }
        }, "#31516e");
        historyButton.style.padding = "4px 8px";
        const storageButton = button("项目存储", async () => {
          try {
            const response = await api.fetchApi(`/ref2va-director/project/${encodeURIComponent(project.project_id)}/storage`);
            const body = await response.json();
            if (!response.ok || !body?.ok) throw new Error(body?.error || `HTTP ${response.status}`);
            const storage = body.storage || {};
            if (!Number(storage.trash_file_count || 0)) {
              window.alert(`当前项目的视频回收区为空。\n\n项目：${project.project_id}\n占用：0 B`);
              return;
            }
            const confirmed = window.confirm(`确定永久清空当前项目的视频回收区吗？\n\n项目：${project.project_id}\n视频版本：${Number(storage.trash_take_count || 0)} 个\n文件：${Number(storage.trash_file_count || 0)} 个\n占用：${formatBytes(storage.trash_bytes)}\n\n只会清理当前项目的 .trash/takes。\n不会删除提示词、项目历史、当前镜头视频、输入素材、合并视频或其他项目。\n\n清空后无法恢复。`);
            if (!confirmed) return;
            storageButton.disabled = true; storageButton.textContent = "正在清空…";
            const purgeResponse = await api.fetchApi("/ref2va-director/project/purge-video-trash", {
              method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ project_id: project.project_id }),
            });
            const purgeBody = await purgeResponse.json();
            if (!purgeResponse.ok || !purgeBody?.ok) throw new Error(purgeBody?.error || `HTTP ${purgeResponse.status}`);
            const result = purgeBody.result || {};
            window.alert(`视频回收区已永久清空。\n\n删除版本：${Number(result.removed_take_count || 0)} 个\n删除文件：${Number(result.removed_file_count || 0)} 个\n释放空间：${formatBytes(result.removed_bytes)}`);
          } catch (error) {
            window.alert(`项目存储操作失败：${error?.message || error}\n\n如果当前后台尚未加载该接口，请确认生成队列空闲后重启 ComfyUI。`);
          } finally {
            storageButton.disabled = false; storageButton.textContent = "项目存储";
          }
        }, "#31516e");
        storageButton.style.padding = "4px 8px";
        const feedbackBox=button("BUG反馈群 · 1106686971",()=>{const field=document.createElement("textarea");field.value="1106686971";field.style.cssText="position:fixed;left:-9999px;top:-9999px";document.body.append(field);field.select();let copied=false;try{copied=document.execCommand("copy");}catch(_){}field.remove();if(!copied&&navigator.clipboard?.writeText)navigator.clipboard.writeText("1106686971").catch(()=>{});feedbackBox.textContent=copied?"已复制群号 · 1106686971":"已请求复制 · 1106686971";window.setTimeout(()=>{if(feedbackBox.isConnected)feedbackBox.textContent="BUG反馈群 · 1106686971";},1800);},"#31516e");
        feedbackBox.style.cssText+="padding:5px 10px;border:1px solid #72c5ff;border-radius:5px;color:#dceeff;font-weight:700;";feedbackBox.title="点击复制 BUG 反馈群号";
        const shareButton=button("导出分享包",()=>{const blob=new Blob([JSON.stringify({product:"梦镜 DreamShot",version:FRONTEND_VERSION,project},null,2)],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=`梦镜 DreamShot+${FRONTEND_VERSION}.json`;document.body.append(link);link.click();link.remove();window.setTimeout(()=>URL.revokeObjectURL(link.href),1000);},"#31516e");shareButton.style.padding="4px 8px";
        const reloadConflict = button("重新加载后台版本", () => {
          if (!conflictProject) return;
          project = conflictProject; conflictProject = null;
          selected = String(project.active_shot_id || project.shots?.[0]?.id || "");
          selectedShotIds.clear(); (project.selected_shot_ids || []).forEach((id) => selectedShotIds.add(id));
          setSaveState(`已加载后台版本 · r${project.revision || 0}`, "#63e3a2");
          save("conflict_reload"); render();
        }, "#7b3650");
        reloadConflict.style.display = conflictProject ? "inline-block" : "none";
        statusBar.append(versionStatusElement, historyButton, storageButton, shareButton, feedbackBox, saveStatusElement, reloadConflict);
        root.append(statusBar); updateStatusBar();
        const head = el("div", "display:flex;flex-wrap:wrap;gap:8px;align-items:center;flex-shrink:0;");
        const brandTitle=el("div","width:100%;font-size:22px;font-weight:900;color:#80c8ff;letter-spacing:.5px;");brandTitle.textContent="梦镜 DreamShot";head.append(brandTitle);
        const refreshProject = async () => {
          const id = String(project.project_id || ""); if (!id) return;
          try {
            const res = await api.fetchApi(`/ref2va-director/project/${encodeURIComponent(id)}`);
            const body = await res.json();
            if (res.ok && body?.ok && body.project?.shots) {
              const incomingById = new Map(body.project.shots.map((item) => [item.id, item]));
              for (const localShot of project.shots) {
                const incomingShot = incomingById.get(localShot.id);
                if (!incomingShot) continue;
                localShot.takes = incomingShot.takes || [];
                localShot.selected_take_id = incomingShot.selected_take_id || localShot.selected_take_id || null;
                const staleActiveStatus = (completedLiveShotIds.has(String(localShot.id)) || (!liveRunning && !project.active_job_id)) && ["queued", "preparing_models", "sampling", "decoding", "upscaling", "saving"].includes(String(incomingShot.status || ""));
                localShot.status = staleActiveStatus && localShot.takes?.length ? "generated" : (incomingShot.status || localShot.status);
              }
              save(); render();
            }
          } catch (_) { /* A new unsaved project has no archive yet. */ }
        };
        const queue = (scope) => {
          const seedModeWidget = this.widgets?.find((item) => item.name === "seed_mode");
          if (String(seedModeWidget?.value || "随机") === "随机") changeRandomSeed();
          normalizeHiddenWidgets();
          save();
          if (scope === "仅当前镜头") liveShotId = selected;
          else if (scope === "选中镜头") liveShotId = project.shots.find((item) => selectedShotIds.has(item.id) && !item.selected_take_id)?.id || selected;
          else liveShotId = project.shots.find((item) => item.enabled !== false)?.id || selected;
          if (liveShotId) completedLiveShotIds.delete(String(liveShotId));
          const scopeWidget = this.widgets?.find((w) => w.name === "run_scope");
          if (scopeWidget) { scopeWidget.value = scope; scopeWidget.callback?.(scope); }
          if (typeof app.queuePrompt !== "function") { window.alert("ComfyUI 当前前端没有提供队列接口；请使用右上角“运行”。项目已保存为对应运行范围。"); return; }
          const queued = app.queuePrompt(0, 1);
          queued?.catch?.((error) => window.alert(`无法加入运行队列：${error?.message || error}`));
        };
        const stopGeneration = async () => {
          if (liveStatusText) liveStatusText.textContent = "正在终止…";
          try { if (typeof api.interrupt === "function") await api.interrupt(); else await api.fetchApi("/interrupt", { method: "POST" }); }
          catch (error) { window.alert(`终止生成失败：${error?.message || error}`); }
        };
        const changeRandomSeed = (feedbackButton = null) => {
          const seedWidget = this.widgets?.find((item) => item.name === "noise_seed");
          if (!seedWidget) { setSaveState("未找到随机种子参数", "#ff8585"); return; }
          const nextSeed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
          seedWidget.value = nextSeed;
          seedWidget.callback?.(seedWidget.value);
          this.graph?.setDirtyCanvas?.(true, true);
          setSaveState(`已更换随机种子 · ${nextSeed}`, "#63e3a2");
          if (feedbackButton) {
            feedbackButton.disabled = true;
            feedbackButton.textContent = "已更换 ✓";
            window.setTimeout(() => { if (feedbackButton.isConnected) { feedbackButton.disabled = false; feedbackButton.textContent = "换随机种子"; } }, 1200);
          }
        };
        const mergeSelected = async (download = false) => {
          if (mergeRunning) return;
          if (!selectedShotIds.size) { window.alert("请先勾选至少一个已生成镜头。"); return; }
          const generatedIds = project.shots.filter((shot) => selectedShotIds.has(shot.id) && shot.takes?.length).map((shot) => shot.id);
          if (!generatedIds.length) { window.alert("所选镜头尚未生成视频，无法合并。"); return; }
          mergeRunning = true;
          mergeStartedAt = Date.now();
          mergeMessage = "";
          clearInterval(mergeTimer);
          mergeTimer = window.setInterval(updateMergeProgress, 500);
          updateMergeProgress();
          try {
            const repairMotion = false;
            setSaveState(repairMotion ? "正在合并并进行 48 FPS 运动补帧，请稍候…" : "正在合并视频…", "#ffcf70");
            const selections=generatedIds.map(shotId=>{const shot=project.shots.find(item=>item.id===shotId);const fallbackTake=shot?.takes?.find(item=>item.take_id===shot.selected_take_id)||shot?.takes?.at?.(-1);const takeId=shot?.merge_take_id||fallbackTake?.take_id;const selectedTake=shot?.takes?.find(item=>item.take_id===takeId)||fallbackTake;const source=shot?.merge_source==="final"&&selectedTake?.files?.final?"final":"initial";return{shot_id:shotId,take_id:takeId,source};});
            const response = await api.fetchApi("/ref2va-director/merge-selected", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ project_id: project.project_id, shot_ids: generatedIds, selections, source: "final_if_available", motion_interpolation: repairMotion }) });
            const body = await response.json();
            if (!response.ok || !body?.ok) throw new Error(body?.error || "合并失败");
            mergedPreview = { filename: body.filename, subfolder: body.subfolder || "", type: body.type || "output", cacheKey: Date.now() };
            mergedBaselinePreview = body.baseline_filename ? { filename: body.baseline_filename, subfolder: body.subfolder || "", type: body.type || "output", cacheKey: Date.now() } : null;
            mergedSourceManifest = Array.isArray(body.source_manifest) ? body.source_manifest : [];
            continuityReport = body.continuity_report || null;
            project.settings = project.settings || {};
            project.settings.video_library = Array.isArray(project.settings.video_library) ? project.settings.video_library : [];
            project.settings.video_library.push({ id:`delivery-${Date.now()}`, kind:"delivery", created_at:new Date().toISOString(), asset:{...mergedPreview}, baseline:mergedBaselinePreview?{...mergedBaselinePreview}:null, source_manifest:mergedSourceManifest });
            save();
            mergeMessage = body.motion_interpolation ? "合并完成 · 48 FPS 接缝修复视频已生成" : "合并完成 · 24 FPS 视频已生成";
            mergeMessageColor = "#63e3a2";
            setSaveState(body.motion_interpolation ? "合并完成 · 已进行 48 FPS 运动补帧" : "合并完成 · 24 FPS", "#63e3a2");
            if (download) {
              const link = document.createElement("a"); link.href = storedViewUrl(mergedPreview); link.download = `${project.name || project.project_id}-选中镜头.mp4`; document.body.append(link); link.click(); link.remove();
            }
          } catch (error) {
            mergeMessage = `合并失败：${error?.message || error}`;
            mergeMessageColor = "#ff8585";
            setSaveState("合并失败", "#ff8585");
            window.alert(`选中镜头处理失败：${error?.message || error}`);
          } finally {
            mergeRunning = false;
            clearInterval(mergeTimer);
            mergeTimer = 0;
            render();
          }
        };
        const deleteMergedDelivery = async () => {
          if (!mergedPreview?.filename) return;
          const filename = String(mergedPreview.filename);
          const confirmed = window.confirm(`确定删除本次合并结果吗？\n\n将移入项目回收区：\n• 24 FPS 原始拼接\n• 48 FPS 接缝修复\n• 连续性验收报告\n• 全部接缝帧条\n\n不会删除镜头 1/2/3，也不会删除它们的原始或最终版本。\n\n合并标识：${filename}`);
          if (!confirmed) return;
          try {
            setSaveState("正在删除本次合并结果…", "#ffcf70");
            const response = await api.fetchApi("/ref2va-director/delete-delivery", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ project_id: project.project_id, merged_filename: filename }) });
            const body = await response.json();
            if (!response.ok || !body?.ok) throw new Error(body?.error || "删除失败");
            mergedPreview = null; mergedBaselinePreview = null; mergedSourceManifest = []; continuityReport = null;
            mergeMessage = "本次合并结果已移入项目回收区"; mergeMessageColor = "#63e3a2";
            setSaveState(body.message || mergeMessage, "#63e3a2");
            render();
          } catch (error) {
            setSaveState(`删除合并结果失败：${error?.message || error}`, "#ff8585");
            window.alert(`删除合并结果失败：${error?.message || error}`);
          }
        };
        const selectableShotIds = project.shots.map((shot) => shot.id);
        const allShotsSelected = selectableShotIds.length > 0 && selectableShotIds.every((id) => selectedShotIds.has(id));
        const toggleSelectAll = () => {
          if (!selectableShotIds.length) return;
          if (allShotsSelected) selectableShotIds.forEach((id) => selectedShotIds.delete(id));
          else selectableShotIds.forEach((id) => selectedShotIds.add(id));
          save(); render();
        };
        const startSelected = () => {
          if (!selectedShotIds.size) { window.alert("请先勾选至少一个镜头。"); return; }
          const existing = project.shots.filter((shot) => selectedShotIds.has(shot.id) && shot.takes?.length && shot.selected_take_id);
          if (existing.length && existing.length < selectedShotIds.size) setSaveState(`批量生成将跳过已有版本 · 新生成 ${selectedShotIds.size - existing.length} 个镜头`, "#ffcf70");
          queue("选中镜头");
        };
        const exportSelectedShots = () => {
          const shots = project.shots.filter((shot) => selectedShotIds.has(shot.id));
          const links = [];
          for (const shot of shots) {
            const take = shot.takes?.find?.((item) => item.take_id === shot.selected_take_id) || shot.takes?.at?.(-1);
            const file = take?.files?.final || take?.files?.initial;
            if (take && file) links.push({ url: outputViewUrl(project.project_id, shot.id, take.take_id, file), name: `${shot.name || shot.id}-${take.take_id}.mp4` });
          }
          if (!links.length) { window.alert("所选镜头没有可导出的视频版本。"); return; }
          links.forEach((item, index) => setTimeout(() => { const link = document.createElement("a"); link.href = item.url; link.download = item.name; document.body.append(link); link.click(); link.remove(); }, index * 250));
        };
        const deleteSelectedShots = () => {
          const targets = project.shots.filter((shot) => selectedShotIds.has(shot.id));
          if (!targets.length) { window.alert("请先勾选要删除的镜头。"); return; }
          if (targets.length >= project.shots.length) { window.alert("至少保留一个镜头，不能删除全部镜头。"); return; }
          const names = targets.map((shot) => shot.name || "未命名镜头").join("、");
          if (!window.confirm(`确定从项目中删除所选镜头吗？\n\n将移除：${names}\n\n本操作只移除项目中的镜头记录和选择状态，已生成的镜头文件暂不删除；不影响未选镜头、全局资源和合并视频。`)) return;
          targets.forEach((shot) => { const index = project.shots.indexOf(shot); if (index >= 0) project.shots.splice(index, 1); selectedShotIds.delete(shot.id); });
          renumberShots();
          if (!project.shots.some((shot) => shot.id === selected)) selected = project.shots[0].id;
          project.active_shot_id = selected; save(); render();
        };
        mergeButton = button("合并所选", () => mergeSelected(false), "#32658a");
        head.append(button(allShotsSelected ? "取消全选" : "全选", toggleSelectAll, "#32658a"), button("▶ 生成当前镜头", () => queue("仅当前镜头"), "#167bb6"), button(`▶ 生成所选${selectedShotIds.size ? `（${selectedShotIds.size}）` : ""}`, startSelected, "#187d58"), button("＋ 添加镜头", () => { const n = project.shots.length + 1; const mode = defaultFollowingMode(project.shots[0]?.mode); project.shots.push({ ...fallback().shots[0], id: makeId(), order: n, name: `镜头 ${n}`, mode, mode_inherited: true, continuity: { enabled: n > 1, strategy: n > 1 ? "auto_seamless" : "off", context_frames: 22, source: "initial" }, takes: [] }); renumberShots(); selected = project.shots.at(-1).id; save(); render(); }), mergeButton, button("导出所选镜头", exportSelectedShots, "#32658a"), button("删除所选镜头", deleteSelectedShots, "#7b3650"));
        const selectedNames = project.shots.filter((shot) => selectedShotIds.has(shot.id)).map((shot) => shot.name || "未命名镜头");
        const selectionSummary = el("div", "box-sizing:border-box;width:100%;min-height:38px;display:flex;align-items:center;gap:10px;margin:5px 0 9px;padding:7px 14px;border:1px solid #37698d;border-radius:7px;background:#0b1c2d;color:#9fc8e7;font-size:13px;overflow:hidden;box-shadow:inset 0 0 0 1px #102b41;");
        selectionSummary.innerHTML = selectedNames.length ? `<span style=\"color:#63e3a2;font-weight:700\">已选 ${selectedNames.length} 个镜头</span><span style=\"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1\" title=\"${selectedNames.join("、")}\">${selectedNames.join("、")}</span>` : `<span>未选择镜头</span><span style=\"opacity:.75\">点击卡片编辑，点击右上角圆标加入批量生成与交付</span>`;
        if (selectedNames.length) { const clearSelection = button("清除选择", () => { selectedShotIds.clear(); save(); render(); }, "#31516e"); clearSelection.style.cssText += "margin-left:auto;flex:0 0 auto;white-space:nowrap;padding:6px 14px;"; selectionSummary.append(clearSelection); }
        root.append(head);
        root.append(selectionSummary);
        mergeProgressPanel = el("div", "box-sizing:border-box;width:100%;display:none;flex-direction:column;gap:7px;padding:9px 12px;border:1px solid #37698d;border-radius:7px;background:#091522;");
        const mergeProgressHeader = el("div", "display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:12px;");
        mergeProgressText = el("span", "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#dceeff;");
        mergeElapsedText = el("span", "flex:0 0 auto;color:#9fc8e7;font-variant-numeric:tabular-nums;");
        mergeProgressHeader.append(mergeProgressText, mergeElapsedText);
        const mergeProgressTrack = el("div", "height:6px;overflow:hidden;border-radius:3px;background:#14283a;");
        mergeProgressMover = el("div", "width:34%;height:100%;border-radius:3px;background:#2cb9ff;animation:ref2va-merge-progress 1.25s ease-in-out infinite;");
        mergeProgressTrack.append(mergeProgressMover); mergeProgressPanel.append(mergeProgressHeader, mergeProgressTrack); root.append(mergeProgressPanel);
        updateMergeProgress();
        const rail = el("div", "display:flex;gap:10px;overflow-x:auto;padding:2px 0 7px;flex-shrink:0;align-items:stretch;");
        const activeJob = (project.jobs || []).find((item) => item.job_id === project.active_job_id);
        if (activeJob?.current_shot_id) liveShotId = String(activeJob.current_shot_id);
        project.shots.forEach((s, i) => {
          const active = s.id === selected;
          const multiSelected = selectedShotIds.has(s.id);
          const card = el("div", `box-sizing:border-box;position:relative;display:flex;flex-direction:column;gap:7px;flex:0 0 238px;min-width:238px;text-align:left;background:${multiSelected && active ? "linear-gradient(145deg,#245783,#1d6758)" : multiSelected ? "#1d6758" : active ? "#245783" : "#13263a"};border:2px solid ${multiSelected ? "#63e3a2" : active ? "#72c5ff" : "#31516e"};border-radius:9px;color:#dceeff;padding:10px;cursor:pointer;box-shadow:${active ? "0 0 0 1px #72c5ff55,0 6px 16px #0004" : "none"};`);
          card.onclick = () => { selected = s.id; save(); render(); };
          const take = s.takes?.find?.((x) => x.take_id === s.selected_take_id) || s.takes?.at?.(-1);
          const taskState = STATUS_LABELS[String(s.status || "draft")];
          const state = taskState && !["draft", "generated", "cached"].includes(String(s.status || "draft")) ? taskState : (s.takes?.length ? `已有 ${s.takes.length} 个版本` : (taskState || "待生成"));
          const videoFile = take?.files?.final || take?.files?.initial;
          const cardHeader = el("div", "padding-right:27px;line-height:1.35;");
          cardHeader.innerHTML = `<b style="font-size:17px;color:#8fd0ff">${s.name || `镜头 ${i + 1}`}　${Number(s.duration_seconds || 5).toFixed(1)}秒</b><br><span style="color:#b8d9ef;font-size:13px">${modeLabel(s.mode)}</span><br><span style="color:${s.takes?.length ? "#63e3a2" : "#9ab0c0"};font-size:12px">${state}</span>`;
          card.append(cardHeader);
          if(s.takes?.some(item=>item.files?.initial||item.files?.final)){
            const mergeSource=document.createElement("select");mergeSource.setAttribute("aria-label",`${s.name||`镜头 ${i+1}`} 合并使用`);mergeSource.title="为本镜头单独选择参与合并的视频版本";mergeSource.style.cssText="width:100%;padding:7px;border:1px solid #63e3a2;border-radius:5px;background:#102f49;color:#fff;font-size:11px;font-weight:700;cursor:pointer;";
            const currentMergeTake=s.merge_take_id||s.selected_take_id||take?.take_id;const currentMergeRecord=s.takes.find(item=>item.take_id===currentMergeTake)||take;const currentMergeSource=s.merge_source==="initial"?"initial":(currentMergeRecord?.files?.final?"final":"initial");for(const candidate of s.takes||[]){for(const source of ["initial","final"]){if(!candidate.files?.[source])continue;const option=document.createElement("option");option.value=`${candidate.take_id}::${source}`;option.textContent=`本次合并：${profileLabel(candidate,source)} · ${candidate.take_id.slice(-6)}`;option.selected=candidate.take_id===currentMergeTake&&source===currentMergeSource;mergeSource.append(option);}}
            mergeSource.onclick=event=>event.stopPropagation();mergeSource.onchange=()=>{const [takeId,source]=mergeSource.value.split("::");s.merge_take_id=takeId;s.merge_source=source;save();render();};card.append(mergeSource);
          }
          const ownsLivePreview = String(s.id) === String(liveShotId) && (liveRunning || livePreviewUrl);
          if (ownsLivePreview) {
            const liveViewport = followConfiguredAspect(el("div", "display:flex;align-items:center;justify-content:center;position:relative;width:100%;background:#03070b;border:1px dashed #2cb9ff;border-radius:7px;overflow:hidden;"));
            livePreviewImage = document.createElement("img"); livePreviewImage.alt = `${s.name || `镜头 ${i + 1}`} 实时生成预览`; livePreviewImage.style.cssText = "display:none;width:100%;height:100%;object-fit:contain;"; followMediaAspect(livePreviewImage, liveViewport);
            const liveEmpty = el("span", "font-size:12px;color:#7894aa;text-align:center;padding:12px;"); liveEmpty.textContent = "等待实时采样画面";
            if (livePreviewUrl) { livePreviewImage.src = livePreviewUrl; livePreviewImage.style.display = "block"; liveEmpty.style.display = "none"; }
            liveViewport.append(livePreviewImage, liveEmpty); card.append(liveViewport);
            const progressTrack = el("div", "height:5px;background:#14283a;border-radius:999px;overflow:hidden;");
            liveProgressFill = el("div", `height:100%;width:${liveProgressMax ? Math.min(100, Math.round(liveProgressValue / liveProgressMax * 100)) : 0}%;background:#2cb9ff;transition:width .15s;`); progressTrack.append(liveProgressFill); card.append(progressTrack);
            liveStatusText = el("div", `font-size:11px;color:${liveRunning ? "#63e3a2" : "#7894aa"};`); liveStatusText.textContent = liveRunning ? (liveProgressMax ? `正在生成 ${liveProgressValue}/${liveProgressMax}` : "正在生成") : "等待运行"; card.append(liveStatusText);
          } else if (videoFile) {
            const thumb = document.createElement("video");
            thumb.src = outputViewUrl(project.project_id, s.id, take.take_id, videoFile);
            thumb.muted = true; thumb.loop = true; thumb.autoplay = true; thumb.playsInline = true; thumb.preload = "metadata";
            thumb.style.cssText = "display:block;width:100%;object-fit:contain;border:1px solid #46779a;border-radius:7px;background:#05090e;box-shadow:inset 0 0 0 1px #0008;";
            followConfiguredAspect(thumb); followMediaAspect(thumb);
            card.append(thumb);
          } else {
            const emptyPreview = followConfiguredAspect(el("div", "display:flex;align-items:center;justify-content:center;width:100%;border:1px dashed #41647f;border-radius:7px;background:#07121d;color:#7894aa;font-size:13px;"));
            emptyPreview.textContent = "生成后在这里显示视频画面";
            card.append(emptyPreview);
          }
          if (active || ownsLivePreview) {
            const cardActions = el("div", "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px;margin-top:auto;");
            const refreshShot = button("刷新成果", (event) => { event.stopPropagation(); refreshProject(); }, "#275e90");
            liveStopButton = button("停止生成", (event) => { event.stopPropagation(); stopGeneration(); }, "#8a3848");
            liveStopButton.disabled = !liveRunning; liveStopButton.style.opacity = liveRunning ? "1" : ".45";
            const randomSeed = button("换随机种子", (event) => { event.stopPropagation(); changeRandomSeed(event.currentTarget); }, "#59658f");
            const rerunShot = button("重跑当前镜头", (event) => { event.stopPropagation(); selected = s.id; queue("仅当前镜头"); }, "#275e90");
            for (const item of [refreshShot, liveStopButton, randomSeed, rerunShot]) { item.style.cssText += "padding:6px 4px;font-size:10px;min-width:0;"; item.onclick = ((action) => (event) => { event.stopPropagation(); action(event); })(item.onclick); }
            cardActions.append(refreshShot, liveStopButton, randomSeed, rerunShot); card.append(cardActions);
          }
          const selectCheck = document.createElement("input");
          selectCheck.type = "checkbox"; selectCheck.checked = multiSelected; selectCheck.title = "加入所选镜头运行 / 合并 / 导出";
          selectCheck.setAttribute("aria-label", `选择 ${s.name || `镜头 ${i + 1}`} 用于生成和交付`);
          selectCheck.style.cssText = "position:absolute;right:7px;top:7px;width:20px;height:20px;accent-color:#63e3a2;cursor:pointer;filter:drop-shadow(0 1px 2px #0008);";
          selectCheck.onclick = (event) => event.stopPropagation();
          selectCheck.onchange = () => { if (selectCheck.checked) selectedShotIds.add(s.id); else selectedShotIds.delete(s.id); save(); render(); };
          card.append(selectCheck);
          rail.append(card);
        });
        root.append(rail);
        project.global_assets = project.global_assets || { images: [], videos: [], audios: [] };
        for (const key of ["images", "videos", "audios"]) {
          if (!Array.isArray(project.global_assets[key])) project.global_assets[key] = project.global_assets[key] ? [project.global_assets[key]] : [];
        }
        const resourceBucket = (file) => {
          const mime = String(file?.type || "").toLowerCase();
          const name = String(file?.name || "").toLowerCase();
          if (mime.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp|tiff?)$/.test(name)) return "images";
          if (mime.startsWith("video/") || /\.(mp4|webm|mov|mkv|avi|m4v)$/.test(name)) return "videos";
          if (mime.startsWith("audio/") || /\.(mp3|wav|flac|m4a|aac|ogg|opus)$/.test(name)) return "audios";
          return null;
        };
        const uploadStoredFiles = async (files) => {
          const uploaded = [];
          for (const file of files) {
            const bucket = resourceBucket(file);
            if (!bucket) continue;
            // ComfyUI's input upload route accepts arbitrary media when the
            // multipart field is named "image".  The previous global uploader
            // used "file" for video/audio, which made those drops fail.
            const form = new FormData(); form.append("image", file, file.name); form.append("overwrite", "false");
            const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
            let body = null;
            try { body = await response.json(); } catch (_) { /* handled below */ }
            if (!response.ok || !body?.name) throw new Error(body?.error || `${file.name} 上传失败`);
            uploaded.push({ bucket, asset: { filename: body.name, subfolder: body.subfolder || "", type: body.type || "input" } });
          }
          return uploaded;
        };
        const globalDrawer = document.createElement("details"); globalDrawer.open = globalDrawerOpen; globalDrawer.style.cssText = "background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:0 10px;flex-shrink:0;";
        const globalSummary = markDrawerSummary(document.createElement("summary")); globalSummary.style.cssText = "cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;font-size:18px;"; globalSummary.textContent = "全局提示词 / 约束 / 资源（应用到所有镜头）";
        const globalBody = el("div", "display:flex;flex-direction:column;gap:9px;padding:0 0 10px;");
        const globalPrompt = el("textarea", "box-sizing:border-box;width:100%;min-height:58px;resize:vertical;background:#091523;color:#e8f4ff;border:1px solid #37658d;border-radius:6px;padding:8px;"); globalPrompt.placeholder = "全局提示词：所有镜头都会继承"; globalPrompt.value = project.global_prompt || ""; globalPrompt.oninput = () => { project.global_prompt = globalPrompt.value; save(); };
        const globalConstraint = el("textarea", "box-sizing:border-box;width:100%;min-height:58px;resize:vertical;background:#091523;color:#e8f4ff;border:1px solid #37658d;border-radius:6px;padding:8px;"); globalConstraint.placeholder = "全局约束提示词：所有镜头必须遵守"; globalConstraint.value = project.global_constraint_prompt || ""; globalConstraint.oninput = () => { project.global_constraint_prompt = globalConstraint.value; save(); };
        const globalStatus = el("div", "min-height:18px;font-size:12px;color:#9dc3df;");
        const addGlobalFiles = async (files) => {
          const supported = files.filter((file) => resourceBucket(file));
          if (!supported.length) { globalStatus.textContent = "未发现支持的图片、视频或音频文件。"; globalStatus.style.color = "#ffb3b3"; return; }
          globalStatus.textContent = `正在上传 ${supported.length} 个资源…`; globalStatus.style.color = "#80c8ff";
          try {
            const uploaded = await uploadStoredFiles(supported);
            for (const { bucket, asset } of uploaded) project.global_assets[bucket].push(asset);
            globalDrawerOpen = true;
            save(); render();
          } catch (error) {
            globalStatus.textContent = `上传失败：${error?.message || error}`; globalStatus.style.color = "#ffb3b3";
          }
        };
        const globalUpload = button("选择并上传全局资源", () => { const input = document.createElement("input"); input.type = "file"; input.multiple = true; input.accept = "image/*,video/*,audio/*"; input.onchange = () => addGlobalFiles([...(input.files || [])]); input.click(); }, "#48698a");
        const clearGlobal = button("清空全部内容", () => {
          if (!window.confirm("确定清空全部全局提示词、约束和全局素材吗？\n\n不会删除镜头、已有视频或单镜头素材。")) return;
          project.global_prompt = ""; project.global_constraint_prompt = "";
          project.global_assets = { images: [], videos: [], audios: [] };
          save(); render();
        }, "#7b3650");
        const globalDrop = el("div", "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;min-height:82px;border:1px dashed #4a789e;border-radius:7px;color:#b8d5e9;background:#0a1724;font-size:13px;text-align:center;padding:12px;cursor:copy;transition:border-color .15s,background .15s;");
        const dropTitle = el("div", "font-weight:700;color:#80c8ff;"); dropTitle.textContent = "把图片、视频或音频直接拖到这里";
        const dropHint = el("div", "font-size:11px;color:#789db8;"); dropHint.textContent = "支持多选和混合拖入，上传后会显示可预览资源卡片";
        globalDrop.append(dropTitle, dropHint);
        const setGlobalDropActive = (active) => { globalDrop.style.borderColor = active ? "#72c5ff" : "#4a789e"; globalDrop.style.background = active ? "#123a58" : "#0a1724"; dropTitle.textContent = active ? "松开即可添加到所有镜头" : "把图片、视频或音频直接拖到这里"; };
        globalDrop.ondragenter = (event) => { event.preventDefault(); setGlobalDropActive(true); };
        globalDrop.ondragover = (event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; setGlobalDropActive(true); };
        globalDrop.ondragleave = (event) => { if (!globalDrop.contains(event.relatedTarget)) setGlobalDropActive(false); };
        globalDrop.ondrop = (event) => { event.preventDefault(); setGlobalDropActive(false); addGlobalFiles([...(event.dataTransfer?.files || [])]); };
        const globalCounts = el("div", "font-size:12px;color:#9dc3df;"); globalCounts.textContent = `全局图 ${project.global_assets.images.length} · 视频 ${project.global_assets.videos.length} · 音频 ${project.global_assets.audios.length}`;
        const globalGallery = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;");
        const globalResourceCard = (bucket, asset, index) => {
          const mediaType = bucket === "images" ? "image" : bucket === "videos" ? "video" : "audio";
          const label = bucket === "images" ? `全局图片 ${index + 1}` : bucket === "videos" ? `全局视频 ${index + 1}` : `全局音频 ${index + 1}`;
          const card = el("div", "position:relative;display:flex;flex-direction:column;gap:6px;min-width:0;background:#0d1d2d;border:1px solid #31516e;border-radius:6px;padding:7px;");
          const top = el("div", "display:flex;gap:6px;align-items:center;min-width:0;");
          const kind = el("span", "flex:0 0 auto;color:#80c8ff;font-size:11px;font-weight:700;"); kind.textContent = label;
          const filename = el("span", "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#dceeff;font-size:11px;"); filename.textContent = asset?.filename || "未命名素材"; filename.title = asset?.filename || "";
          const remove = button("×", () => { project.global_assets[bucket].splice(index, 1); globalDrawerOpen = true; save(); render(); }, "#71364b"); remove.title = `移除${label}`; remove.setAttribute("aria-label", `移除${label} ${asset?.filename || ""}`); remove.style.cssText += "margin-left:auto;padding:1px 7px;font-size:16px;line-height:20px;";
          top.append(kind, filename, remove); card.append(top);
          if (mediaType === "image") {
            const preview = document.createElement("img"); preview.src = storedViewUrl(asset); preview.alt = `${label} ${asset?.filename || ""}`; preview.loading = "lazy"; preview.style.cssText = "width:100%;height:120px;object-fit:contain;background:#05090e;border-radius:4px;"; card.append(preview);
          } else if (mediaType === "video") {
            const preview = document.createElement("video"); preview.src = storedViewUrl(asset); preview.controls = true; preview.preload = "metadata"; preview.style.cssText = "width:100%;height:120px;object-fit:contain;background:#05090e;border-radius:4px;"; card.append(preview);
          } else {
            const preview = document.createElement("audio"); preview.src = storedViewUrl(asset); preview.controls = true; preview.preload = "metadata"; preview.style.cssText = "width:100%;height:38px;"; card.append(preview);
          }
          return card;
        };
        for (const bucket of ["images", "videos", "audios"]) project.global_assets[bucket].forEach((asset, index) => globalGallery.append(globalResourceCard(bucket, asset, index)));
        const globalActions = el("div", "display:flex;flex-wrap:wrap;gap:8px;");
        globalActions.append(globalUpload, clearGlobal);
        globalBody.append(globalPrompt, globalConstraint, globalActions, globalDrop, globalStatus, globalCounts);
        if (globalGallery.children.length) globalBody.append(globalGallery);
        else { const empty = el("div", "padding:12px;border:1px dashed #31516e;border-radius:6px;color:#7894aa;font-size:12px;text-align:center;"); empty.textContent = "尚未添加全局资源；拖入后将在这里显示缩略图和播放器。"; globalBody.append(empty); }
        globalDrawer.append(globalSummary, globalBody); globalDrawer.addEventListener("toggle", () => { globalDrawerOpen = globalDrawer.open; globalSummary.textContent = "全局提示词 / 约束 / 资源（应用到所有镜头）"; requestAnimationFrame(fitToContent); }); root.append(globalDrawer);
        const current = shot(); if (!current) return;
        const field = (label, control, minWidth = 100) => {
          const box = el("label", `display:flex;flex-direction:column;gap:4px;min-width:${minWidth}px;font-size:12px;color:#9dc3df;`);
          const caption = el("span", "font-weight:600;"); caption.textContent = label;
          box.append(caption, control); return box;
        };
        const controlCss = "box-sizing:border-box;width:100%;max-width:100%;background:#102943;color:#fff;border:1px solid #3a6790;border-radius:5px;padding:7px;min-width:0;overflow:hidden;text-overflow:ellipsis;";
        const toolbar = document.createElement("details"); toolbar.open = shotSettingsOpen; toolbar.style.cssText="display:flex;flex-direction:column;gap:3px;background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:0 10px;flex-shrink:0;";
        // Keep the active shot controls immediately above the global prompt
        // drawer so the primary editing context is visible first.
        root.insertBefore(toolbar, globalDrawer);
        const shotTitle = markDrawerSummary(document.createElement("summary")); shotTitle.style.cssText="cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;font-size:18px;line-height:1.2;"; shotTitle.textContent = `镜头 ${project.shots.indexOf(current) + 1} 设置`;
        const toolbarBody=el("div","display:flex;flex-direction:column;gap:8px;padding:0 0 10px;");
        const toolbarMain = el("div", "display:grid;grid-template-columns:minmax(135px,1fr) minmax(72px,.6fr) minmax(72px,.6fr) minmax(185px,1.2fr);gap:14px;align-items:end;min-width:0;");
        const mode = el("select", controlCss); MODES.forEach(([key, label]) => { const o = document.createElement("option"); o.value = key; o.textContent = label; o.selected = key === current.mode; mode.append(o); }); mode.onchange = () => {
          current.mode = mode.value;
          const currentIndex = project.shots.indexOf(current);
          if (currentIndex === 0) {
            const inheritedMode = defaultFollowingMode(current.mode);
            project.shots.slice(1).forEach((followingShot) => { if (followingShot.mode_inherited !== false && !followingShot.takes?.length) followingShot.mode = inheritedMode; });
          } else current.mode_inherited = false;
          save(); render();
        };
        const duration = el("input", controlCss); duration.type = "number"; duration.min = "0.2"; duration.max = "150"; duration.step = "0.1"; duration.value = current.duration_seconds || 5; duration.onchange = () => { current.duration_seconds = Math.max(.2, Math.min(150, Number(duration.value) || 5)); save(); render(); };
        const fps = el("input", controlCss); fps.type = "number"; fps.min = "1"; fps.max = "120"; fps.step = "1"; fps.value = current.fps || 24; fps.onchange = () => { current.fps = Math.max(1, Math.min(120, Number(fps.value) || 24)); save(); render(); };
        const continuity = el("select", controlCss); [["off","不承接上一镜头"],["tail","承接安全尾帧"],["motion","承接视频 + 音频上下文"],["auto_seamless","自动无缝接力（推荐）"]].forEach(([key,label]) => { const o=document.createElement("option");o.value=key;o.textContent=label;o.selected=(current.continuity?.strategy||"off")===key;continuity.append(o);}); continuity.onchange=()=>{current.continuity={...(current.continuity||{}),enabled:continuity.value!=="off",strategy:continuity.value,context_frames:current.continuity?.context_frames||22,source:current.continuity?.source||"initial"};save();render();};
        toolbarMain.append(field("模式", mode, 145), field("时长（秒）", duration, 72), field("帧率 FPS", fps, 72), field("续接方式", continuity, 168));
        const continuationFields = [];
        if (current.continuity?.enabled) {
          const source = el("select", controlCss);
          [["initial", "用上一段原始版本续接（默认）"], ["final", "优先用上一段最终版本续接"]].forEach(([key, label]) => { const o = document.createElement("option"); o.value = key; o.textContent = label; o.selected = (current.continuity?.source || "initial") === key; source.append(o); });
          source.onchange = () => { current.continuity = { ...(current.continuity || {}), source: source.value }; save(); };
          toolbarMain.append(field("续接版本", source, 185));
        }
        if (current.mode === "fl2v" && current.continuity?.enabled) {
          const startPolicy = el("select", controlCss);
          [["previous_tail","以上段尾帧作首帧"],["explicit_first","保留手动首帧，尾段仅参考"]].forEach(([key,label]) => { const o=document.createElement("option");o.value=key;o.textContent=label;o.selected=(current.continuity?.fl2v_start_policy||"previous_tail")===key;startPolicy.append(o); });
          startPolicy.onchange=()=>{current.continuity={...(current.continuity||{}),fl2v_start_policy:startPolicy.value};save();};
          continuationFields.push(field("首尾帧起点", startPolicy, 188));
        }
        if (current.mode === "i2v" && current.continuity?.enabled) {
          const startPolicy = el("select", controlCss);
          [["previous_tail","以上段尾帧作首帧（连续优先）"],["explicit_first","保留上传首帧（新起点优先）"]].forEach(([key,label]) => { const o=document.createElement("option");o.value=key;o.textContent=label;o.selected=(current.continuity?.i2v_start_policy||"previous_tail")===key;startPolicy.append(o); });
          startPolicy.onchange=()=>{current.continuity={...(current.continuity||{}),i2v_start_policy:startPolicy.value};save();};
          continuationFields.push(field("图生起点", startPolicy, 210));
        }
        toolbarBody.append(toolbarMain);
        if (continuationFields.length) {
          const continuationRow = el("div", "display:grid;grid-template-columns:minmax(205px,.9fr) minmax(240px,1fr) minmax(0,2.92fr);gap:9px;align-items:end;padding-top:9px;border-top:1px solid #1e354a;min-width:0;");
          const continuationLabel = el("div", "display:flex;align-items:center;height:34px;padding:0 9px;color:#80c8ff;font-size:12px;font-weight:700;");
          continuationLabel.textContent = current.mode === "fl2v" ? "首尾帧衔接设置" : "图生衔接设置";
          const continuationHint = el("div", "display:flex;align-items:center;min-height:34px;color:#789db8;font-size:11px;line-height:1.45;");
          continuationHint.textContent = current.mode === "fl2v" ? "决定当前镜头首帧是自动承接上一段尾帧，还是保留手动上传的首帧。" : "决定优先承接上一段尾帧，还是保留当前镜头上传的首帧。";
          continuationRow.append(continuationLabel, ...continuationFields, continuationHint);
          toolbarBody.append(continuationRow);
        }
        toolbar.append(shotTitle, toolbarBody); toolbar.addEventListener("toggle",()=>{shotSettingsOpen=toolbar.open;shotTitle.textContent=`镜头 ${project.shots.indexOf(current)+1} 设置`;requestAnimationFrame(fitToContent);});
        const uploadFiles = async (target, files) => {
          if (!files.length) return;
          try {
            const uploaded = (await uploadStoredFiles(files)).map((item) => item.asset);
            current.assets = current.assets || { first_frame: null, last_frame: null, images: [], videos: [], audios: [] };
            if (["images", "videos", "audios"].includes(target)) {
              const existing = Array.isArray(current.assets[target]) ? current.assets[target] : (current.assets[target] ? [current.assets[target]] : []);
              current.assets[target] = [...existing, ...uploaded];
            } else current.assets[target] = uploaded[0];
            save(); render();
          } catch (error) { window.alert(`镜头素材上传失败：${error?.message || error}`); }
        };
        const uploadAsset = (target, accept, multiple = false) => {
          const chooser = document.createElement("input"); chooser.type = "file"; chooser.accept = accept; chooser.multiple = multiple;
          chooser.onchange = () => uploadFiles(target, [...(chooser.files || [])]);
          chooser.click();
        };
        current.assets = current.assets || { first_frame: null, last_frame: null, images: [], videos: [], audios: [] };
        for (const key of ["images", "videos", "audios"]) {
          if (!Array.isArray(current.assets[key])) current.assets[key] = current.assets[key] ? [current.assets[key]] : [];
        }
        const imageCount = current.assets.images.length;
        const videoCount = current.assets.videos.length;
        const audioCount = current.assets.audios.length;
        const selectedTake = current.takes?.find?.((x) => x.take_id === current.selected_take_id) || current.takes?.at?.(-1);
        const selectedSource = current.selected_take_source === "final" && selectedTake?.files?.final ? "final" : (selectedTake?.files?.initial ? "initial" : "final");
        const version = document.createElement("details"); version.open=versionOpen; version.style.cssText="width:100%;box-sizing:border-box;background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:0 10px;flex-shrink:0;";
        const versionTitle = markDrawerSummary(document.createElement("summary")); versionTitle.style.cssText="cursor:pointer;padding:10px 0;display:flex;align-items:center;justify-content:space-between;gap:8px;font-weight:700;color:#80c8ff;";
        const versionBody=el("div","display:flex;flex-direction:column;gap:8px;padding:0 0 10px;");
        const versionLabel = el("span", ""); versionLabel.textContent = "生成视频与版本";
        const quickShotSwitch=document.createElement("select");quickShotSwitch.setAttribute("aria-label","生成视频与版本快捷切换当前镜头");quickShotSwitch.title="快捷切换当前镜头";quickShotSwitch.style.cssText="min-width:142px;max-width:220px;padding:6px 8px;background:#102f49;color:#fff;border:1px solid #4d8fc0;border-radius:5px;font-size:12px;";project.shots.forEach((shot,index)=>{const option=document.createElement("option");option.value=shot.id;option.textContent=`当前镜头 ${index+1}`;option.selected=shot.id===current.id;quickShotSwitch.append(option);});quickShotSwitch.onchange=()=>{selected=quickShotSwitch.value;project.active_shot_id=selected;save();render();};
        quickShotSwitch.onclick=event=>event.stopPropagation(); versionTitle.append(versionLabel,quickShotSwitch); version.append(versionTitle,versionBody); version.addEventListener("toggle",()=>{versionOpen=version.open;requestAnimationFrame(fitToContent);});
        if (current.takes?.length) {
          const takes = el("select", "background:#102943;color:#fff;border:1px solid #3a6790;border-radius:5px;padding:7px;");
          let originalNumber = 0, refinementNumber = 0;
          current.takes.forEach((take) => {
            if (take.files?.initial) { originalNumber += 1; const o = document.createElement("option"); o.value = `${take.take_id}::initial`; o.textContent = `${profileLabel(take,"initial")} · ${originalNumber}`; o.selected = take.take_id === current.selected_take_id && selectedSource === "initial"; takes.append(o); }
            if (take.files?.final) { refinementNumber += 1; const o = document.createElement("option"); o.value = `${take.take_id}::final`; o.textContent = `${profileLabel(take,"final")} · ${refinementNumber}`; o.selected = take.take_id === current.selected_take_id && selectedSource === "final"; takes.append(o); }
          });
          takes.onchange = () => { const [takeId, source] = takes.value.split("::"); current.selected_take_id = takeId; current.selected_take_source = source; current.status = "approved"; save(); render(); };
          versionBody.append(takes);
          if (selectedTake?.files?.initial && selectedSource === "initial") {
            const continueFromOriginal = button("基于此原版继续二采 / 放大", () => {
              current.selected_take_id = selectedTake.take_id;
              current.selected_take_source = "initial";
              const finalWidget = this.widgets?.find((item) => item.name === "enable_final_video");
              if (!finalWidget) { window.alert("找不到二采设置，请重新打开工作台后再试。"); return; }
              finalWidget.value = true; finalWidget.callback?.(true);
              this.graph?.setDirtyCanvas?.(true, true);
              save();
              setSaveState("已锁定该一采版本。设置二采/放大参数后再次生成当前镜头；一采参数未变时将复用 latent，不会重新一采。", "#63e3a2");
              refinementDrawer.open = true;
              requestAnimationFrame(() => { refinementDrawer.scrollIntoView({ block: "center", behavior: "smooth" }); fitToContent(); });
            }, "#187d58");
            continueFromOriginal.title = "仅复用兼容的一采 latent；修改主采样、画布或提示词后会安全地重新一采";
            versionBody.append(continueFromOriginal);
          }
        }
        const editor = el("div", "display:flex;flex-direction:column;gap:8px;background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:10px;");
        const promptLabel = el("div", "font-weight:700;color:#80c8ff;"); promptLabel.textContent = "本镜头提示词";
        const prompt = el("textarea", "height:165px;resize:vertical;background:#091523;color:#e8f4ff;border:1px solid #37658d;border-radius:6px;padding:9px;font-size:14px;line-height:1.45;"); prompt.placeholder = "填写本镜头提示词"; prompt.value = current.prompt || ""; prompt.oninput = () => { current.prompt = prompt.value; save(); };
        const clearShot = button("清空本镜头全部内容", () => {
          if (!window.confirm(`确定清空“${current.name || "当前镜头"}”的提示词和素材吗？\n\n不会删除镜头本身或已有视频版本。`)) return;
          current.prompt = ""; current.assets = { first_frame: null, last_frame: null, images: [], videos: [], audios: [] };
          save(); render();
        }, "#7b3650");
        const note = el("div", "font-size:12px;color:#9dc3df;min-height:20px;"); note.textContent = current.takes?.length ? `当前镜头已有 ${current.takes.length} 个生成版本；选择的版本会用于下一镜头续接，交付节点会自动优先取最终版本。` : "尚未生成：点击镜头卡片编辑；点击卡片右上角圆标可加入批量生成和交付。";
        const remove = button("删除当前镜头", () => { if (project.shots.length <= 1) return; const i = project.shots.indexOf(current); selectedShotIds.delete(current.id); project.shots.splice(i, 1); renumberShots(); selected = project.shots[Math.min(i, project.shots.length - 1)].id; project.active_shot_id = selected; save(); render(); }, "#7b3650");
        editor.append(promptLabel, prompt, clearShot, note, remove);
        if (current.mode === "fl2v") {
          const shotIndex = project.shots.indexOf(current);
          const previousShot = shotIndex > 0 ? project.shots[shotIndex - 1] : null;
          const previousTake = previousShot?.takes?.find?.((item) => item.take_id === previousShot.selected_take_id) || previousShot?.takes?.at?.(-1);
          const previousTail = previousTake?.files?.tail ? outputViewUrl(project.project_id, previousShot.id, previousTake.take_id, previousTake.files.tail) : "";
          const inheritsPrevious = Boolean(current.continuity?.enabled && current.continuity?.fl2v_start_policy !== "explicit_first");
          const keyframePanel = el("div", "display:flex;flex-direction:column;gap:8px;background:#091523;border:1px solid #31516e;border-radius:7px;padding:9px;");
          const keyframeTitle = el("div", "display:flex;align-items:center;justify-content:space-between;gap:8px;font-weight:700;color:#80c8ff;");
          keyframeTitle.innerHTML = `<span>首尾帧控制</span><span style="font-size:11px;color:#789db8;font-weight:500">${Number(current.duration_seconds || 5).toFixed(1)} 秒</span>`;
          const keyframeGrid = el("div", "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;");
          const makeKeyframeSlot = (target, label, asset, inherited = false) => {
            const slot = el("div", `position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;min-height:142px;overflow:hidden;border:1px dashed ${inherited ? "#2bcf87" : "#4a789e"};border-radius:6px;background:#060d14;color:#9dc3df;cursor:${inherited ? "default" : "pointer"};`);
            slot.setAttribute("role", inherited ? "img" : "button"); slot.setAttribute("aria-label", inherited ? `${label}引用上一镜头尾帧` : `上传${label}`);
            const badge = el("span", `position:absolute;z-index:2;left:6px;top:6px;padding:2px 5px;border-radius:3px;background:${target === "first_frame" ? "#1da968" : "#d8942f"};color:#071018;font-size:10px;font-weight:800;`); badge.textContent = label;
            slot.append(badge);
            const sourceUrl = inherited ? previousTail : (asset ? storedViewUrl(asset) : "");
            if (sourceUrl) {
              const image = document.createElement("img"); image.src = sourceUrl; image.alt = inherited ? `${label}：上一镜头尾帧` : `${label}：${asset?.filename || ""}`; image.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#05090e;"; slot.append(image, badge);
            } else {
              const emptyTitle = el("div", "font-size:12px;font-weight:700;color:#b8d5e9;"); emptyTitle.textContent = inherited ? "引用上一镜头尾帧" : `点击或拖入${label}`;
              const emptyHint = el("div", "max-width:90%;font-size:10px;color:#66869d;text-align:center;"); emptyHint.textContent = inherited ? "上一镜头生成后自动显示" : "仅接收图片";
              slot.append(emptyTitle, emptyHint);
            }
            if (!inherited) {
              slot.tabIndex = 0; slot.onclick = () => uploadAsset(target, "image/*"); slot.onkeydown = (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); uploadAsset(target, "image/*"); } };
              slot.ondragover = (event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; slot.style.borderColor = "#72c5ff"; };
              slot.ondragleave = () => { slot.style.borderColor = "#4a789e"; };
              slot.ondrop = (event) => { event.preventDefault(); event.stopPropagation(); slot.style.borderColor = "#4a789e"; const images = [...(event.dataTransfer?.files || [])].filter((file) => resourceBucket(file) === "images"); if (images.length) uploadFiles(target, [images[0]]); };
              if (asset) {
                const clear = button("×", (event) => { event.stopPropagation(); current.assets[target] = null; save(); render(); }, "#71364b"); clear.title = `移除${label}`; clear.style.cssText += "position:absolute;z-index:3;right:5px;top:5px;padding:0 7px;font-size:16px;line-height:22px;"; slot.append(clear);
              }
            }
            return slot;
          };
          keyframeGrid.append(
            makeKeyframeSlot("first_frame", "首帧", current.assets.first_frame, inheritsPrevious),
            makeKeyframeSlot("last_frame", "尾帧", current.assets.last_frame, false),
          );
          const keyframeHint = el("div", "font-size:11px;color:#789db8;"); keyframeHint.textContent = inheritsPrevious ? "首帧由上一镜头尾帧自动承接；尾帧可选。也可以切换为手动首帧。" : "首帧和尾帧都可单独使用：只放首帧会从该画面向后生成，只放尾帧会把该画面作为结束约束；两张都放则使用完整首尾约束。";
          keyframePanel.append(keyframeTitle, keyframeGrid, keyframeHint);
          editor.insertBefore(keyframePanel, promptLabel);
        }
        if (selectedTake?.files) {
          const file = selectedTake.files[selectedSource] || selectedTake.files.initial;
          if (file) {
            const player = document.createElement("video"); player.controls = true; player.preload = "metadata"; player.style.cssText = "display:block;width:100%;aspect-ratio:16/9;max-height:300px;object-fit:contain;background:#05090e;border:1px solid #2e526e;border-radius:6px;";
            player.src = outputViewUrl(project.project_id, current.id, selectedTake.take_id, file);
            versionBody.append(player);
            if (selectedTake.files.initial && selectedTake.files.final) {
              const compare = el("div", "position:relative;width:100%;aspect-ratio:16/9;max-height:300px;background:#05090e;border:1px solid #2e526e;border-radius:6px;overflow:hidden;");
              const base = document.createElement("video");
              const enhanced = document.createElement("video");
              [base, enhanced].forEach((video) => { video.src = outputViewUrl(project.project_id, current.id, selectedTake.take_id, video === base ? selectedTake.files.initial : selectedTake.files.final); video.preload = "metadata"; video.muted = true; video.playsInline = true; video.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#05090e;"; compare.append(video); });
              enhanced.style.clipPath = "inset(0 0 0 50%)";
              const divider = el("div", "position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;transform:translateX(-1px);pointer-events:none;");
              const handle = el("div", "position:absolute;top:50%;left:50%;width:12px;height:25px;background:#fff;border-radius:8px;transform:translate(-50%,-50%);box-shadow:0 1px 4px #0008;pointer-events:none;"); divider.append(handle); compare.append(divider);
              const range = document.createElement("input"); range.type = "range"; range.min = "0"; range.max = "100"; range.value = "50"; range.setAttribute("aria-label", "调整原版和超分对比范围"); range.style.cssText = "position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;";
              range.oninput = () => { const value = Number(range.value); enhanced.style.clipPath = `inset(0 0 0 ${value}%)`; divider.style.left = `${value}%`; };
              compare.append(range);
              const label = el("div", "position:absolute;left:8px;top:8px;right:8px;display:flex;justify-content:space-between;color:#fff;font-size:11px;text-shadow:0 1px 3px #000;"); label.innerHTML = "<span>原版</span><span>超分</span>"; compare.append(label);
              const sync = (source, event) => { [base, enhanced].forEach((video) => { if (video !== source && Math.abs(video.currentTime - source.currentTime) > 0.08) video.currentTime = source.currentTime; if (event === "play") video.play().catch(() => {}); if (event === "pause") video.pause(); }); };
              base.onplay = () => sync(base, "play"); enhanced.onplay = () => sync(enhanced, "play"); base.onpause = () => sync(base, "pause"); enhanced.onpause = () => sync(enhanced, "pause"); base.ontimeupdate = () => sync(base, "time");
              let comparePlaying = false;
              const compareControls = el("div", "display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;");
              const comparePlay = button("▶ 播放动态对比", async () => { if(comparePlaying){base.pause();enhanced.pause();comparePlaying=false;comparePlay.textContent="▶ 播放动态对比";return;}if(Number.isFinite(base.duration)&&base.currentTime>=base.duration-.05){base.currentTime=0;enhanced.currentTime=0;}base.muted=false;enhanced.muted=true;try{await Promise.all([base.play(),enhanced.play()]);comparePlaying=true;comparePlay.textContent="Ⅱ 暂停动态对比";}catch(error){comparePlay.textContent="播放失败，请重新加载视频";} }, "#187d58");
              const timeSlider=document.createElement("input");timeSlider.type="range";timeSlider.min="0";timeSlider.max="1000";timeSlider.value="0";timeSlider.setAttribute("aria-label","原版和处理版播放进度");timeSlider.oninput=()=>{if(Number.isFinite(base.duration)){const t=Number(timeSlider.value)/1000*base.duration;base.currentTime=t;enhanced.currentTime=t;}};base.ontimeupdate=()=>{sync(base,"time");if(Number.isFinite(base.duration)&&base.duration>0)timeSlider.value=String(Math.round(base.currentTime/base.duration*1000));};base.onended=()=>{comparePlaying=false;comparePlay.textContent="▶ 播放动态对比";};
              const compareState=el("span","color:#82b9d9;font-size:11px;");compareState.textContent="原版 / 处理版同步播放";compareControls.append(comparePlay,timeSlider,compareState);
              versionBody.append(compare,compareControls);
            }
            const exports = el("div", "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;align-items:start;");
            const exportFile = (label, filename) => {
              if (!filename) return null;
              return button(label, () => {
                const link = document.createElement("a");
                link.href = outputViewUrl(project.project_id, current.id, selectedTake.take_id, filename);
                link.download = `${current.name || current.id}-${selectedTake.take_id}-${filename}`;
                document.body.append(link); link.click(); link.remove();
              }, "#275e90");
            };
            const initialActions = el("div", "display:flex;flex-direction:column;gap:6px;min-width:0;");
            const finalActions = el("div", "display:flex;flex-direction:column;gap:6px;min-width:0;");
            const exportInitial = exportFile("导出原始视频", selectedTake.files.initial);
            const exportFinal = exportFile("导出最终视频", selectedTake.files.final);
            if (exportInitial) initialActions.append(exportInitial);
            if (exportFinal) finalActions.append(exportFinal);
            if (initialActions.children.length) exports.append(initialActions);
            if (finalActions.children.length) exports.append(finalActions);
            versionBody.append(exports);
            const deleteSelected = button("删除当前选中视频", async () => {
                if (liveRunning) { window.alert("当前正在生成，不能删除视频。"); return; }
                const deletingFinal = selectedSource === "final";
                const candidates = (current.takes || []).filter((take) => take.files?.[selectedSource]);
                const selectedIndex = candidates.findIndex((take) => take.take_id === selectedTake.take_id);
                const selectedOptionLabel = selectedIndex >= 0 ? `${deletingFinal ? "超分" : "原版"} ${selectedIndex + 1}` : selectedTake.take_id;
                const filename = selectedTake.files[selectedSource];
                const scope = deletingFinal
                  ? "只会把该超分视频移入项目回收区，原始视频、其他版本、合并视频和输入素材都会保留。"
                  : "该原版及明确属于同一版本的附属输出将移入项目回收区，不会影响其他镜头、其他版本、合并视频或输入素材。";
                const confirmed = window.confirm(`确认删除当前选中的“${selectedOptionLabel}”吗？\n文件：${filename}\n\n${scope}`);
                if (!confirmed) return;
                deleteSelected.disabled = true; deleteSelected.textContent = "正在删除…";
                try {
                  const response = await api.fetchApi("/ref2va-director/delete-selected-video", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ project_id: project.project_id, shot_id: current.id, take_id: selectedTake.take_id, source: selectedSource }),
                  });
                  let body = null;
                  try { body = await response.json(); } catch (_) { /* handled below */ }
                  if (!response.ok || !body?.ok) {
                    const oldBackend = (response.status === 404 || response.status === 405) && !body?.error;
                    throw new Error(oldBackend ? "当前服务仍是旧版本，删除接口尚未加载；请重启 ComfyUI 后再试。" : (body?.error || `删除接口返回 ${response.status}`));
                  }
                  if (!body.project?.shots) throw new Error("后端没有返回更新后的项目状态，已阻止误报成功。");
                  mergeSaveAcknowledgement(body.project);
                  selected = String(project.active_shot_id || current.id);
                  const serialized = JSON.stringify(project);
                  dataWidget.value = serialized;
                  try { localStorage.setItem(nodeStorageKey, serialized); } catch (_) {}
                  render();
                  window.alert(body.message || "当前选中视频已移入项目回收区。");
                } catch (error) {
                  deleteSelected.disabled = false; deleteSelected.textContent = "删除当前选中视频";
                  window.alert(`删除当前选中视频失败：${error?.message || error}`);
                }
            }, "#8a3848");
            deleteSelected.style.cssText += "width:100%;border:1px solid #d56b78;font-weight:700;";
            versionBody.append(deleteSelected);
          }
        } else {
          const pending = el("div", "display:flex;width:100%;min-height:120px;max-height:180px;box-sizing:border-box;align-items:center;justify-content:center;text-align:center;font-size:12px;color:#7894aa;border:1px dashed #31516e;border-radius:6px;padding:14px;background:#07111c;"); pending.textContent = "生成后在这里选择版本并横向预览；选定版本将作为下一镜头的续接来源。"; versionBody.append(pending);
        }
        const workspace = el("div", "display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,34%);gap:10px;align-items:start;flex-shrink:0;");
        const assets = current.assets || {};
        const resourceDrawer = document.createElement("details"); resourceDrawer.open = shotResourceOpen; resourceDrawer.style.cssText = "box-sizing:border-box;width:100%;background:#091523;border:1px solid #31516e;border-radius:7px;padding:0 10px;";
        const summary = markDrawerSummary(document.createElement("summary")); summary.style.cssText = "cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;font-size:18px;";
        summary.textContent = `当前镜头资源 · 首帧${assets.first_frame ? " ✓" : " —"} · 尾帧${assets.last_frame ? " ✓" : " —"} · 图 ${imageCount} · 视频 ${videoCount} · 音频 ${audioCount}`;
        const resourceBody = el("div", "display:flex;flex-direction:column;gap:10px;padding:0 0 10px;");
        const resourceActions = el("div", "display:flex;flex-wrap:wrap;gap:8px;");
        if (current.mode !== "fl2v") resourceActions.append(button("上传首帧", () => uploadAsset("first_frame", "image/*"), "#48698a"), button("上传尾帧", () => uploadAsset("last_frame", "image/*"), "#48698a"));
        resourceActions.append(button(`添加参考图（${imageCount}）`, () => uploadAsset("images", "image/*", true), "#48698a"), button(`添加参考视频（${videoCount}）`, () => uploadAsset("videos", "video/*", true), "#48698a"), button(`添加参考音频（${audioCount}）`, () => uploadAsset("audios", "audio/*", true), "#48698a"));
        resourceActions.append(button("清空本镜头素材", () => {
          if (!window.confirm(`确定清空“${current.name || "当前镜头"}”的全部素材吗？\n\n不会清空提示词，也不会删除已有视频版本。`)) return;
          current.assets = { first_frame: null, last_frame: null, images: [], videos: [], audios: [] };
          save(); render();
        }, "#7b3650"));
        const dropZone = el("div", "display:flex;align-items:center;justify-content:center;min-height:64px;border:1px dashed #4a789e;border-radius:6px;color:#9dc3df;background:#0a1724;font-size:12px;text-align:center;padding:10px;");
        dropZone.textContent = "拖入图片、视频或音频；图片作为参考图，首帧/尾帧请使用上方按钮";
        const setDropActive = (active) => { dropZone.style.borderColor = active ? "#72c5ff" : "#4a789e"; dropZone.style.background = active ? "#12324c" : "#0a1724"; };
        dropZone.ondragover = (event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; setDropActive(true); };
        dropZone.ondragleave = () => setDropActive(false);
        dropZone.ondrop = async (event) => {
          event.preventDefault(); setDropActive(false);
          const files = [...(event.dataTransfer?.files || [])];
          const groups = { images: [], videos: [], audios: [] };
          for (const file of files) {
            if (file.type.startsWith("image/")) groups.images.push(file);
            else if (file.type.startsWith("video/")) groups.videos.push(file);
            else if (file.type.startsWith("audio/")) groups.audios.push(file);
          }
          for (const target of ["images", "videos", "audios"]) await uploadFiles(target, groups[target]);
        };
        resourceBody.append(resourceActions, dropZone);
        const assetUrl = (asset) => {
          return `/view?filename=${encodeURIComponent(asset?.filename || "")}&subfolder=${encodeURIComponent(asset?.subfolder || "")}&type=${encodeURIComponent(asset?.type || "input")}`;
        };
        const removeAsset = (target, index = 0) => {
          if (["first_frame", "last_frame"].includes(target)) current.assets[target] = null;
          else current.assets[target].splice(index, 1);
          save(); render();
        };
        const assetCard = (label, target, asset, index = 0, mediaType = "image") => {
          const card = el("div", "position:relative;display:flex;flex-direction:column;gap:6px;min-width:0;background:#0d1d2d;border:1px solid #31516e;border-radius:6px;padding:7px;");
          const top = el("div", "display:flex;gap:6px;align-items:center;min-width:0;");
          const kind = el("span", "flex:0 0 auto;color:#80c8ff;font-size:11px;font-weight:700;"); kind.textContent = label;
          const filename = el("span", "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#dceeff;font-size:11px;"); filename.textContent = asset?.filename || "未命名素材"; filename.title = asset?.filename || "";
          const remove = button("×", () => removeAsset(target, index), "#71364b"); remove.title = `移除${label}`; remove.setAttribute("aria-label", `移除${label} ${asset?.filename || ""}`); remove.style.cssText += "margin-left:auto;padding:1px 7px;font-size:16px;line-height:20px;";
          top.append(kind, filename, remove); card.append(top);
          if (mediaType === "image") {
            const preview = document.createElement("img"); preview.src = assetUrl(asset); preview.alt = `${label} ${asset?.filename || ""}`; preview.loading = "lazy"; preview.style.cssText = "width:100%;height:112px;object-fit:contain;background:#05090e;border-radius:4px;"; card.append(preview);
          } else if (mediaType === "video") {
            const preview = document.createElement("video"); preview.src = assetUrl(asset); preview.controls = true; preview.preload = "metadata"; preview.style.cssText = "width:100%;height:112px;background:#05090e;border-radius:4px;"; card.append(preview);
          } else {
            const preview = document.createElement("audio"); preview.src = assetUrl(asset); preview.controls = true; preview.preload = "metadata"; preview.style.cssText = "width:100%;height:36px;"; card.append(preview);
          }
          return card;
        };
        const gallery = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;");
        if (current.mode !== "fl2v" && assets.first_frame) gallery.append(assetCard("首帧", "first_frame", assets.first_frame));
        if (current.mode !== "fl2v" && assets.last_frame) gallery.append(assetCard("尾帧", "last_frame", assets.last_frame));
        assets.images.forEach((asset, index) => gallery.append(assetCard(`参考图 ${index + 1}`, "images", asset, index)));
        assets.videos.forEach((asset, index) => gallery.append(assetCard(`参考视频 ${index + 1}`, "videos", asset, index, "video")));
        assets.audios.forEach((asset, index) => gallery.append(assetCard(`参考音频 ${index + 1}`, "audios", asset, index, "audio")));
        if (gallery.children.length) resourceBody.append(gallery);
        else { const empty = el("div", "padding:12px;border:1px dashed #31516e;border-radius:6px;color:#7894aa;font-size:12px;text-align:center;"); empty.textContent = "当前镜头尚未上传资源。"; resourceBody.append(empty); }
        resourceDrawer.append(summary, resourceBody);
        resourceDrawer.addEventListener("toggle", () => { shotResourceOpen = resourceDrawer.open; summary.textContent = `当前镜头资源 · 首帧${assets.first_frame ? " ✓" : " —"} · 尾帧${assets.last_frame ? " ✓" : " —"} · 图 ${imageCount} · 视频 ${videoCount} · 音频 ${audioCount}`; requestAnimationFrame(fitToContent); });
        // Prompt editing stays on the left; the current-shot resource drawer
        // is the dedicated right column.  Generated versions are a separate
        // full-width horizontal row below both columns.
        workspace.append(editor, resourceDrawer);

        // Shot-level material timeline.  It is an advanced view over the same
        // stored assets, not a second director node or a detached window.
        current.timeline = current.timeline || { enabled: false, prompt: "", generation_start: 0, generation_end: Number(current.duration_seconds || 5), snap_seconds: .25, clips: [] };
        current.timeline.clips = Array.isArray(current.timeline.clips) ? current.timeline.clips : [];
        const timelineDrawer = document.createElement("details"); timelineDrawer.open = shotTimelineOpen;
        timelineDrawer.style.cssText = "box-sizing:border-box;width:100%;background:#081827;border:2px solid #22b8cf;border-radius:9px;padding:0 12px;box-shadow:inset 0 0 0 1px #0d3445;flex-shrink:0;";
        const timelineSummary = markDrawerSummary(document.createElement("summary"));
        timelineSummary.style.cssText = "cursor:pointer;padding:10px 0;color:#7ee7f2;font-weight:700;";
        timelineSummary.textContent = `高级素材编辑台 · ${current.timeline.enabled ? "已启用" : "未启用"} · ${current.timeline.clips.length} 段`;
        const timelineBody = el("div", "display:flex;flex-direction:column;gap:9px;padding:0 0 10px;");
        const timelineLibrary = el("div", "display:flex;flex-direction:column;gap:8px;padding:9px;background:#0b1522;border:1px solid #31516e;border-radius:7px;");
        const libraryHead = el("div", "display:flex;align-items:center;gap:8px;flex-wrap:wrap;");
        const libraryTitle = el("b", "color:#80c8ff;margin-right:auto;"); libraryTitle.textContent = "时间线专用素材库";
        const uploadTimeline = (kind, accept, usage="conditioning") => { const chooser=document.createElement("input");chooser.type="file";chooser.accept=accept;chooser.multiple=true;chooser.onchange=async()=>{try{const uploaded=(await uploadStoredFiles([...(chooser.files||[])])).map(item=>item.asset);let editOffset=current.timeline.clips.filter(c=>c.usage==="edit").reduce((end,c)=>Math.max(end,Number(c.start||0)+Number(c.duration||0)),0);const offset=current.timeline.clips.length*.25;uploaded.forEach((asset,index)=>{const clipDuration=Number(current.duration_seconds||5);const start=usage==="edit"?editOffset:Math.min(clipDuration,offset+index*.25);current.timeline.clips.push({id:`clip-${Math.random().toString(36).slice(2,12)}`,kind,usage,asset:{...asset},start,duration:clipDuration,source_in:0,source_out:clipDuration,role:"editable_reference",audio_enabled:true});if(usage==="edit")editOffset+=clipDuration;});current.timeline.enabled=true;shotTimelineOpen=true;save();render();}catch(error){window.alert(`时间线素材上传失败：${error?.message||error}`);}};chooser.click(); };
        const addCurrentTakeToEdit = () => { const take=current.takes?.find?.(item=>item.take_id===current.selected_take_id)||current.takes?.at?.(-1);const filename=take?.files?.final||take?.files?.initial;if(!take||!filename){window.alert("当前镜头还没有可编辑的生成成片。");return;}const asset={filename,subfolder:`video/Ref2VA_Director/${project.project_id}/shots/${current.id}/takes/${take.take_id}`,type:"output"};current.timeline.clips.push({id:`clip-${Math.random().toString(36).slice(2,12)}`,kind:"video",usage:"edit",asset,start:current.timeline.clips.filter(c=>c.usage==="edit").reduce((sum,c)=>sum+Number(c.duration||0),0),duration:Number(current.duration_seconds||5),source_in:0,source_out:Number(current.duration_seconds||5),role:"editable_reference",audio_enabled:true});current.timeline.enabled=true;shotTimelineOpen=true;save();render();};
        libraryHead.append(libraryTitle, button("＋引导图片",()=>uploadTimeline("image","image/*","conditioning"),"#32658a"), button("添加当前镜头成片",addCurrentTakeToEdit,"#167a88"), button("添加外部剪辑视频",()=>uploadTimeline("video","video/*","edit"),"#187d58"), button("＋引导音频",()=>uploadTimeline("audio","audio/*","conditioning"),"#32658a"));
        const libraryGrid=el("div","display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:7px;");
        const librarySeen=new Set(); current.timeline.clips.forEach((clip,index)=>{const key=`${clip.kind}:${clip.asset?.subfolder||""}/${clip.asset?.filename||""}`;if(librarySeen.has(key))return;librarySeen.add(key);const card=el("div","display:flex;align-items:center;gap:7px;min-width:0;padding:7px;background:#0d2436;border:1px solid #2d526d;border-radius:5px;");const badge=el("b","flex:0 0 auto;color:#6fe4ef;font-size:10px;");badge.textContent={image:"图片",video:"视频",audio:"音频"}[clip.kind]||clip.kind;const name=el("span","min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#dceeff;font-size:11px;");name.textContent=clip.asset?.filename||`素材 ${index+1}`;card.append(badge,name);libraryGrid.append(card);});
        if(!libraryGrid.children.length){const empty=el("div","padding:10px;color:#7894aa;font-size:11px;text-align:center;border:1px dashed #31516e;border-radius:5px;");empty.textContent="此素材库只服务当前镜头的高级时间线；可直接添加，也可从普通镜头素材同步。";libraryGrid.append(empty);}
        timelineLibrary.append(libraryHead,libraryGrid); timelineBody.append(timelineLibrary);
        const timelinePrompt = el("textarea", "box-sizing:border-box;width:100%;min-height:92px;resize:vertical;background:#091523;color:#e8f4ff;border:1px solid #37658d;border-radius:6px;padding:9px;font-size:13px;line-height:1.45;");
        timelinePrompt.placeholder = "只填写时间线相关补充指令，例如：0-2 秒严格跟随固定引导动作，2-5 秒平滑延续；不会覆盖本镜头主提示词。";
        timelinePrompt.value = current.timeline.prompt || "";
        timelinePrompt.oninput = () => { current.timeline.prompt = timelinePrompt.value; save(); };
        timelineBody.append(field("时间线补充提示词（独立）", timelinePrompt));
        const timelineControls = el("div", "display:grid;grid-template-columns:minmax(140px,1fr) repeat(3,minmax(92px,.7fr));gap:8px;align-items:end;");
        const enabled = el("input", "width:18px;height:18px;accent-color:#21c7d9;"); enabled.type = "checkbox"; enabled.checked = Boolean(current.timeline.enabled);
        enabled.onchange = () => { current.timeline.enabled = enabled.checked; save(); render(); };
        const enabledBox = el("label", "display:flex;align-items:center;gap:8px;min-height:34px;color:#dceeff;font-size:12px;font-weight:700;"); enabledBox.append(enabled, document.createTextNode("启用时间线参与本镜头生成"));
        const numberControl = (value, min, max, step, onChange) => { const input = el("input", controlCss); input.type="number"; input.min=String(min); input.max=String(max); input.step=String(step); input.value=String(value); input.onchange=()=>onChange(Number(input.value)); return input; };
        const shotDuration = Number(current.duration_seconds || 5);
        const timelineDuration = Math.max(shotDuration, ...current.timeline.clips.filter(clip=>clip.usage==="edit").map(clip=>Number(clip.start||0)+Number(clip.duration||0)), .1);
        const rangeStart = numberControl(current.timeline.generation_start ?? 0, 0, shotDuration, .1, (value)=>{ current.timeline.generation_start=Math.max(0,Math.min(shotDuration,value||0)); current.timeline.generation_end=Math.max(current.timeline.generation_start+.01,Number(current.timeline.generation_end||shotDuration)); save(); render(); });
        const rangeEnd = numberControl(current.timeline.generation_end ?? shotDuration, .01, shotDuration, .1, (value)=>{ current.timeline.generation_end=Math.max(Number(current.timeline.generation_start||0)+.01,Math.min(shotDuration,value||shotDuration)); save(); render(); });
        const snap = el("select", controlCss); [[0,"关闭吸附"],[.1,"0.1 秒"],[.25,"0.25 秒"],[.5,"0.5 秒"],[1,"1 秒"]].forEach(([value,label])=>{const option=document.createElement("option");option.value=String(value);option.textContent=label;option.selected=Number(current.timeline.snap_seconds??.25)===value;snap.append(option);}); snap.onchange=()=>{current.timeline.snap_seconds=Number(snap.value);save();};
        timelineControls.append(enabledBox, field("生成起点",rangeStart), field("生成终点",rangeEnd), field("拖动吸附",snap));
        const timelineActions = el("div", "display:flex;flex-wrap:wrap;gap:7px;");
        const syncAssets = () => {
          const existing = new Set(current.timeline.clips.map((clip)=>`${clip.kind}:${clip.asset?.type||"input"}:${clip.asset?.subfolder||""}/${clip.asset?.filename||""}`));
          const add = (kind, asset, index) => { const key=`${kind}:${asset?.type||"input"}:${asset?.subfolder||""}/${asset?.filename||""}`; if(existing.has(key))return; existing.add(key); current.timeline.clips.push({id:`clip-${Math.random().toString(36).slice(2,12)}`,kind,asset:{...asset},start:Math.min(shotDuration, index*.25),duration:shotDuration,source_in:0,source_out:shotDuration,role:"editable_reference",audio_enabled:true}); };
          (current.assets.images||[]).forEach((asset,index)=>add("image",asset,index));
          (current.assets.videos||[]).forEach((asset,index)=>add("video",asset,index));
          (current.assets.audios||[]).forEach((asset,index)=>add("audio",asset,index));
          current.timeline.enabled = true; shotTimelineOpen=true; save(); render();
        };
        const renderEdit = async () => { const editClips=current.timeline.clips.filter(clip=>clip.usage==="edit"&&clip.kind==="video");if(!editClips.length){window.alert("请先添加当前镜头成片或外部剪辑视频。");return;}try{const response=await api.fetchApi("/ref2va-director/render-edit-timeline",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clips:editClips})});const body=await response.json();if(!response.ok||!body?.ok)throw new Error(body?.error||`编辑接口返回 ${response.status}`);editPreviewAsset=body.asset;shotTimelineOpen=true;render();}catch(error){window.alert(`生成编辑预览失败：${error?.message||error}`);}};
        timelineActions.append(button("同步当前镜头参考素材", syncAssets, "#32658a"), button("生成剪辑预览", renderEdit, "#187d58"));
        const ruler = el("div", "position:relative;height:154px;overflow:hidden;background:#06111c;border:1px solid #31516e;border-radius:6px;user-select:none;");
        const selectionStart = Math.max(0,Math.min(shotDuration,Number(current.timeline.generation_start||0)));
        const selectionEnd = Math.max(selectionStart,Math.min(shotDuration,Number(current.timeline.generation_end||shotDuration)));
        const rangeOverlay = el("div", `position:absolute;z-index:0;top:0;bottom:0;left:${selectionStart/timelineDuration*100}%;width:${(selectionEnd-selectionStart)/timelineDuration*100}%;background:#19d4e322;border-left:2px solid #20d9ea;border-right:2px solid #20d9ea;pointer-events:none;`); ruler.append(rangeOverlay);
        for(let second=0;second<=Math.ceil(timelineDuration);second++){const tick=el("div",`position:absolute;z-index:1;top:0;bottom:0;left:${Math.min(100,second/timelineDuration*100)}%;border-left:1px solid #29465d;pointer-events:none;`);const label=el("span","position:absolute;top:2px;left:3px;color:#6e99b7;font-size:9px;");label.textContent=`${second}s`;tick.append(label);ruler.append(tick);}
        const kindLabel={image:"图",video:"视频",audio:"音频"};
        const roleLabel={editable_reference:"可编辑参考",fixed_guide:"固定引导",boundary_only:"仅边界"};
        current.timeline.clips.forEach((clip,index)=>{
          const lane={video:32,image:76,audio:120}[clip.kind]??32;
          const start=Math.max(0,Math.min(timelineDuration,Number(clip.start||0))), durationValue=Math.max(.05,Math.min(timelineDuration-start,Number(clip.duration||shotDuration)));
          const block=el("div",`position:absolute;z-index:2;left:${start/timelineDuration*100}%;width:${Math.max(2,durationValue/timelineDuration*100)}%;top:${lane}px;height:30px;box-sizing:border-box;padding:5px 7px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border:2px solid ${selectedTimelineClipId===clip.id?"#fff":"#2ce0ef"};border-radius:5px;background:${clip.kind==="audio"?"#65458d":clip.kind==="image"?"#81602d":"#176a7b"};color:#fff;font-size:10px;cursor:grab;`);
          block.textContent=`${clip.usage==="edit"?"剪辑":"引导"}${kindLabel[clip.kind]||clip.kind} ${index+1} · ${clip.usage==="edit"?"成片编辑":(roleLabel[clip.role]||clip.role)} · ${clip.asset?.filename||"素材"}`; block.title=block.textContent;
          block.onclick=(event)=>{event.stopPropagation();selectedTimelineClipId=clip.id;shotTimelineOpen=true;render();};
          block.onpointerdown=(event)=>{event.preventDefault();event.stopPropagation();selectedTimelineClipId=clip.id;block.setPointerCapture(event.pointerId);const originX=event.clientX,originStart=start,rect=ruler.getBoundingClientRect();block.onpointermove=(move)=>{const raw=originStart+(move.clientX-originX)/Math.max(1,rect.width)*timelineDuration;const step=Number(current.timeline.snap_seconds||0);clip.start=Math.max(0,Math.min(timelineDuration-durationValue,step?Math.round(raw/step)*step:raw));block.style.left=`${clip.start/timelineDuration*100}%`;};block.onpointerup=()=>{block.onpointermove=null;save();render();};};
          ruler.append(block);
        });
        ruler.onclick=(event)=>{const rect=ruler.getBoundingClientRect();current.timeline.playhead=Math.max(0,Math.min(timelineDuration,(event.clientX-rect.left)/rect.width*timelineDuration));save();render();};
        timelineBody.append(timelineControls,timelineActions,ruler);
        const selectedClip=current.timeline.clips.find((clip)=>clip.id===selectedTimelineClipId);
        if(selectedClip){
          const inspector=el("div","display:grid;grid-template-columns:repeat(6,minmax(90px,1fr));gap:7px;align-items:end;background:#0d1d2d;border:1px solid #31516e;border-radius:6px;padding:8px;");
          const updateNumber=(key,value,min=0)=>{selectedClip[key]=Math.max(min,Number(value)||0);save();render();};
          const role=el("select",controlCss); const roleOptions=selectedClip.usage==="edit"?[["editable_reference","镜头剪辑片段（不参与 H3 参考）"]]:[["editable_reference","可编辑参考"],["fixed_guide","固定引导"],["boundary_only","仅使用首尾边界"]];roleOptions.forEach(([value,label])=>{const option=document.createElement("option");option.value=value;option.textContent=label;option.selected=selectedClip.role===value;role.append(option);});role.disabled=selectedClip.usage==="edit";role.onchange=()=>{selectedClip.role=role.value;save();render();};
          inspector.append(field(selectedClip.usage==="edit"?"用途":"角色",role),field("时间线位置",numberControl(selectedClip.start,0,9999,.1,(v)=>updateNumber("start",v))),field("片段时长",numberControl(selectedClip.duration,.05,9999,.1,(v)=>updateNumber("duration",v,.05))),field("源入点",numberControl(selectedClip.source_in||0,0,9999,.1,(v)=>updateNumber("source_in",v))),field("源出点",numberControl(selectedClip.source_out||selectedClip.duration,.01,9999,.1,(v)=>updateNumber("source_out",v,.01))));
          const inspectorActions=el("div","display:flex;flex-direction:column;gap:5px;");
          inspectorActions.append(button("在游标处分割",()=>{const at=Number(current.timeline.playhead??(Number(selectedClip.start)+Number(selectedClip.duration)/2));const local=at-Number(selectedClip.start);if(local<=.05||local>=Number(selectedClip.duration)-.05){window.alert("游标必须位于所选片段内部。 ");return;}const right={...selectedClip,id:`clip-${Math.random().toString(36).slice(2,12)}`,start:at,duration:Number(selectedClip.duration)-local,source_in:Number(selectedClip.source_in||0)+local};selectedClip.duration=local;selectedClip.source_out=Number(selectedClip.source_in||0)+local;current.timeline.clips.push(right);selectedTimelineClipId=right.id;save();render();},"#32658e"),button("移除时间线片段",()=>{if(!window.confirm(`只从时间线移除“${selectedClip.asset?.filename||"所选素材"}”？\n不会删除输入文件或素材库中的原素材。`))return;current.timeline.clips=current.timeline.clips.filter((clip)=>clip.id!==selectedClip.id);selectedTimelineClipId=null;save();render();},"#7b3650"));
          if(selectedClip.kind==="video"){const audioToggle=el("label","display:flex;align-items:center;gap:5px;color:#cfe7f7;font-size:11px;");const box=document.createElement("input");box.type="checkbox";box.checked=selectedClip.audio_enabled!==false;box.onchange=()=>{selectedClip.audio_enabled=box.checked;save();};audioToggle.append(box,document.createTextNode("携带原视频音频"));inspectorActions.prepend(audioToggle);}
          inspector.append(inspectorActions); timelineBody.append(inspector);
        }
        const timelineHint=el("div","font-size:11px;color:#7fa9c3;line-height:1.5;");timelineHint.textContent="“引导”片段只负责 H3 条件；“剪辑视频”是独立成片，可来自当前镜头或电脑外部，能裁切、分割、排序并导出，不会作为 H3 参考素材。";timelineBody.append(timelineHint);
        if(editPreviewAsset){const editResult=el("div","display:grid;grid-template-columns:minmax(0,1fr) auto;gap:9px;align-items:center;padding:9px;background:#07111c;border:1px solid #2bcf87;border-radius:7px;");const editVideo=document.createElement("video");editVideo.controls=true;editVideo.preload="metadata";editVideo.src=storedViewUrl(editPreviewAsset);editVideo.style.cssText="display:block;width:100%;aspect-ratio:16/9;max-height:360px;object-fit:contain;background:#000;border-radius:5px;";const exportEdit=button("导出剪辑视频",()=>{const link=document.createElement("a");link.href=storedViewUrl(editPreviewAsset);link.download=`${current.name||current.id}-剪辑成片.mp4`;document.body.append(link);link.click();link.remove();},"#187d58");editResult.append(editVideo,exportEdit);timelineBody.append(editResult);}
        timelineDrawer.append(timelineSummary,timelineBody);timelineDrawer.addEventListener("toggle",()=>{shotTimelineOpen=timelineDrawer.open;requestAnimationFrame(fitToContent);});
        // Rendered below the prompt/version workspace as an independent tool.

        const mergedDrawer = document.createElement("details");
        mergedDrawer.style.cssText = "background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:0 10px;flex-shrink:0;";
        const mergedSummary = markDrawerSummary(document.createElement("summary"));
        mergedSummary.style.cssText = "cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;";
        mergedSummary.textContent = "合并后的视频预览";
        const mergedBody = el("div", "padding:0 0 10px;");
        if (mergedPreview) {
          const deliveryActions = el("div", "display:flex;justify-content:flex-end;gap:8px;margin-bottom:9px;");
          let comparisonPlaying = false;
          const playTogetherButton = button("同时播放", async () => {
            const videos = Array.from(mergedBody.querySelectorAll("video"));
            if (videos.length < 2) return;
            if (comparisonPlaying) {
              videos.forEach((video) => video.pause());
              comparisonPlaying = false; playTogetherButton.textContent = "同时播放";
              return;
            }
            const maximumCommonTime = Math.max(0, Math.min(...videos.map((video) => Number.isFinite(video.duration) ? video.duration : Infinity)) - 0.05);
            let startTime = Math.min(...videos.map((video) => Number(video.currentTime || 0)));
            if (!Number.isFinite(startTime) || startTime >= maximumCommonTime) startTime = 0;
            videos.forEach((video, index) => { video.currentTime = startTime; video.muted = index > 0; });
            try {
              await Promise.all(videos.map((video) => video.play()));
              comparisonPlaying = true; playTogetherButton.textContent = "同时暂停";
            } catch (_) {
              videos.forEach((video) => video.pause());
              comparisonPlaying = false; playTogetherButton.textContent = "同时播放";
            }
          }, "#187d58");
          const deleteDeliveryButton = button("删除本次合并视频", deleteMergedDelivery, "#8d3f53");
          deleteDeliveryButton.title = "只删除当前显示的合并交付及其验收文件，不删除任何源镜头版本";
          deliveryActions.append(playTogetherButton, deleteDeliveryButton); mergedBody.append(deliveryActions);
          const sourceAudit=el("div","display:flex;flex-wrap:wrap;gap:7px;margin-bottom:10px;padding:9px;border:1px solid #37698d;border-radius:6px;background:#091522;");
          const sourceTitle=el("b","width:100%;color:#80c8ff;font-size:12px;");sourceTitle.textContent="本次合并实际使用文件";sourceAudit.append(sourceTitle);
          (mergedSourceManifest||[]).forEach(item=>{const badge=el("span",`padding:5px 8px;border-radius:5px;background:${item.source==="final"?"#176a4e":"#31516e"};color:#fff;font-size:11px;`);badge.textContent=`${item.shot_name||item.shot_id}：${item.source==="final"?profileLabel({output_profile:item.output_profile},"final"):"H3 原始生成"} · ${item.filename||"文件未知"}`;sourceAudit.append(badge);});
          if(!mergedSourceManifest?.length){const unknown=el("span","color:#ffcf70;font-size:11px;");unknown.textContent="旧合并结果没有来源清单；重新合并后会逐镜头显示原始或处理版。";sourceAudit.append(unknown);}const manifestExport=button("导出合并清单",()=>{const payload={product:"梦镜 DreamShot",version:FRONTEND_VERSION,merged_file:mergedPreview?.filename,created_at:new Date().toISOString(),shots:mergedSourceManifest};const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=`梦镜 DreamShot-${FRONTEND_VERSION}-合并文件清单.json`;document.body.append(link);link.click();link.remove();window.setTimeout(()=>URL.revokeObjectURL(link.href),1000);},"#32658a");manifestExport.style.marginLeft="auto";sourceAudit.append(manifestExport);mergedBody.append(sourceAudit);
          const mergeSourceStem=(mergedSourceManifest||[]).map((item,index)=>`S${index+1}-${item.source==="final"?profileLabel({output_profile:item.output_profile},"final"):"Original"}`).join("_").replace(/[^A-Za-z0-9._+-]+/g,"-").slice(0,120)||"Selected-Sources";
          const comparison = el("div", "display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;align-items:start;");
          const addPreview = (asset, label, detail, exportName) => {
            if (!asset) return;
            const column = el("div", "min-width:0;display:flex;flex-direction:column;gap:7px;");
            const previewHeader = el("div", "display:flex;align-items:center;justify-content:space-between;gap:10px;");
            const previewLabel = el("div", "min-width:0;");
            previewLabel.innerHTML = `<b style="color:#dceeff;font-size:13px">${label}</b><br><span style="color:#789fb9;font-size:11px">${detail}</span>`;
            const exportButton = button("导出", () => { const link = document.createElement("a"); link.href = storedViewUrl(asset); link.download = `梦镜 DreamShot-${FRONTEND_VERSION}-${mergeSourceStem}-${exportName}.mp4`; document.body.append(link); link.click(); link.remove(); }, "#187d58");
            exportButton.style.padding = "5px 9px"; previewHeader.append(previewLabel, exportButton); column.append(previewHeader);
            const player = document.createElement("video");
            player.controls = true; player.preload = "metadata"; player.src = storedViewUrl(asset);
            player.style.cssText = "display:block;width:100%;max-height:460px;object-fit:contain;background:#05090e;border:1px solid #2e526e;border-radius:6px;";
            followConfiguredAspect(player); followMediaAspect(player); column.append(player); comparison.append(column);
          };
          addPreview(mergedBaselinePreview, "24 FPS 完整直接拼接", "保留全部镜头内容，用于对照接缝和时长", "24FPS完整拼接");
          addPreview(mergedPreview, mergedBaselinePreview ? "48 FPS 接缝修复" : "合并视频", mergedBaselinePreview ? "完整时长，仅平滑接缝局部" : "当前合并结果", "48FPS接缝修复");
          mergedBody.append(comparison);
          const comparisonPlayers = Array.from(comparison.querySelectorAll("video"));
          if (comparisonPlayers.length >= 2) {
            const leader = comparisonPlayers[0], follower = comparisonPlayers[1];
            leader.addEventListener("timeupdate", () => { if (comparisonPlaying && Math.abs(follower.currentTime - leader.currentTime) > 0.08) follower.currentTime = leader.currentTime; });
            leader.addEventListener("pause", () => { if (comparisonPlaying && !leader.ended) follower.pause(); });
            leader.addEventListener("ended", () => { comparisonPlayers.forEach((video) => video.pause()); comparisonPlaying = false; playTogetherButton.textContent = "同时播放"; });
          }
          if (continuityReport?.baseline && continuityReport?.repaired) {
            const facts = el("div", "display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;margin-top:12px;");
            const fact = (label, media) => {
              const card = el("div", "padding:9px 11px;border:1px solid #31516e;border-radius:6px;background:#091522;color:#b9dbf2;font-size:12px;");
              card.innerHTML = `<b style="color:#dceeff">${label}</b><br>${media.fps} FPS · ${media.frames} 帧 · ${Number(media.duration_seconds).toFixed(3)} 秒`;
              facts.append(card);
            };
            fact("24 FPS 原始拼接", continuityReport.baseline);
            fact("48 FPS 接缝修复", continuityReport.repaired);
            mergedBody.append(facts);
            const players = Array.from(comparison.querySelectorAll("video"));
            for (const seam of continuityReport.seams || []) {
              const panel = el("div", "display:flex;flex-direction:column;gap:8px;margin-top:12px;padding:11px;border:1px solid #37698d;border-radius:7px;background:#091522;");
              const header = el("div", "display:flex;flex-wrap:wrap;align-items:center;gap:8px;color:#b9dbf2;font-size:12px;");
              const title = el("b", "color:#dceeff;margin-right:auto;");
              title.textContent = `接缝 ${seam.index} · ${Number(seam.time_seconds).toFixed(3)} 秒`;
              const score = el("span", "color:#9fc8e7;");
              score.textContent = `变化值 ${seam.baseline.peak_change} → ${seam.repaired.peak_change} · 改善 ${seam.improvement_percent}%`;
              const jump = button("跳到接缝", () => { players.forEach((player) => { player.currentTime = Math.max(0, Number(seam.time_seconds) - 0.15); player.pause(); }); }, "#32658a");
              let looping = false;
              const loopHandlers = new Map();
              const loop = button("循环接缝", () => {
                looping = !looping;
                loop.textContent = looping ? "停止循环" : "循环接缝";
                players.forEach((player) => {
                  const previousHandler = loopHandlers.get(player);
                  if (previousHandler) player.removeEventListener("timeupdate", previousHandler);
                  loopHandlers.delete(player);
                  if (!looping) { player.pause(); return; }
                  const handler = () => { if (player.currentTime >= Number(seam.loop_end_seconds)) player.currentTime = Number(seam.loop_start_seconds); };
                  loopHandlers.set(player, handler);
                  player.addEventListener("timeupdate", handler);
                  player.currentTime = Number(seam.loop_start_seconds);
                  player.play().catch(() => {});
                });
              }, "#187d58");
              header.append(title, score, jump, loop);
              panel.append(header);
              const strips = el("div", "display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:10px;");
              [[seam.baseline, "24 FPS 接缝前后 7 帧"], [seam.repaired, "48 FPS 接缝前后 7 帧"]].forEach(([evidence, label]) => {
                if (!evidence?.strip_filename) return;
                const item = el("div", "min-width:0;color:#789fb9;font-size:11px;");
                item.textContent = label;
                const image = document.createElement("img");
                image.src = storedViewUrl({ filename: evidence.strip_filename, subfolder: mergedPreview.subfolder, type: mergedPreview.type, cacheKey: Date.now() });
                image.alt = label;
                image.style.cssText = "display:block;width:100%;margin-top:5px;border:1px solid #263f57;border-radius:5px;background:#05090e;";
                item.append(image); strips.append(item);
              });
              panel.append(strips);
              mergedBody.append(panel);
            }
          }
        } else {
          const emptyMerged = el("div", "padding:14px;border:1px dashed #31516e;border-radius:6px;color:#7894aa;font-size:12px;text-align:center;");
          emptyMerged.textContent = "合并选中镜头后，视频会显示在这里。";
          mergedBody.append(emptyMerged);
        }
        mergedDrawer.append(mergedSummary, mergedBody);
        mergedDrawer.addEventListener("toggle", () => { mergedSummary.textContent = "合并后的视频预览"; requestAnimationFrame(fitToContent); });

        const libraryDrawer=document.createElement("details");libraryDrawer.style.cssText="background:#0b1522;border:1px solid #37698d;border-radius:7px;padding:0 10px;flex-shrink:0;";
        const librarySummary=markDrawerSummary(document.createElement("summary"));librarySummary.style.cssText="cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;";
        const takeEntries=[];for(const shot of project.shots||[]){for(const take of shot.takes||[]){for(const source of ["initial","final"]){const filename=take.files?.[source];if(filename)takeEntries.push({kind:"take",shot,take,source,asset:{filename,subfolder:`video/Ref2VA_Director/${project.project_id}/shots/${shot.id}/takes/${take.take_id}`,type:"output"}});}}}
        const originalEntries=takeEntries.filter(item=>item.source==="initial");
        const hasVideoUpscale=(entry)=>["NVIDIA RTX","TE FlashVSR"].includes(String(entry.take?.output_profile?.final_upscale_method||"关闭"));
        const secondaryEntries=takeEntries.filter(item=>item.source==="final"&&!hasVideoUpscale(item));
        const videoUpscaleEntries=takeEntries.filter(item=>item.source==="final"&&hasVideoUpscale(item));
        const deliveryEntries=(project.settings?.video_library||[]).filter(item=>item?.kind==="delivery"&&item.asset?.filename);
        librarySummary.textContent=`视频库 · 原版 ${originalEntries.length} · 二采/超分 ${secondaryEntries.length} · 视频放大 ${videoUpscaleEntries.length} · 合并 ${deliveryEntries.length}`;
        const libraryBody=el("div","display:flex;flex-direction:column;gap:12px;padding:0 0 10px;");
        const sendToRefinement=(entry)=>{selected=entry.shot.id;project.active_shot_id=selected;entry.shot.selected_take_id=entry.take.take_id;entry.shot.selected_take_source="initial";const finalWidget=this.widgets?.find(item=>item.name==="enable_final_video");if(!finalWidget){window.alert("找不到二采设置，请重新打开工作台后再试。");return;}finalWidget.value=true;finalWidget.callback?.(true);this.graph?.setDirtyCanvas?.(true,true);save();setSaveState(`已锁定 ${entry.shot.name} 的原版；设置二采/放大参数后再次生成当前镜头。`,"#63e3a2");render();requestAnimationFrame(()=>{const panel=[...root.querySelectorAll("details")].find(item=>item.textContent.includes("超分 / 多次采样"));if(panel){panel.open=true;panel.scrollIntoView({block:"center",behavior:"smooth"});}});};
        const sendToExistingUpscale=(entry)=>{existingUpscaleSource={project_id:project.project_id,shot_id:entry.shot.id,take_id:entry.take.take_id,source:entry.source,asset:entry.asset,label:`${entry.shot.name} · ${profileLabel(entry.take,entry.source)} · ${entry.take.take_id}`};existingUpscaleOpen=true;render();requestAnimationFrame(()=>{const panel=[...root.querySelectorAll("details")].find(item=>item.textContent.includes("已有视频放大"));if(panel){panel.open=true;panel.scrollIntoView({block:"center",behavior:"smooth"});}});};
        const addLibraryCard=(entry,grid)=>{const card=el("div","display:flex;flex-direction:column;gap:7px;padding:9px;background:#07111c;border:1px solid #31516e;border-radius:7px;");const title=el("b","color:#dceeff;font-size:12px;");title.textContent=entry.kind==="take"?`${entry.shot.name} · ${profileLabel(entry.take,entry.source)} · ${entry.take.take_id}`:`历史合并 · ${entry.created_at?new Date(entry.created_at).toLocaleString():entry.id}`;const player=document.createElement("video");player.controls=true;player.preload="metadata";player.src=storedViewUrl(entry.asset);player.style.cssText="display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:5px;";const actions=el("div","display:grid;grid-template-columns:repeat(2,1fr);gap:7px;");const exportBtn=button("导出",()=>{const link=document.createElement("a");link.href=storedViewUrl(entry.asset);link.download=`梦镜 DreamShot-${FRONTEND_VERSION}-${entry.asset.filename}`;document.body.append(link);link.click();link.remove();},"#187d58");actions.append(exportBtn);if(entry.kind==="take"&&entry.source==="initial")actions.append(button("送去二采",()=>sendToRefinement(entry),"#32658a"));if(entry.kind==="take")actions.append(button("送去放大",()=>sendToExistingUpscale(entry),"#167a88"));const removeBtn=button("删除",async()=>{const label=title.textContent;if(!window.confirm(`确认从视频库删除“${label}”吗？文件会移入项目回收区。`))return;try{let response;if(entry.kind==="take")response=await api.fetchApi("/ref2va-director/delete-selected-video",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({project_id:project.project_id,shot_id:entry.shot.id,take_id:entry.take.take_id,source:entry.source})});else response=await api.fetchApi("/ref2va-director/delete-delivery",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({project_id:project.project_id,merged_filename:entry.asset.filename})});const body=await response.json();if(!response.ok||!body?.ok)throw new Error(body?.error||"删除失败");if(entry.kind==="delivery")project.settings.video_library=project.settings.video_library.filter(item=>item.id!==entry.id);else mergeSaveAcknowledgement(body.project);save();render();}catch(error){window.alert(`视频库删除失败：${error?.message||error}`);}},"#8d3f53");actions.append(removeBtn);card.append(title,player,actions);grid.append(card);};
        const addLibraryGroup=(title,detail,entries)=>{const section=el("section","display:flex;flex-direction:column;gap:8px;padding:9px;border:1px solid #263f57;border-radius:7px;background:#091522;");const heading=el("b","color:#80c8ff;font-size:13px;");heading.textContent=`${title}（${entries.length}）`;const hint=el("span","color:#789fb9;font-size:11px;");hint.textContent=detail;section.append(heading,hint);const grid=el("div","display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;");entries.slice().reverse().forEach(entry=>addLibraryCard(entry,grid));if(!entries.length){const empty=el("div","padding:12px;color:#7894aa;text-align:center;border:1px dashed #31516e;border-radius:6px;");empty.textContent="暂无视频。";grid.append(empty);}section.append(grid);libraryBody.append(section);};
        addLibraryGroup("原版视频","一采输出。可直接送去二采，或送去已有视频放大。",originalEntries);addLibraryGroup("二采 / H3 超分视频","已完成二采或 H3 Latent 超分、但尚未 RTX/TE 视频放大的版本；可继续送去放大。",secondaryEntries);addLibraryGroup("RTX / TE 视频放大","已使用 NVIDIA RTX 或 TE FlashVSR 的处理版本；仍可再次送去放大。",videoUpscaleEntries);addLibraryGroup("合并结果","每次合并都保留独立历史版本。",deliveryEntries);libraryDrawer.append(librarySummary,libraryBody);libraryDrawer.addEventListener("toggle",()=>requestAnimationFrame(fitToContent));

        const parameterLabels = {
          aspect_ratio: "输出比例", megapixels: "百万像素", seed_mode: "种子模式", noise_seed: "种子数值",
          scheduler: "调度器", steps: "采样步数", denoise: "降噪强度",
          sampler_name: "采样器", main_extra_steps: "Sigma 加步",
          main_start_at_sigma: "Sigma 起点", main_spacing: "Sigma 曲线",
        };
        const makeWidgetControl = (widgetName, afterChange) => {
          const widget = this.widgets?.find((item) => item.name === widgetName);
          if (!widget) return null;
          const choices = Array.isArray(widget.options?.values) ? widget.options.values : null;
          let control;
          if (choices) {
            control = el("select", controlCss);
            for (const value of choices) { const option = document.createElement("option"); option.value = String(value); option.textContent = String(value); option.selected = value === widget.value; control.append(option); }
            control.onchange = () => { widget.value = choices.find((value) => String(value) === control.value) ?? control.value; widget.callback?.(widget.value); this.graph?.setDirtyCanvas?.(true, true); afterChange?.(widget.value); };
          } else if (typeof hiddenDefaults[widgetName] === "boolean") {
            control = document.createElement("input"); control.type = "checkbox"; control.checked = Boolean(widget.value);
            control.onchange = () => { widget.value = control.checked; widget.callback?.(widget.value); this.graph?.setDirtyCanvas?.(true, true); afterChange?.(widget.value); };
          } else if (typeof hiddenDefaults[widgetName] === "string") {
            control = el("input", controlCss); control.type = "text"; control.value = String(widget.value ?? hiddenDefaults[widgetName]);
            control.onchange = () => { widget.value = control.value; widget.callback?.(widget.value); this.graph?.setDirtyCanvas?.(true, true); afterChange?.(widget.value); };
          } else {
            control = el("input", controlCss); control.type = "number"; control.value = String(widget.value ?? hiddenDefaults[widgetName]);
            if (Number.isFinite(Number(widget.options?.min))) control.min = String(widget.options.min);
            if (Number.isFinite(Number(widget.options?.max))) control.max = String(widget.options.max);
            control.step = String(widget.options?.step ?? 1);
            control.onchange = () => { widget.value = Number(control.value); normalizeHiddenWidgets(); control.value = String(widget.value); widget.callback?.(widget.value); this.graph?.setDirtyCanvas?.(true, true); afterChange?.(widget.value); };
          }
          return control;
        };
        const makeFamilyLoraControl = (widgetName, familyTokens, preferredName) => {
          const widget = this.widgets?.find((item) => item.name === widgetName);
          if (!widget) return null;
          const allChoices = Array.isArray(widget.options?.values) ? widget.options.values.map(String) : [];
          const familyChoices = allChoices.filter((name) => familyTokens.some((token) => name.toLowerCase().includes(token)));
          const preferred = familyChoices.includes(preferredName) ? preferredName : familyChoices.find((name) => /4(?:step|_step|-step)/i.test(name)) || familyChoices[0] || allChoices[0] || "";
          if (!allChoices.includes(String(widget.value || ""))) {
            widget.value = preferred;
            widget.callback?.(widget.value);
            this.graph?.setDirtyCanvas?.(true, true);
          }
          const control = el("select", controlCss);
          if (!allChoices.length) {
            const missing = document.createElement("option"); missing.value = ""; missing.textContent = "未找到对应模型家族的 LoRA"; control.append(missing); control.disabled = true;
          } else {
            for (const value of allChoices) { const option = document.createElement("option"); option.value = value; option.textContent = value; option.selected = value === String(widget.value); control.append(option); }
          }
          control.onchange = () => { widget.value = control.value; widget.callback?.(widget.value); this.graph?.setDirtyCanvas?.(true, true); };
          return control;
        };
        const systemDrawer = document.createElement("details");
        systemDrawer.style.cssText = "background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:0 10px;flex-shrink:0;";
        const systemSummary = markDrawerSummary(document.createElement("summary"));
        systemSummary.style.cssText = "cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;display:block;";
        systemSummary.textContent = "导演系统加载器（模型 / VAE）";
        const systemBody = el("div", "display:flex;flex-direction:column;gap:9px;padding:0 0 10px;min-width:0;");
        const baseModelGrid = el("div", "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;min-width:0;");
        const baseSystemLabels = {
          ref2va_unet_name: "参考 / 连续镜头 UNet（Ref2VA）",
          fl2va_unet_name: "文生 / 图生 / 首尾帧 UNet（FL2VA）",
          weight_dtype: "模型数据类型", clip_name: "H3 CLIP", clip_type: "CLIP 类型",
          video_vae_name: "视频 VAE", audio_vae_name: "音频 VAE",
        };
        for (const [widgetName, label] of Object.entries(baseSystemLabels)) {
          const control = makeWidgetControl(widgetName, widgetName === "aspect_ratio" ? updateConfiguredAspects : undefined); if (!control) continue;
          const row = field(label, control, 0); row.style.minWidth = "0";
          baseModelGrid.append(row);
        }
        const turboWidget = this.widgets?.find((item) => item.name === "enable_turbo_lora");
        const turboSwitch = el("select", controlCss);
        for (const [value, label] of [["off", "关"], ["on", "开"]]) {
          const option = document.createElement("option"); option.value = value; option.textContent = label;
          option.selected = (value === "on") === Boolean(turboWidget?.value); turboSwitch.append(option);
        }
        const refTurboModelControl = makeFamilyLoraControl("ref2va_turbo_lora_name", ["ref2va", "ref2v"], hiddenDefaults.ref2va_turbo_lora_name);
        const flTurboModelControl = makeFamilyLoraControl("fl2v_turbo_lora_name", ["fl2va", "fl2v"], hiddenDefaults.fl2v_turbo_lora_name);
        const turboStrengthControl = makeWidgetControl("turbo_lora_strength");
        const refTurboModelField = field("参考 / 连续参考模式 · LoRA（默认 4 步）", refTurboModelControl, 0);
        const flTurboModelField = field("文生 / 图生 / 首尾帧模式 · LoRA（默认 4 步）", flTurboModelControl, 0);
        const turboStrengthField = field("Turbo LoRA 强度", turboStrengthControl, 0);
        const activeTurboHint = el("div", "grid-column:1/-1;padding:7px 9px;border:1px solid #31516e;border-radius:5px;color:#9dc3df;font-size:12px;");
        const refreshActiveTurboHint = () => {
          const usesRef = ["ref2va", "continuous_ref2va"].includes(String(current?.mode || ""));
          const selectedLora = usesRef ? refTurboModelControl?.value : flTurboModelControl?.value;
          activeTurboHint.textContent = `当前镜头“${modeLabel(current?.mode)}”将自动使用 ${usesRef ? "Ref2VA" : "FL2V"} 模型与同家族 LoRA：${selectedLora || "未选择"}`;
          activeTurboHint.style.color = selectedLora ? "#63e3a2" : "#ffb36b";
        };
        refTurboModelControl?.addEventListener("change", refreshActiveTurboHint);
        flTurboModelControl?.addEventListener("change", refreshActiveTurboHint);
        const setTurboControlsVisible = (enabled) => {
          refTurboModelField.style.display = enabled ? "flex" : "none";
          flTurboModelField.style.display = enabled ? "flex" : "none";
          turboStrengthField.style.display = enabled ? "flex" : "none";
          activeTurboHint.style.display = enabled ? "block" : "none";
          requestAnimationFrame(fitToContent);
        };
        turboSwitch.onchange = () => {
          if (!turboWidget) return;
          turboWidget.value = turboSwitch.value === "on";
          turboWidget.callback?.(turboWidget.value);
          setTurboControlsVisible(turboWidget.value);
          this.graph?.setDirtyCanvas?.(true, true);
        };
        setTurboControlsVisible(Boolean(turboWidget?.value));
        refreshActiveTurboHint();
        const turboDrawer = document.createElement("details");
        turboDrawer.style.cssText = "background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:0 10px;flex-shrink:0;";
        const turboSummary = markDrawerSummary(document.createElement("summary"));
        turboSummary.style.cssText = "cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;display:block;";
        turboSummary.textContent = "Turbo LoRA（按需展开）";
        const turboBody = el("div", "display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;padding:0 0 10px;min-width:0;align-items:end;");
        turboBody.append(
          field("按镜头模式自动匹配", turboSwitch, 0),
          refTurboModelField,
          flTurboModelField,
          turboStrengthField,
          activeTurboHint,
        );
        const loraStackWidget = this.widgets?.find((item) => item.name === "lora_stack_json");
        const loraSourceWidget = this.widgets?.find((item) => item.name === "ref2va_turbo_lora_name") || this.widgets?.find((item) => item.name === "turbo_lora_name");
        const loraChoices = Array.isArray(loraSourceWidget?.options?.values) ? loraSourceWidget.options.values : [];
        let loraStack = [];
        try { const parsed = JSON.parse(String(loraStackWidget?.value || "[]")); if (Array.isArray(parsed)) loraStack = parsed.filter((item) => item && typeof item === "object"); } catch (_) { loraStack = []; }
        const stackSection = el("div", "grid-column:1/-1;display:flex;flex-direction:column;gap:8px;padding-top:8px;border-top:1px solid #263f57;");
        const stackHeader = el("div", "display:flex;align-items:center;gap:9px;");
        const stackTitle = el("b", "color:#80c8ff;font-size:12px;flex:1;"); stackTitle.textContent = "附加 LoRA Stack（按顺序应用到 Ref2VA / FL2VA）";
        const addLoraButton = button("＋ 添加 LoRA", () => { if (!loraChoices.length) { window.alert("ComfyUI 的 loras 目录中没有可用 LoRA。"); return; } loraStack.push({ name: String(loraChoices[0]), strength: 1.0 }); commitLoraStack(); drawLoraStack(); }, "#187d58");
        stackHeader.append(stackTitle, addLoraButton);
        const stackRows = el("div", "display:flex;flex-direction:column;gap:7px;");
        const commitLoraStack = () => {
          if (!loraStackWidget) return;
          loraStackWidget.value = JSON.stringify(loraStack.map((item) => ({ name: String(item.name || ""), strength: Number(item.strength ?? 1.0) })));
          loraStackWidget.callback?.(loraStackWidget.value); this.graph?.setDirtyCanvas?.(true, true);
        };
        const drawLoraStack = () => {
          stackRows.replaceChildren();
          if (!loraStack.length) { const empty = el("div", "padding:8px;border:1px dashed #31516e;border-radius:5px;color:#7894aa;font-size:11px;text-align:center;"); empty.textContent = "尚未添加普通 LoRA；Turbo LoRA 可独立开关。"; stackRows.append(empty); return; }
          loraStack.forEach((item, index) => {
            const row = el("div", "display:grid;grid-template-columns:32px minmax(0,1fr) 130px auto;gap:8px;align-items:end;");
            const order = el("span", "align-self:center;color:#9dc3df;font-size:12px;text-align:center;"); order.textContent = String(index + 1);
            const model = el("select", controlCss);
            for (const value of loraChoices) { const option = document.createElement("option"); option.value = String(value); option.textContent = String(value); option.selected = String(value) === String(item.name); model.append(option); }
            if (!loraChoices.includes(item.name) && item.name) { const missing = document.createElement("option"); missing.value = String(item.name); missing.textContent = `缺失：${item.name}`; missing.selected = true; model.prepend(missing); }
            model.onchange = () => { item.name = model.value; commitLoraStack(); };
            const strength = el("input", controlCss); strength.type = "number"; strength.min = "-100"; strength.max = "100"; strength.step = "0.01"; strength.value = String(Number(item.strength ?? 1.0));
            strength.onchange = () => { item.strength = Math.max(-100, Math.min(100, Number(strength.value) || 0)); strength.value = String(item.strength); commitLoraStack(); };
            const removeLora = button("删除", () => { loraStack.splice(index, 1); commitLoraStack(); drawLoraStack(); }, "#7b3650");
            row.append(order, field("LoRA 模型", model, 0), field("权重", strength, 0), removeLora); stackRows.append(row);
          });
        };
        stackSection.append(stackHeader, stackRows); turboBody.append(stackSection); drawLoraStack();
        systemBody.append(baseModelGrid);
        const unloadRow = el("div", "display:flex;align-items:center;gap:9px;padding:8px 0 1px;border-top:1px solid #263f57;margin-top:2px;");
        const unloadStatus = el("span", "flex:1;min-width:0;color:#9dc3df;font-size:12px;"); unloadStatus.textContent = "模型加载后可在这里释放显存缓存。";
        const unloadButton = button("释放显存 / 卸载模型", async () => {
          if (!window.confirm("确定要卸载当前已加载模型并释放显存吗？\n\n不会删除模型文件；下一次生成需要重新加载模型。运行中的任务不会被强制中断。")) return;
          unloadButton.disabled = true; unloadStatus.textContent = "正在卸载模型并清理显存…";
          try {
            const response = await api.fetchApi("/ref2va-director/unload-models", { method: "POST" });
            const raw = await response.text();
            let body = null; try { body = raw ? JSON.parse(raw) : null; } catch (_) { throw new Error(`后端返回了无效响应（HTTP ${response.status}）`); }
            if (!response.ok || !body?.ok) throw new Error(body?.error || "卸载失败");
            const before = Number(body.free_before); const after = Number(body.free_after);
            const gained = Number.isFinite(before) && Number.isFinite(after) ? ` 可用显存增加约 ${(Math.max(0, after - before) / 1073741824).toFixed(1)} GB。` : "";
            unloadStatus.textContent = `${body.message || "模型已卸载。"}${gained}`; unloadStatus.style.color = "#63e3a2";
          } catch (error) { unloadStatus.textContent = `卸载失败：${error?.message || error}`; unloadStatus.style.color = "#ffb3b3"; }
          finally { unloadButton.disabled = false; }
        }, "#8b4b32"); unloadButton.title = "卸载模型并释放显存，不删除模型文件";
        unloadRow.append(unloadStatus, unloadButton); systemBody.append(unloadRow);
        systemDrawer.append(systemSummary, systemBody);
        systemDrawer.addEventListener("toggle", () => { systemSummary.textContent = "导演系统加载器（模型 / VAE）"; requestAnimationFrame(fitToContent); });
        turboDrawer.append(turboSummary, turboBody);
        turboDrawer.addEventListener("toggle", () => { turboSummary.textContent = "Turbo LoRA（按需展开）"; requestAnimationFrame(fitToContent); });
        const parameterDrawer = document.createElement("details");
        parameterDrawer.style.cssText = "background:#0b1522;border:1px solid #263f57;border-radius:7px;padding:0 10px;flex-shrink:0;";
        const parameterSummary = markDrawerSummary(document.createElement("summary"));
        parameterSummary.style.cssText = "cursor:pointer;padding:10px 0;color:#80c8ff;font-weight:700;display:block;";
        parameterSummary.textContent = "首次生成参数（按需展开）";
        const parameterBody = el("div", "display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:9px;padding:0 0 10px;min-width:0;");
        const parameterRows = new Map();
        for (const [widgetName, label] of Object.entries(parameterLabels)) {
          const control = makeWidgetControl(widgetName, widgetName === "aspect_ratio" ? updateConfiguredAspects : undefined); if (!control) continue;
          const row = field(label, control, 140); parameterRows.set(widgetName, row); parameterBody.append(row);
        }
        const seedModeControl = parameterRows.get("seed_mode")?.querySelector("select");
        const seedValueControl = parameterRows.get("noise_seed")?.querySelector("input");
        const updateSeedMode = () => {
          const fixed = String(seedModeControl?.value || "随机") === "固定";
          if (seedValueControl) { seedValueControl.disabled = !fixed; seedValueControl.style.opacity = fixed ? "1" : ".55"; seedValueControl.title = fixed ? "固定种子：每次生成保持此数值" : "随机模式：加入队列时自动更换种子"; }
        };
        seedModeControl?.addEventListener("change", updateSeedMode); updateSeedMode();
        parameterDrawer.append(parameterSummary, parameterBody);
        parameterDrawer.addEventListener("toggle", () => { parameterSummary.textContent = "首次生成参数（按需展开）"; requestAnimationFrame(fitToContent); });

        const refinementLabels = {
          second_sampling_mode: "二采模式", latent_upscale_model: "H3 Latent 超分模型",
          second_megapixels: "二次百万像素",
          passes: "二次采样次数", upscale_passes: "超分后细化次数",
          refine_scheduler: "细化调度器", refine_steps: "每遍细化步数",
          refine_denoise: "细化降噪", refine_extra_steps: "细化 Sigma 加步",
          refine_start_at_sigma: "细化 Sigma 起点", refine_spacing: "细化 Sigma 曲线",
        };
        const refinementDrawer = document.createElement("details");
        refinementDrawer.style.cssText = parameterDrawer.style.cssText + ";margin-top:2px;box-sizing:border-box;";
        const refinementSummary = markDrawerSummary(document.createElement("summary")); refinementSummary.style.cssText = parameterSummary.style.cssText; refinementSummary.textContent = "超分 / 多次采样 / 最终视频放大（按需展开）";
        const refinementBody = el("div", "display:flex;flex-direction:column;gap:10px;padding:0 0 10px;min-width:0;");
        const refinementGrid = el("div", "display:grid;grid-template-columns:repeat(4,minmax(0,1fr));grid-auto-flow:row dense;gap:9px;padding:0 1px;min-width:0;align-items:start;");
        const setRefinementVisibility = (enabled) => { refinementGrid.style.display = enabled ? "grid" : "none"; requestAnimationFrame(fitToContent); };
        const latentOnlyFields = new Set(["latent_upscale_model", "second_megapixels", "upscale_passes"]);
        const refinementFieldRows = new Map();
        const updateRefinementMode = (modeValue) => {
          const isLatent = String(modeValue || "") === "H3 Latent 超分";
          refinementFieldRows.forEach((row, widgetName) => { row.style.display = !isLatent && latentOnlyFields.has(widgetName) ? "none" : ""; });
          requestAnimationFrame(fitToContent);
        };
        const enableWidget = this.widgets?.find((item) => item.name === "enable_final_video");
        const enableButton = button(enableWidget?.value ? "已开启 / 超分多次采样" : "开启 / 超分多次采样", () => {
          if (!enableWidget) return;
          enableWidget.value = !Boolean(enableWidget.value);
          enableWidget.callback?.(enableWidget.value);
          setRefinementVisibility(Boolean(enableWidget.value));
          enableButton.textContent = enableWidget.value ? "已开启 / 超分多次采样" : "开启 / 超分多次采样";
          this.graph?.setDirtyCanvas?.(true, true);
        }, enableWidget?.value ? "#187d58" : "#48698a");
        refinementBody.append(enableButton);
        for (const [widgetName, label] of Object.entries(refinementLabels)) {
          const control = makeWidgetControl(widgetName); if (!control) continue;
          const row = field(label, control, 0);
          row.style.minWidth = "0";
          refinementFieldRows.set(widgetName, row);
          refinementGrid.append(row);
        }
        const modeControl = refinementFieldRows.get("second_sampling_mode")?.querySelector("select");
        if (modeControl) {
          modeControl.addEventListener("change", () => updateRefinementMode(modeControl.value));
          updateRefinementMode(modeControl.value);
        }
        setRefinementVisibility(Boolean(this.widgets?.find((item) => item.name === "enable_final_video")?.value));
        refinementBody.append(refinementGrid);
        const rtxDivider = el("div", "height:1px;background:#263f57;margin:2px 0;");
        const rtxTitle = el("div", "font-size:12px;font-weight:700;color:#80c8ff;"); rtxTitle.textContent = "最终视频放大（位于 H3 二采之后，三选一）";
        const rtxGrid = el("div", "display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px;min-width:0;");
        const rtxEnableWidget = this.widgets?.find((item) => item.name === "enable_rtx_upscale");
        const finalMethodWidget = this.widgets?.find((item) => item.name === "final_upscale_method");
        if (finalMethodWidget && String(finalMethodWidget.value || "关闭") === "关闭" && Boolean(rtxEnableWidget?.value)) finalMethodWidget.value = "NVIDIA RTX";
        const finalMethodControl = makeWidgetControl("final_upscale_method");
        for (const [widgetName, label] of [["rtx_scale", "RTX 放大倍数"], ["rtx_quality", "RTX 质量"], ["rtx_filename_prefix", "保存文件名前缀"]]) {
          const control = makeWidgetControl(widgetName); if (!control) continue;
          rtxGrid.append(field(label, control, 0));
        }
        const teGrid = el("div", "display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px;min-width:0;");
        const teLabels = {
          te_flashvsr_model: "FlashVSR 模型", te_flashvsr_mode: "推理模式",
          te_flashvsr_precision: "计算精度", te_flashvsr_scale: "放大倍数",
          te_flashvsr_quality: "质量模式", te_flashvsr_spatial: "空间策略",
          te_flashvsr_memory: "显存策略", te_flashvsr_attention: "稀疏注意力",
          te_flashvsr_color_fix: "颜色修正",
        };
        for (const [widgetName, label] of Object.entries(teLabels)) {
          const control = makeWidgetControl(widgetName); if (!control) continue;
          teGrid.append(field(label, control, 0));
        }
        const upscaleHint = el("div", "padding:7px 9px;border:1px solid #31516e;border-radius:5px;color:#9dc3df;font-size:12px;");
        const updateFinalUpscale = () => {
          const method = String(finalMethodControl?.value || "关闭");
          rtxGrid.style.display = method === "NVIDIA RTX" ? "grid" : "none";
          teGrid.style.display = method === "TE FlashVSR" ? "grid" : "none";
          if (rtxEnableWidget) { rtxEnableWidget.value = false; rtxEnableWidget.callback?.(false); }
          upscaleHint.textContent = method === "关闭" ? "保留 H3 输出，不执行视频放大。" : method === "NVIDIA RTX" ? "使用 NVIDIA RTX 快速放大；不会再执行 TE FlashVSR。" : "H3 视频解码后执行 TE FlashVSR 超分修复；不会再执行 NVIDIA RTX 放大。";
          upscaleHint.style.color = method === "关闭" ? "#9dc3df" : "#63e3a2";
          requestAnimationFrame(fitToContent);
        };
        finalMethodControl?.addEventListener("change", updateFinalUpscale);
        updateFinalUpscale();
        refinementBody.append(rtxDivider, rtxTitle, field("最终放大方式", finalMethodControl, 0), upscaleHint, rtxGrid, teGrid); refinementDrawer.append(refinementSummary, refinementBody);
        refinementDrawer.addEventListener("toggle", () => { refinementSummary.textContent = "超分 / 多次采样 / 最终视频放大（按需展开）"; requestAnimationFrame(fitToContent); });
        // Keep the two generation drawers near the top of the director rail,
        // then place the merged delivery preview as a full-width footer.
        const existingUpscale = document.createElement("details");
        existingUpscale.open = existingUpscaleOpen;
        existingUpscale.style.cssText = "margin-top:8px;border:1px solid #31516e;border-radius:6px;background:#0b1a2b;padding:0 8px;";
        const existingSummary = document.createElement("summary"); existingSummary.textContent = "已有视频放大（直接添加视频，不重新生成）"; existingSummary.style.cssText = "cursor:pointer;color:#8fd3ff;font-weight:700;padding:9px 0;";
        const existingBody = document.createElement("div"); existingBody.style.cssText = "display:grid;grid-template-columns:minmax(240px,1fr) minmax(240px,1fr);gap:12px;padding:8px 0;align-items:stretch;";
        const existingPreviewColumn = document.createElement("div"); existingPreviewColumn.style.cssText = "display:flex;flex-direction:column;gap:6px;min-width:0;";
        const existingPreview = document.createElement("div"); existingPreview.style.cssText = "height:240px;min-height:180px;border:1px dashed #4b7598;border-radius:6px;background:#071321;display:flex;align-items:center;justify-content:center;color:#8fb4cf;text-align:center;overflow:hidden;position:relative;";
        const existingUploadHint = document.createElement("div"); existingUploadHint.textContent = existingUpscaleSource ? `已选择项目视频：${existingUpscaleSource.label}` : "拖入视频到上方，或点击视频区域选择视频"; existingUploadHint.style.cssText = "color:#8fb4cf;font-size:12px;text-align:center;min-height:18px;line-height:18px;"; existingPreviewColumn.append(existingPreview, existingUploadHint);
        const existingFile = document.createElement("input"); existingFile.type = "file"; existingFile.accept = "video/*"; existingFile.style.display="none";
        existingPreview.addEventListener("click",()=>{ if(existingCompare.style.display !== "block") existingFile.click(); }); existingPreview.addEventListener("dragover",e=>{e.preventDefault();}); existingPreview.addEventListener("drop",e=>{e.preventDefault(); if(e.dataTransfer.files?.[0]){existingFile.files=e.dataTransfer.files; existingFile.dispatchEvent(new Event("change"));}});
        let existingPreviewVideo = null;
        const showExistingSourcePreview=(url,label)=>{if(existingPreviewVideo)existingPreviewVideo.remove();const v=document.createElement("video");v.controls=true;v.preload="metadata";v.src=url;v.style.cssText="width:100%;height:100%;object-fit:contain;background:#000;";existingPreview.insertBefore(v,existingCompare);existingPreviewVideo=v;existingUploadHint.textContent=label;};
        if(existingUpscaleSource?.asset)showExistingSourcePreview(storedViewUrl(existingUpscaleSource.asset),`已选择项目视频：${existingUpscaleSource.label}`);
        existingFile.addEventListener("change",()=>{const f=existingFile.files?.[0]; if(!f)return; existingUpscaleSource=null;showExistingSourcePreview(URL.createObjectURL(f),`已选择外部视频：${f.name}`);});
        const existingParams = document.createElement("div"); existingParams.style.cssText="display:flex;flex-direction:column;gap:8px;min-height:240px;";
        const existingEngine = document.createElement("select"); ["NVIDIA RTX","TE FlashVSR"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=v; existingEngine.append(o); });
        const existingScale = document.createElement("input"); existingScale.type="number"; existingScale.min="1"; existingScale.max="4"; existingScale.step="0.05"; existingScale.value="2";
        const existingQuality = document.createElement("select"); ["LOW","MEDIUM","HIGH","ULTRA"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=`RTX 质量 ${v}`; existingQuality.append(o); }); existingQuality.value="HIGH";
        [existingEngine, existingScale, existingQuality].forEach(control => { control.style.cssText="width:100%;box-sizing:border-box;height:31px;padding:5px 9px;background:#12304c;color:#e7f4ff;border:1px solid #37678e;border-radius:5px;font-size:13px;"; });
        const teMode = document.createElement("select"); ["tiny","tiny-long","full"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=`TE 模式 ${v}`; teMode.append(o); });
        const teQuality = document.createElement("select"); ["detail","balanced","throughput"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=`TE 质量 ${v}`; teQuality.append(o); }); teQuality.value="balanced";
        const teSpatial = document.createElement("select"); ["auto","full_frame","adaptive_tiles"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=`空间 ${v}`; teSpatial.append(o); });
        const teMemory = document.createElement("select"); ["auto","resident","staged"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=`显存 ${v}`; teMemory.append(o); }); teMemory.value="staged";
        const teAttention = document.createElement("select"); ["sparse_sage2","block_sparse_attn","auto"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=`注意力 ${v}`; teAttention.append(o); });
        const tePrecision = document.createElement("select"); ["bf16","fp16"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=`精度 ${v}`; tePrecision.append(o); });
        const teColorFix = document.createElement("select"); ["开启","关闭"].forEach(v => { const o=document.createElement("option"); o.value=v; o.textContent=v; teColorFix.append(o); }); teColorFix.value="开启";
        [teMode, teQuality, teSpatial, teMemory, teAttention, tePrecision, teColorFix].forEach(control => { control.style.cssText="width:100%;box-sizing:border-box;height:31px;padding:5px 9px;background:#12304c;color:#e7f4ff;border:1px solid #37678e;border-radius:5px;font-size:13px;"; });
        const existingStatus = document.createElement("div"); existingStatus.style.cssText="grid-column:1/-1;color:#9dc3df;font-size:12px;min-height:18px;"; existingStatus.textContent="等待执行";
        const existingProgress = document.createElement("progress"); existingProgress.max=100; existingProgress.value=0; existingProgress.style.cssText="grid-column:1/-1;width:100%;height:8px;display:none;";
        const existingRun = button("执行已有视频放大", async () => {
          const file = existingFile.files?.[0]; if (!file && !existingUpscaleSource) { existingStatus.textContent = "请先选择视频。"; existingStatus.style.color = "#ffcc66"; return; }
          existingRun.disabled = true; existingProgress.style.display = "block"; existingProgress.value = 10; existingStatus.textContent = existingUpscaleSource ? "正在提交项目原版放大任务…" : "正在上传视频并提交放大任务…"; existingStatus.style.color = "#9dc3df";
          try {
            let asset=existingUpscaleSource?.asset||null;let sourcePayload={};
            if(existingUpscaleSource){sourcePayload={project_id:existingUpscaleSource.project_id,shot_id:existingUpscaleSource.shot_id,take_id:existingUpscaleSource.take_id,source:existingUpscaleSource.source};}
            else {const form = new FormData(); form.append("image", file, file.name);const upload = await api.fetchApi("/upload/image", { method: "POST", body: form }); const uploadText = await upload.text(); try { asset = JSON.parse(uploadText); } catch (_) { throw new Error(`视频上传接口返回无效响应（${upload.status}）：${uploadText.slice(0, 160)}`); }if (!upload.ok || !asset.name) throw new Error(asset.error || "视频上传失败");sourcePayload={name:asset.name,subfolder:asset.subfolder||""};}
            existingProgress.value = 35;
            const response = await api.fetchApi("/ref2va-director/upscale-uploaded", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...sourcePayload, engine: existingEngine.value, scale: existingScale.value, quality: existingEngine.value === "TE FlashVSR" ? teQuality.value : existingQuality.value, mode: teMode.value, precision: tePrecision.value, spatial: teSpatial.value, memory: teMemory.value, attention: teAttention.value, color_fix: teColorFix.value === "开启" }) });
            const resultText = await response.text(); let result; try { result = JSON.parse(resultText); } catch (_) { throw new Error(`放大接口返回无效响应（${response.status}）：${resultText.slice(0, 160)}`); } if (!response.ok || !result.ok) throw new Error(result.error || "放大任务失败");
            existingProgress.value = 100; existingStatus.textContent = "放大完成，结果已临时保存；点击“导出放大后视频”下载。"; existingStatus.style.color = "#63e3a2";
            existingOutput = result; existingCompare.style.display = "block";
            existingOutputVideo.src = `/view?filename=${encodeURIComponent(result.filename)}&subfolder=${encodeURIComponent(result.subfolder || "")}&type=${encodeURIComponent(result.type || "temp")}`;
            existingOriginalVideo.src = storedViewUrl(asset); existingCompare.style.setProperty("--split", "50%"); existingCompareHandle.value = 50; existingOriginalVideo.currentTime = 0; existingOutputVideo.currentTime = 0; existingOriginalVideo.play().catch(()=>{}); existingOutputVideo.play().catch(()=>{});
          } catch (error) { existingStatus.textContent = String(error.message || error); existingStatus.style.color = "#ffcc66"; } finally { existingRun.disabled = false; }
        }, "#187d58"); existingRun.style.gridColumn="1/-1";
        const teColorField = field("颜色修正", teColorFix, 0);
        const teFields = [field("TE 推理模式", teMode, 0), field("TE 质量模式", teQuality, 0), field("空间策略", teSpatial, 0), field("显存策略", teMemory, 0), field("注意力后端", teAttention, 0), field("计算精度", tePrecision, 0)];
        const teGroup = document.createElement("div"); teGroup.style.cssText="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;"; teFields.forEach(x=>{x.style.cssText += ";min-width:0;display:flex;flex-direction:column;gap:4px;"; teGroup.append(x);});
        const updateExistingEngine = () => { const te = existingEngine.value === "TE FlashVSR"; teFields.forEach(x=>x.style.display=te?"flex":"none"); teColorField.style.display=te?"flex":"none"; existingQuality.parentElement.style.display=te?"none":"block"; scaleColorRow.style.gridTemplateColumns=te?"1fr 1fr":"1fr"; };
        let existingOutput = null;
        const existingCompare = document.createElement("div"); existingCompare.style.cssText = "display:none;position:absolute;inset:0;overflow:hidden;border-radius:6px;background:#000;--split:50%;z-index:2;";
        const existingOriginalVideo = document.createElement("video"); existingOriginalVideo.muted = true; existingOriginalVideo.playsInline = true; existingOriginalVideo.loop = true; existingOriginalVideo.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;";
        const existingOutputVideo = document.createElement("video"); existingOutputVideo.muted = true; existingOutputVideo.playsInline = true; existingOutputVideo.loop = true; existingOutputVideo.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;clip-path:inset(0 0 0 var(--split));";
        const originalLabel = document.createElement("span"); originalLabel.textContent = "原版"; originalLabel.style.cssText = "position:absolute;left:10px;top:8px;padding:3px 7px;background:#071321cc;color:#fff;border-radius:3px;z-index:3;font-size:12px;";
        const outputLabel = document.createElement("span"); outputLabel.textContent = "放大"; outputLabel.style.cssText = "position:absolute;right:10px;top:8px;padding:3px 7px;background:#071321cc;color:#fff;border-radius:3px;z-index:3;font-size:12px;";
        const existingCompareHandle = document.createElement("input"); existingCompareHandle.type = "range"; existingCompareHandle.min = 0; existingCompareHandle.max = 100; existingCompareHandle.value = 50; existingCompareHandle.title = "拖动比较原版与放大"; existingCompareHandle.setAttribute("aria-label", "拖动比较原版与放大"); existingCompareHandle.style.cssText = "position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;z-index:5;";
        const divider = document.createElement("div"); divider.style.cssText = "position:absolute;top:0;bottom:0;left:var(--split);width:2px;background:#fff;box-shadow:0 0 0 1px #0008;z-index:4;pointer-events:none;";
        existingCompareHandle.addEventListener("input", () => existingCompare.style.setProperty("--split", `${existingCompareHandle.value}%`)); existingCompareHandle.addEventListener("click", event => event.stopPropagation()); existingCompareHandle.addEventListener("pointerdown", event => event.stopPropagation());
        existingOutputVideo.addEventListener("play", () => existingOriginalVideo.play().catch(()=>{})); existingOutputVideo.addEventListener("pause", () => existingOriginalVideo.pause()); existingOutputVideo.addEventListener("seeking", () => { existingOriginalVideo.currentTime = existingOutputVideo.currentTime; }); existingOutputVideo.addEventListener("timeupdate", () => { if(Math.abs(existingOriginalVideo.currentTime - existingOutputVideo.currentTime) > 0.15) existingOriginalVideo.currentTime = existingOutputVideo.currentTime; });
        existingCompare.append(existingOriginalVideo, existingOutputVideo, originalLabel, outputLabel, divider, existingCompareHandle); existingPreview.append(existingCompare);
        const deleteButton = button("删除生成前后对比", async () => { const pending = existingOutput; existingOutput = null; existingCompare.style.display = "none"; existingOutputVideo.removeAttribute("src"); existingOriginalVideo.removeAttribute("src"); if(pending?.filename && pending.type === "temp") await api.fetchApi("/ref2va-director/delete-pending-upscale", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({filename:pending.filename})}).catch(()=>{}); existingStatus.textContent = "已清除本次对比和临时放大结果。"; }, "#7b3650");
        const exportButton = button("导出放大后视频", () => { if (!existingOutput?.filename) { existingStatus.textContent = "请先执行放大。"; return; } const link = document.createElement("a"); link.href = `/view?filename=${encodeURIComponent(existingOutput.filename)}&subfolder=${encodeURIComponent(existingOutput.subfolder || "")}&type=${encodeURIComponent(existingOutput.type || "temp")}`; link.download = existingOutput.filename; document.body.append(link); link.click(); link.remove(); existingStatus.textContent = "已开始导出放大后视频。"; existingStatus.style.color = "#63e3a2"; }, "#32658a");
        const actionRow = document.createElement("div"); actionRow.style.cssText="margin-top:auto;display:grid;grid-template-columns:repeat(2,1fr);gap:8px;"; actionRow.append(deleteButton, exportButton);
        existingEngine.addEventListener("change", updateExistingEngine); const engineField=field("最终放大方式", existingEngine, 0); engineField.style.gridColumn="1/-1"; const scaleField=field("放大倍数", existingScale, 0); const scaleColorRow=document.createElement("div"); scaleColorRow.style.cssText="display:grid;grid-template-columns:1fr 1fr;gap:8px;grid-column:1/-1;"; scaleColorRow.append(scaleField, teColorField); existingParams.append(engineField, scaleColorRow, field("RTX 质量", existingQuality, 0), teGroup, existingRun, existingStatus, existingProgress, actionRow); updateExistingEngine();
        existingBody.append(existingPreviewColumn, existingParams);
        existingUpscale.append(existingSummary, existingBody);
        existingUpscale.addEventListener("toggle",()=>{existingUpscaleOpen=existingUpscale.open;requestAnimationFrame(fitToContent);});
        root.append(systemDrawer, turboDrawer, parameterDrawer, refinementDrawer, workspace, version, timelineDrawer, existingUpscale, mergedDrawer, libraryDrawer);
        // Put the project-wide configuration drawers at the very top of the
        // workbench, before the shot rail and active-shot editor.
        const topAnchor = statusBar.nextElementSibling;
        [systemDrawer, turboDrawer, parameterDrawer, refinementDrawer].forEach((panel) => {
          if (panel && panel !== topAnchor) root.insertBefore(panel, topAnchor);
        });
        // Keep the two panels as adjacent siblings in the requested order:
        // global prompt first, then the active-shot settings toolbar.
        root.insertBefore(globalDrawer, toolbar);
        requestAnimationFrame(fitToContent);
        setTimeout(fitToContent, 50);
      };
      // Immediately synchronize whichever valid mirror won restoration.  This
      // prevents a later workspace serialization from writing an older widget
      // default over the recovered prompts, assets, modes, and durations.
      save("mount_local", false);
      render();
      checkBackendVersion();
      loadAuthoritativeProject();
      const updateLiveState = () => {
        if (liveProgressFill) liveProgressFill.style.width = liveProgressMax ? `${Math.min(100, Math.round(liveProgressValue / liveProgressMax * 100))}%` : "0%";
        if (liveStatusText) { liveStatusText.textContent = liveRunning ? (liveProgressMax ? `正在生成 ${liveProgressValue}/${liveProgressMax}` : "正在生成") : "等待运行"; liveStatusText.style.color = liveRunning ? "#63e3a2" : "#7894aa"; }
        if (liveStopButton) { liveStopButton.disabled = !liveRunning; liveStopButton.style.opacity = liveRunning ? "1" : ".45"; }
      };
      const suppressNativeCanvasPreview = () => {
        // ComfyUI also paints websocket previews through node.imgs.  This
        // director already owns a dedicated DOM preview, so keeping the native
        // canvas copy creates a second large image below the node.
        if (this.imgs != null) this.imgs = [];
        this.imageIndex = null;
        if (this.previewMedia != null) this.previewMedia = null;
        if (this.animatedImages != null) this.animatedImages = [];
      };
      // Native previews are assigned asynchronously after websocket events.
      // Clear them immediately before every canvas draw for this node, so the
      // studio DOM preview remains the only preview without affecting others.
      const priorDrawBackground = this.onDrawBackground;
      this.onDrawBackground = function () {
        suppressNativeCanvasPreview();
        return priorDrawBackground?.apply(this, arguments);
      };
      const showLiveBlob = (blob) => {
        if (!(blob instanceof Blob)) return;
        if (livePreviewUrl) URL.revokeObjectURL(livePreviewUrl);
        livePreviewUrl = URL.createObjectURL(blob);
        if (livePreviewImage) { livePreviewImage.src = livePreviewUrl; livePreviewImage.style.display = "block"; const empty = livePreviewImage.parentElement?.querySelector("span"); if (empty) empty.style.display = "none"; }
        suppressNativeCanvasPreview();
        requestAnimationFrame(suppressNativeCanvasPreview);
        setTimeout(suppressNativeCanvasPreview, 0);
      };
      const refreshAfterExecution = async () => {
        if (!project?.project_id) return;
        try {
          const response = await api.fetchApi(`/ref2va-director/project/${encodeURIComponent(project.project_id)}`);
          const body = response.ok ? await response.json() : null;
          if (body?.ok && body.project?.shots) mergeSaveAcknowledgement(body.project);
        } catch (_) { /* The next delayed refresh retries a late project write. */ }
        if (liveShotId) {
          const completedShot = project.shots.find((item) => String(item.id) === String(liveShotId));
          if (completedShot?.takes?.length) completedShot.status = "generated";
        }
        render();
        if (!liveRunning && saveQueued && !saveInFlight) {
          saveQueued = false;
          persistBackend("after_generation");
        }
      };
      const finishLivePreview = () => {
        if (liveShotId) completedLiveShotIds.add(String(liveShotId));
        liveRunning = false; liveProgressValue = 0; liveProgressMax = 0;
        if (livePreviewUrl) { URL.revokeObjectURL(livePreviewUrl); livePreviewUrl = ""; }
        livePreviewImage = null; liveProgressFill = null; liveStatusText = null; liveStopButton = null;
        refreshAfterExecution();
        setTimeout(refreshAfterExecution, 700);
        setTimeout(refreshAfterExecution, 1800);
      };
      const onExecuting = (event) => {
        const nodeId = event.detail;
        if (String(nodeId) === String(this.id)) {
          liveRunning = true; liveProgressValue = 0; liveProgressMax = 0; liveCompletedPreview = null;
          if (!liveShotId) liveShotId = selected;
          if (liveShotId) completedLiveShotIds.delete(String(liveShotId));
          if (livePreviewUrl) { URL.revokeObjectURL(livePreviewUrl); livePreviewUrl = ""; }
          suppressNativeCanvasPreview();
          updateLiveState(); render();
        }
        else if (nodeId == null && liveRunning) finishLivePreview();
      };
      const onProgress = (event) => {
        const detail = event.detail || {};
        if (!liveRunning && String(detail.node) !== String(this.id)) return;
        liveRunning = true; liveProgressValue = Number(detail.value || 0); liveProgressMax = Number(detail.max || 0); updateLiveState();
      };
      const onPreviewWithMetadata = (event) => {
        const detail = event.detail || {};
        const previewNodeId = detail.displayNodeId || detail.display_node_id || detail.nodeId || detail.node_id;
        if (String(previewNodeId) !== String(this.id) && !liveRunning) return;
        showLiveBlob(detail.blob);
      };
      const onPlainPreview = (event) => { if (liveRunning && !api.serverSupportsFeature?.("supports_preview_metadata")) showLiveBlob(event.detail); };
      const onInterrupted = () => {
        if (!liveRunning) return;
        const interruptedShot = project.shots.find((item) => String(item.id) === String(liveShotId));
        if (interruptedShot && !interruptedShot.takes?.length) interruptedShot.status = "stopped";
        finishLivePreview();
      };
      api.addEventListener("executing", onExecuting);
      api.addEventListener("progress", onProgress);
      api.addEventListener("b_preview_with_metadata", onPreviewWithMetadata);
      api.addEventListener("b_preview", onPlainPreview);
      api.addEventListener("execution_interrupted", onInterrupted);
      let refreshTimer = window.setInterval(() => {
        suppressNativeCanvasPreview();
        if (!project?.project_id || document.hidden) return;
        // Polling is deliberately light; it lets cards receive completed takes
        // during a long ComfyUI queue without relying on private websocket APIs.
        api.fetchApi(`/ref2va-director/project/${encodeURIComponent(project.project_id)}`).then((r) => r.ok ? r.json() : null).then((body) => {
          if (!body?.ok || !body.project?.shots) return;
          let changed = false;
          let newestCompleted = null;
          const activePriority = ["saving", "upscaling", "decoding", "sampling", "preparing_models", "queued"];
          const activeIncomingShot = activePriority.map((status) => body.project.shots.find((shot) => String(shot.status || "") === status)).find(Boolean);
          if (activeIncomingShot && String(liveShotId) !== String(activeIncomingShot.id)) {
            liveShotId = String(activeIncomingShot.id);
            completedLiveShotIds.delete(liveShotId);
            changed = true;
          }
          const incomingById = new Map(body.project.shots.map((item) => [item.id, item]));
          for (const localShot of project.shots) {
            const incomingShot = incomingById.get(localShot.id);
            if (!incomingShot) continue;
            const before = JSON.stringify([localShot.takes || [], localShot.selected_take_id, localShot.status]);
            localShot.takes = incomingShot.takes || [];
            localShot.selected_take_id = incomingShot.selected_take_id || localShot.selected_take_id || null;
            const staleActiveStatus = (completedLiveShotIds.has(String(localShot.id)) || (!liveRunning && !project.active_job_id)) && ["queued", "preparing_models", "sampling", "decoding", "upscaling", "saving"].includes(String(incomingShot.status || ""));
            localShot.status = staleActiveStatus && localShot.takes?.length ? "generated" : (incomingShot.status || localShot.status);
            const after = JSON.stringify([localShot.takes, localShot.selected_take_id, localShot.status]);
            if (before !== after) {
              changed = true;
              const latestTake = localShot.takes?.find?.((item) => item.take_id === localShot.selected_take_id) || localShot.takes?.at?.(-1);
              const filename = latestTake?.files?.final || latestTake?.files?.initial;
              if (filename) newestCompleted = {
                projectId: project.project_id, shotId: localShot.id, takeId: latestTake.take_id,
                filename, shotName: localShot.name,
              };
            }
          }
          if (newestCompleted) {
            liveCompletedPreview = newestCompleted;
            if (livePreviewUrl) { URL.revokeObjectURL(livePreviewUrl); livePreviewUrl = ""; }
          }
          project.revision = Number(body.project.revision || project.revision || 0);
          project.updated_at = body.project.updated_at || project.updated_at;
          project.jobs = body.project.jobs || project.jobs || [];
          project.active_job_id = body.project.active_job_id || null;
          if (changed) { save(); render(); }
        }).catch(() => {});
      }, 2500);
      const onVisibilityChange = () => {
        if (!document.hidden) { checkBackendVersion(); return; }
        window.clearTimeout(saveTimer);
        persistBackend("page_hidden");
      };
      const onBeforeUnload = () => {
        try {
          const payload = new Blob([JSON.stringify({ project, expected_revision: Number(project.revision || 0), reason: "page_unload" })], { type: "application/json" });
          navigator.sendBeacon?.("/ref2va-director/project/save", payload);
        } catch (_) {}
      };
      document.addEventListener("visibilitychange", onVisibilityChange);
      window.addEventListener("beforeunload", onBeforeUnload);
      const priorRemoved = this.onRemoved;
      this.onRemoved = function () {
        window.clearInterval(refreshTimer);
        window.clearTimeout(saveTimer);
        api.removeEventListener("executing", onExecuting);
        api.removeEventListener("progress", onProgress);
        api.removeEventListener("b_preview_with_metadata", onPreviewWithMetadata);
        api.removeEventListener("b_preview", onPlainPreview);
        api.removeEventListener("execution_interrupted", onInterrupted);
        document.removeEventListener("visibilitychange", onVisibilityChange);
        window.removeEventListener("beforeunload", onBeforeUnload);
        if (livePreviewUrl) URL.revokeObjectURL(livePreviewUrl);
        return priorRemoved?.apply(this, arguments);
      };
      const fit = (normalize = false) => {
        const width = Math.max(900, Number(this.size?.[0] || 0));
        if (normalize) fitToContent();
        else fitToContent();
      };
      // Normalize once when the workflow is opened.  This repairs nodes whose
      // old layout feedback loop left a very tall empty canvas area.  Later
      // renders only grow when content genuinely needs more room.
      fit(false); setTimeout(fit, 100); setTimeout(fit, 500);
      return result;
    };
  },
});
