import { app } from "../../scripts/app.js";

// ComfyUI 1.49 can register node definitions before it imports an extension.
// This late-mount layer deliberately uses nodeCreated *and* setup() so the
// v0.3 editor attaches whether the node is newly added or restored from a
// workflow. It replaces only the hidden JSON widget, never any graph links.
const TYPE = "Ref2VASequenceWorkbenchV03";

const fallback = () => ({
  project: "连续镜头项目", active: "shot-1",
  shots: [
    { id: "shot-1", name: "镜头 1", duration: 5, prompt: "建立镜头：明确人物、场景、光线与动作起点。" },
    { id: "shot-2", name: "镜头 2", duration: 5, prompt: "承接上一镜头尾帧：只描述后续动作、镜头运动或情绪变化。" },
    { id: "shot-3", name: "镜头 3", duration: 5, prompt: "承接上一镜头尾帧：完成动作并收束画面。" },
  ],
});

function normalize(value) {
  const base = fallback();
  if (!value || typeof value !== "object" || !Array.isArray(value.shots) || !value.shots.length) return base;
  const shots = value.shots.map((shot, index) => ({
    id: String(shot.id || `shot-${index + 1}`),
    name: String(shot.name || `镜头 ${index + 1}`),
    duration: Math.max(0.2, Math.min(150, Number(shot.duration) || 5)),
    prompt: String(shot.prompt || ""),
    enabled: shot.enabled !== false,
  }));
  return {
    project: String(value.project || base.project),
    active: shots.some((shot) => shot.id === value.active) ? value.active : shots[0].id,
    shots,
  };
}

const style = (element, css) => { element.style.cssText = css; return element; };

function mount(node) {
  if (!node || node.type !== TYPE || node.ref2vaV03Mounted) return;
  const dataWidget = node.widgets?.find((widget) => widget?.name === "timeline_data");
  if (!dataWidget || typeof node.addDOMWidget !== "function") return;
  node.ref2vaV03Mounted = true;
  // Share the same guard with the original v0.3 registration hook.  Either
  // extension loading order now produces exactly one editor instance.
  node.ref2vaSequenceReady = true;
  dataWidget.type = "hidden";
  dataWidget.computeSize = () => [0, -4];
  let timeline;
  try { timeline = normalize(JSON.parse(dataWidget.value)); } catch { timeline = fallback(); }

  const sync = () => {
    const value = JSON.stringify(timeline);
    dataWidget.value = value;
    node.properties = node.properties || {};
    node.properties.ref2va_sequence_timeline = value;
    dataWidget.callback?.(value);
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
  };
  dataWidget.serializeValue = () => { const value = JSON.stringify(timeline); dataWidget.value = value; return value; };

  // Match the known-good v0.1 widget geometry.  A fixed-height DOM root made
  // ComfyUI reserve an oversized black node body on some canvases.
  const root = style(document.createElement("div"), "box-sizing:border-box;width:100%;min-width:520px;color:#e8f1fc;font:13px system-ui;background:#101824;border:1px solid #314b6d;border-radius:10px;padding:10px;");
  const render = () => {
    root.replaceChildren();
    const header = style(document.createElement("div"), "display:flex;gap:8px;align-items:center;margin-bottom:9px;");
    const project = style(document.createElement("input"), "min-width:0;flex:1;background:#182535;border:1px solid #406084;border-radius:6px;padding:6px 8px;color:#fff;font-weight:650;");
    project.value = timeline.project;
    project.oninput = () => { timeline.project = project.value; sync(); };
    const add = style(document.createElement("button"), "background:#1d7155;border:1px solid #51c795;border-radius:6px;padding:6px 10px;color:#fff;cursor:pointer;");
    add.textContent = "＋ 镜头";
    add.onclick = () => { const n = timeline.shots.length + 1; const shot = { id: `shot-${Date.now()}`, name: `镜头 ${n}`, duration: 5, prompt: "承接上一镜头尾帧：描述下一步动作和镜头变化。" }; timeline.shots.push(shot); timeline.active = shot.id; sync(); render(); };
    header.append(project, add); root.append(header);

    const body = style(document.createElement("div"), "display:grid;grid-template-columns:178px minmax(0,1fr);gap:10px;");
    const list = style(document.createElement("div"), "background:#121f2d;border:1px solid #2f4965;border-radius:7px;padding:6px;max-height:245px;overflow-y:auto;");
    const caption = style(document.createElement("div"), "color:#9fc0e1;font-size:12px;margin:0 2px 6px;"); caption.textContent = `镜头目录 · ${timeline.shots.length} 段`; list.append(caption);
    timeline.shots.forEach((shot, index) => {
      const selected = shot.id === timeline.active;
      const item = style(document.createElement("button"), `display:block;width:100%;margin:0 0 5px;padding:7px;text-align:left;color:#eaf5ff;background:${selected ? "#244568" : "#182535"};border:1px solid ${selected ? "#66b7ff" : "#31485f"};border-radius:6px;cursor:pointer;`);
      const line1 = document.createElement("div"); line1.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:650;"; line1.textContent = `${String(index + 1).padStart(2, "0")} · ${shot.name || `镜头 ${index + 1}`}`;
      const line2 = document.createElement("small"); line2.style.color = selected ? "#bfe5ff" : "#9bb3ce"; line2.textContent = `${shot.duration.toFixed(1)} 秒 · ${index === 0 ? "首镜" : `承接 ${index}`}`;
      item.append(line1, line2); item.onclick = () => { timeline.active = shot.id; sync(); render(); }; list.append(item);
    });
    const active = timeline.shots.find((shot) => shot.id === timeline.active) || timeline.shots[0];
    const editor = style(document.createElement("div"), "min-width:0;");
    const hint = style(document.createElement("div"), "color:#86c9ff;font-size:12px;margin:1px 0 6px;"); hint.textContent = timeline.shots.indexOf(active) === 0 ? "当前镜头 · 首镜头" : "当前镜头 · 自动承接上一段尾帧"; editor.append(hint);
    const fields = style(document.createElement("div"), "display:grid;grid-template-columns:minmax(0,1fr) 78px;gap:7px;margin-bottom:7px;");
    const name = style(document.createElement("input"), "min-width:0;background:#182535;border:1px solid #406084;border-radius:6px;padding:6px;color:#fff;"); name.value = active.name; name.oninput = () => { active.name = name.value; sync(); };
    const duration = style(document.createElement("input"), "min-width:0;background:#182535;border:1px solid #406084;border-radius:6px;padding:6px;color:#fff;"); duration.type = "number"; duration.min = "0.2"; duration.max = "150"; duration.step = "0.1"; duration.value = active.duration; duration.onchange = () => { active.duration = Math.max(.2, Math.min(150, Number(duration.value) || 5)); sync(); render(); };
    fields.append(name, duration); editor.append(fields);
    const prompt = style(document.createElement("textarea"), "box-sizing:border-box;width:100%;height:136px;resize:vertical;background:#0d141e;border:1px solid #406084;border-radius:6px;padding:8px;color:#f3f7ff;line-height:1.45;"); prompt.value = active.prompt; prompt.placeholder = "当前镜头提示词"; prompt.oninput = () => { active.prompt = prompt.value; sync(); }; editor.append(prompt);
    const footer = style(document.createElement("div"), "display:flex;justify-content:space-between;align-items:center;margin-top:7px;color:#8caccc;font-size:11px;"); footer.append(document.createTextNode("v0.3 · 当前镜头的提示词与时长已连接主流程"));
    if (timeline.shots.length > 1) { const remove = style(document.createElement("button"), "background:#3e2430;border:1px solid #a65a6c;border-radius:5px;padding:4px 8px;color:#ffdce4;cursor:pointer;"); remove.textContent = "删除"; remove.onclick = () => { const index = timeline.shots.indexOf(active); timeline.shots.splice(index, 1); timeline.active = timeline.shots[Math.max(0, index - 1)].id; sync(); render(); }; footer.append(remove); }
    editor.append(footer); body.append(list, editor); root.append(body);
  };
  const domWidget = node.addDOMWidget("连续镜头工作台", "custom", root, { serialize: false, hideOnZoom: false });
  domWidget.computeSize = () => [620, 312];
  const fit = () => node.setSize([640, 382]);
  fit();
  // Workflow restore may apply the old saved size after nodeCreated. Reassert
  // the good v0.1 geometry once that restore has completed.
  setTimeout(fit, 0);
  setTimeout(fit, 180);
  render(); sync();
}

app.registerExtension({
  name: "ref2va.sequence-workbench-v03.compat",
  nodeCreated(node) { mount(node); },
  setup() { setTimeout(() => { const graph = app.graph || app.canvas?.graph; for (const node of graph?._nodes || graph?.nodes || []) mount(node); }, 500); },
});
