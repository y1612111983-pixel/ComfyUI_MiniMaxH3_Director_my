import { app } from "../../scripts/app.js";

// Ref2VA's own compact v0.1 shot-queue editor. It controls one real current
// shot in the existing Ref2VA / FL2V graph; it intentionally does not pretend
// to batch-run or auto-chain clips before that executor exists.
const TYPE = "Ref2VASequenceWorkbenchV03";

const fallback = () => ({
  project: "连续镜头项目",
  active: "shot-1",
  shots: [
    { id: "shot-1", name: "镜头 1", duration: 5, enabled: true, prompt: "建立镜头：明确人物、场景、光线与动作起点。" },
    { id: "shot-2", name: "镜头 2", duration: 5, enabled: true, prompt: "承接上一镜头尾帧：只描述后续动作、镜头运动或情绪变化。" },
    { id: "shot-3", name: "镜头 3", duration: 5, enabled: true, prompt: "承接上一镜头尾帧：完成动作并收束画面。" },
  ],
});

function tidy(data) {
  const base = fallback();
  if (!data || typeof data !== "object" || !Array.isArray(data.shots) || !data.shots.length) return base;
  const shots = data.shots.map((shot, index) => ({
    id: String(shot.id || `shot-${index + 1}`),
    name: String(shot.name || `镜头 ${index + 1}`),
    duration: Math.max(0.2, Math.min(150, Number(shot.duration) || 5)),
    // Retain this legacy field for saved v0.1 data compatibility. It has no UI
    // control because v0.1 only runs the selected current shot.
    enabled: shot.enabled !== false,
    prompt: String(shot.prompt || ""),
  }));
  const active = shots.some((shot) => shot.id === data.active) ? data.active : shots[0].id;
  return { project: String(data.project || base.project), active, shots };
}

function setStyle(element, style) {
  element.style.cssText = style;
  return element;
}

function button(text, style) {
  const element = document.createElement("button");
  element.textContent = text;
  return setStyle(element, style);
}

app.registerExtension({
  name: "ref2va.sequence-workbench-v01",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TYPE) return;
    const original = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const result = original?.apply(this, arguments);
      const dataWidget = this.widgets?.find((item) => item.name === "timeline_data");
      if (!dataWidget || this.ref2vaSequenceReady || this.ref2vaV03Mounted) return result;
      this.ref2vaSequenceReady = true;
      this.ref2vaV03Mounted = true;
      dataWidget.type = "hidden";
      dataWidget.computeSize = () => [0, -4];

      let timeline;
      try { timeline = tidy(JSON.parse(dataWidget.value)); } catch { timeline = fallback(); }

      const root = document.createElement("div");
      root.className = "ref2va-sequence-v03";
      // This panel has a deliberate fixed canvas footprint.  v0.1 allowed a
      // vertically-resizable editor inside a DOM widget, which could make the
      // LiteGraph node grow into a tall empty column after reload/zoom.
      setStyle(root, "box-sizing:border-box;width:590px;height:266px;overflow:hidden;color:#e8f1fc;font:13px system-ui;background:#101824;border:1px solid #314b6d;border-radius:10px;padding:9px;");

      // LiteGraph keeps widget values for both workflow saves and prompt
      // submission.  The DOM editor is only a view, so make the hidden native
      // widget serialize the live timeline every time.  This prevents a
      // visible "镜头 2" selection from accidentally submitting stale
      // "镜头 1" data.
      const serializeTimeline = () => JSON.stringify(timeline);
      dataWidget.serializeValue = function (...args) {
        const value = serializeTimeline();
        dataWidget.value = value;
        return value;
      };
      const sync = () => {
        const value = serializeTimeline();
        dataWidget.value = value;
        // Keep an explicit copy on the node too.  This is a safe fallback for
        // ComfyUI builds that serialize properties before widget values.
        this.properties = this.properties || {};
        this.properties.ref2va_sequence_timeline = value;
        dataWidget.callback?.(value);
        this.graph?.setDirtyCanvas?.(true, true);
        app.canvas?.setDirty?.(true, true);
      };
      const activeShot = () => timeline.shots.find((shot) => shot.id === timeline.active) || timeline.shots[0];
      const safeName = (shot, index) => shot.name.trim() || `镜头 ${index + 1}`;

      const render = () => {
        root.replaceChildren();

        const header = setStyle(document.createElement("div"), "display:flex;gap:8px;align-items:center;margin-bottom:7px;");
        const title = setStyle(document.createElement("input"), "flex:1;min-width:0;background:#182535;border:1px solid #406084;color:#fff;border-radius:6px;padding:6px 8px;font-weight:650;");
        title.value = timeline.project;
        title.placeholder = "项目名称";
        title.oninput = () => { timeline.project = title.value; sync(); };
        const add = button("＋ 添加镜头", "flex:0 0 auto;background:#1d6e54;border:1px solid #4cc58d;color:#fff;border-radius:6px;padding:6px 10px;cursor:pointer;");
        add.onclick = () => {
          const number = timeline.shots.length + 1;
          const shot = { id: `shot-${Date.now()}`, name: `镜头 ${number}`, duration: 5, enabled: true, prompt: "承接上一镜头尾帧：描述下一步动作和镜头变化。" };
          timeline.shots.push(shot);
          timeline.active = shot.id;
          sync();
          render();
        };
        header.append(title, add);
        root.append(header);

        const body = setStyle(document.createElement("div"), "display:grid;grid-template-columns:166px minmax(0,1fr);gap:8px;height:213px;");
        const directory = setStyle(document.createElement("div"), "box-sizing:border-box;background:#121f2d;border:1px solid #2f4965;border-radius:7px;padding:6px;height:213px;overflow-y:auto;");
        const directoryTitle = setStyle(document.createElement("div"), "display:flex;justify-content:space-between;align-items:center;color:#9fc0e1;font-size:12px;margin:1px 2px 6px;");
        directoryTitle.append(document.createTextNode("镜头目录"), document.createTextNode(`${timeline.shots.length} 段`));
        directory.append(directoryTitle);

        timeline.shots.forEach((shot, index) => {
          const selected = shot.id === timeline.active;
          const item = button("", `display:block;width:100%;text-align:left;margin:0 0 4px;padding:6px;border-radius:6px;cursor:pointer;color:#eaf5ff;background:${selected ? "#244568" : "#182535"};border:1px solid ${selected ? "#66b7ff" : "#31485f"};`);
          const prefix = index === 0 ? "首镜" : `承接 ${index}`;
          item.innerHTML = `<div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:650">${String(index + 1).padStart(2, "0")} · ${safeName(shot, index)}</div><small style="color:${selected ? "#bfe5ff" : "#9bb3ce"}">${shot.duration.toFixed(1)} 秒 · ${prefix}</small>`;
          item.onclick = () => { timeline.active = shot.id; sync(); render(); };
          directory.append(item);
        });

        const shot = activeShot();
        const editor = setStyle(document.createElement("div"), "min-width:0;");
        const marker = setStyle(document.createElement("div"), "display:flex;justify-content:space-between;align-items:center;margin:1px 0 5px;color:#86c9ff;font-size:12px;");
        const activeIndex = timeline.shots.indexOf(shot);
        marker.append(document.createTextNode(activeIndex === 0 ? "当前镜头 · 首镜头" : `当前镜头 · 承接镜头 ${activeIndex}`), document.createTextNode("运行时只输出这一段"));
        editor.append(marker);

        const fields = setStyle(document.createElement("div"), "display:grid;grid-template-columns:minmax(0,1fr) 90px;gap:7px;margin-bottom:7px;");
        const name = setStyle(document.createElement("input"), "min-width:0;background:#182535;border:1px solid #406084;color:#fff;border-radius:6px;padding:6px;");
        name.value = shot.name;
        name.placeholder = "镜头名称";
        name.oninput = () => { shot.name = name.value; sync(); };
        const duration = setStyle(document.createElement("input"), "min-width:0;background:#182535;border:1px solid #406084;color:#fff;border-radius:6px;padding:6px;");
        duration.type = "number";
        duration.min = "0.2";
        duration.max = "150";
        duration.step = "0.1";
        duration.value = shot.duration;
        duration.title = "时长（秒）";
        duration.onchange = () => { shot.duration = Math.max(.2, Math.min(150, Number(duration.value) || 5)); sync(); render(); };
        fields.append(name, duration);
        editor.append(fields);

        const prompt = setStyle(document.createElement("textarea"), "box-sizing:border-box;width:100%;height:104px;resize:none;overflow-y:auto;background:#0d141e;border:1px solid #406084;color:#f3f7ff;border-radius:6px;padding:8px;line-height:1.4;");
        prompt.value = shot.prompt;
        prompt.placeholder = "当前镜头提示词";
        prompt.oninput = () => { shot.prompt = prompt.value; sync(); };
        editor.append(prompt);

        const footer = setStyle(document.createElement("div"), "display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:7px;color:#8caccc;font-size:11px;");
        footer.append(document.createTextNode("v0.3：单镜头运行 · 连续性尾帧由归档分支保护"));
        if (timeline.shots.length > 1) {
          const remove = button("删除", "flex:0 0 auto;background:#3e2430;border:1px solid #a65a6c;color:#ffdce4;border-radius:5px;padding:4px 8px;cursor:pointer;");
          remove.onclick = () => {
            const index = timeline.shots.indexOf(shot);
            timeline.shots.splice(index, 1);
            timeline.active = timeline.shots[Math.max(0, index - 1)].id;
            sync();
            render();
          };
          footer.append(remove);
        }
        editor.append(footer);
        body.append(directory, editor);
        root.append(body);
      };

      const widget = this.addDOMWidget("镜头队列", "custom", root, { serialize: false, hideOnZoom: false });
      widget.computeSize = () => [590, 266];
      // A brand-new v0.3 node ignores the oversized persisted dimensions of
      // v0.1.  Keep it compact even after workflow reload.
      this.resizable = false;
      this.setSize([610, 316]);
      render();
      sync();
      return result;
    };
  },
});
