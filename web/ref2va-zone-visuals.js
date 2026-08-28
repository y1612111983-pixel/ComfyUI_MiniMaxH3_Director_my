import { app } from "../../scripts/app.js";

// Group bypass visuals are now handled by rgthree's native Fast Groups
// Bypasser.  Keep this extension passive so it cannot add wires or override
// the native purple bypass styling.
// Reference-zone toggles are managed here.  Second sampling is deliberately
// handled by rgthree's native Fast Groups Bypasser, so it gets a genuine
// bypass state instead of a simulated visual state.
const ZONES = [];

// Current ComfyUI builds may restore legacy custom-node widget values without
// their display metadata, rendering the label as UNKNOWN.  Assign labels at
// canvas level after both new-node and workflow-load paths have completed.
const SAMPLING_PANEL_LABELS = [
  "随机种子",
  "生成后控制",
  "主采样调度器",
  "主采样步数",
  "主采样降噪",
  "采样器",
  "主采样 Sigma 加步",
  "主采样 Sigma 阈值",
  "主采样 Sigma 曲线",
  "二次采样模式",
  "二次输出比例",
  "二次输出百万像素",
  "超分后细化次数",
  "采样次数",
  "每遍细化步数",
  "细化降噪",
  "细化调度器",
  "细化 Sigma 加步",
  "细化 Sigma 阈值",
  "细化 Sigma 曲线",
];

const PRIMARY_PANEL_LABELS = ["随机种子", "生成后控制", "主采样调度器", "主采样步数", "主采样降噪", "采样器", "主采样 Sigma 加步", "主采样 Sigma 阈值", "主采样 Sigma 曲线"];
const REFINEMENT_PANEL_LABELS = ["二次采样模式", "H3 Latent 超分模型", "二次输出比例", "二次输出百万像素", "超分后细化次数", "二次采样次数", "细化调度器", "每遍细化步数", "细化降噪", "细化 Sigma 加步", "细化 Sigma 阈值", "细化 Sigma 曲线"];

function labelSamplingPanel(node) {
  const labels = node?.type === "Ref2VAPrimarySamplingPanel"
    ? PRIMARY_PANEL_LABELS
    : node?.type === "Ref2VARefinementPanel"
      ? REFINEMENT_PANEL_LABELS
      : node?.type === "Ref2VASamplingControlPanel"
        ? SAMPLING_PANEL_LABELS
        : null;
  if (!labels) return;
  const widgets = node.widgets || [];
  for (let index = 0; index < labels.length; index += 1) {
    const widget = widgets[index];
    if (!widget) continue;
    widget.label = labels[index];
  }
  node.setSize?.(node.computeSize?.() || node.size);
  // Widget hidden-state changes do not always invalidate a restored workflow
  // node in the current Comfy frontend. Force one canvas redraw so widgets
  // that have just been restored from the H3 page are painted again.
  app.canvas?.setDirty?.(true, true);
  if (node.type === "Ref2VARefinementPanel") {
    attachRefinementTemplateListener(node);
    applyRefinementTemplate(node);
  }
  if (node.type === "Ref2VAAllInOne") applyGenerationModeTemplate(node);
  app.canvas?.setDirty?.(true, true);
}

function applyGenerationModeTemplate(node) {
  const widgets = node.widgets || [];
  const modeWidget = widgets.find((w) => w.name === "generation_mode") || widgets[0];
  if (modeWidget && !modeWidget.ref2vaModeCallbackInstalled) {
    const originalCallback = modeWidget.callback;
    modeWidget.callback = function (value) {
      if (originalCallback) originalCallback.call(this, value);
      applyGenerationModeTemplate(node);
    };
    modeWidget.ref2vaModeCallbackInstalled = true;
  }
  const mode = String(modeWidget?.value || "多参考图/视频/音频（Ref2VA）");
  const isFl2v = mode.includes("FL2V");
  const isT2v = mode.includes("T2V") && !isFl2v;
  const requestedTemplate = isFl2v || isT2v ? "fl2v" : "ref2va";

  // v0.3 keeps every physical input socket and every real cable in place.
  // The backend now selects only the inputs needed by the four modes. The
  // older hide-and-disconnect page switch made the canvas look tidy, but it
  // also made users lose/appear to lose wiring while changing mode.
  node._ref2vaAppliedTemplate = requestedTemplate;
  node.setSize?.(node.computeSize?.() || node.size);
  app.canvas?.setDirty?.(true, true);
  return;

  // The installed Comfy frontend draws every input slot even when
  // `input.hidden` is true.  To make the page really switch, keep a complete
  // slot catalogue and replace the rendered input list. Links that belong to
  // the hidden page are captured and reconnected automatically when returning
  // to that page.
  if (!node._ref2vaAllInputs) {
    node._ref2vaAllInputs = [...(node.inputs || [])];
    node._ref2vaSavedLinks = { ref2va: [], fl2v: [] };
  }
  // Dynamic reference slots can be appended after the third image/video/audio
  // is connected. Add any newly created slot to the catalogue before changing
  // pages so it is restored with the rest of the reference area.
  for (const input of node.inputs || []) {
    if (!node._ref2vaAllInputs.some((known) => known === input || known.name === input.name)) {
      node._ref2vaAllInputs.push(input);
    }
  }
  if (node._ref2vaAppliedTemplate === requestedTemplate) return;

  const isReference = (input) => {
    const name = String(input?.name || "");
    return name.startsWith("ref_image_") || name.startsWith("ref_video_") || name.startsWith("ref_audio_") ||
      name.startsWith("ref_images.") || name.startsWith("ref_videos.") ||
      name.startsWith("ref_video_audios.") || name.startsWith("ref_audios.") || name === "ref_image_size";
  };
  const isFrame = (input) => ["first_frame", "last_frame"].includes(String(input?.name || ""));
  const hideForNext = requestedTemplate === "fl2v" ? isReference : isFrame;
  const restoreKey = requestedTemplate === "fl2v" ? "fl2v" : "ref2va";
  const parkKey = requestedTemplate === "fl2v" ? "ref2va" : "fl2v";
  const currentInputs = node.inputs || [];
  const graph = node.graph || app.graph || app.canvas?.graph;
  const parked = [];

  // Capture and detach only the links for the section about to disappear.
  for (let index = currentInputs.length - 1; index >= 0; index -= 1) {
    const input = currentInputs[index];
    if (!hideForNext(input)) continue;
    const linkId = input.link;
    const link = linkId != null ? graph?.links?.[linkId] : null;
    if (link) parked.push({ originId: link.origin_id, originSlot: link.origin_slot, inputName: input.name });
    if (linkId != null) node.disconnectInput?.(index);
  }
  node._ref2vaSavedLinks[parkKey] = parked;

  // Use the original object references so widget settings and dynamic slots
  // remain intact.  The inactive area truly vanishes from the canvas.
  node.inputs = node._ref2vaAllInputs.filter((input) => !hideForNext(input));

  // Restore the section that belongs to the newly selected page.
  const restoreLinks = node._ref2vaSavedLinks[restoreKey] || [];
  for (const saved of restoreLinks) {
    const slot = node.inputs.findIndex((input) => input.name === saved.inputName);
    const source = graph?.getNodeById?.(saved.originId);
    if (slot >= 0 && source) source.connect?.(saved.originSlot, node, slot);
  }
  node._ref2vaSavedLinks[restoreKey] = [];
  node._ref2vaAppliedTemplate = requestedTemplate;
  node.setSize?.(node.computeSize?.() || node.size);
  app.canvas?.setDirty?.(true, true);
}

function attachGenerationModeListener(node) {
  if (!node || node.type !== "Ref2VAAllInOne" || node._ref2vaGenerationModeListenerAttached) return;
  node._ref2vaGenerationModeListenerAttached = true;
  const previous = node.onWidgetChanged;
  node.onWidgetChanged = function (name, ...args) {
    const result = previous?.call(this, name, ...args);
    if (name === "generation_mode") setTimeout(() => applyGenerationModeTemplate(node), 0);
    return result;
  };
}

// The sampler/output panel owns two real output branches.  Their direct
// consumers are the normal SaveVideo/RTX nodes.  Disabled consumers use
// LiteGraph's NEVER mode (2), which excludes them from a prompt rather than
// merely recolouring them.
function applyVideoBranch(node, widgetName, outputIndex, stateKey) {
  if (!node || node.type !== "Ref2VASamplerVideoOutputPanel") return;
  const branchWidget = node.widgets?.find((widget) => widget?.name === widgetName);
  const branchEnabled = isEnabled(branchWidget?.value);
  const currentGraph = node.graph || graph();
  const output = node.outputs?.[outputIndex];
  for (const linkId of output?.links || []) {
    const link = currentGraph?.links?.[linkId];
    const target = currentGraph?.getNodeById?.(link?.target_id);
    if (!target) continue;
    // The continuity archive is a protected hand-off branch, not a normal
    // final-save consumer. When final output is off it must still run from the
    // first-pass video and write the tail frame for the next shot.
    if (target.type === "Ref2VAStoryboardAutoArchive") {
      if (target[stateKey] !== undefined) {
        target.mode = target[stateKey];
        delete target[stateKey];
      } else if (target.mode === 2) {
        target.mode = 0;
      }
      continue;
    }
    if (!branchEnabled) {
      if (target[stateKey] === undefined) {
        target[stateKey] = target.mode;
      }
      target.mode = 2; // LiteGraph.NEVER: exclude final save/RTX from prompt.
    } else if (target[stateKey] !== undefined) {
      target.mode = target[stateKey];
      delete target[stateKey];
    } else if (target.mode === 2) {
      // The graph may have been saved while this branch was bypassed. Runtime
      // state is not serialized, so on a fresh page load there is no stored
      // previous mode to restore. An enabled branch must still revive its
      // direct SaveVideo/RTX consumer instead of leaving it grey and skipped.
      target.mode = 0;
    }
  }
  app.canvas?.setDirty?.(true, true);
}

function applyInitialVideoBranch(node) {
  applyVideoBranch(node, "enable_initial_video", 0, "_ref2vaInitialVideoModeBeforeMute");
}

function applyFinalVideoBranch(node) {
  applyVideoBranch(node, "enable_final_video", 1, "_ref2vaFinalVideoModeBeforeMute");
}

// Keep the two sampling-control boards in the *same real bypass state* as
// their dependent output branches.  This deliberately uses LiteGraph NEVER
// mode (2), just like the grey SaveVideo/RTX nodes, rather than recolouring
// the boards.  The dependency rules are important:
// - primary settings remain active whenever either video can be produced;
// - refinement settings are needed only for the final-video branch.
// Thus "initial off + final on" still leaves the first-pass board enabled,
// because the final video genuinely depends on it.
function findLinkedInputSource(node, inputName) {
  const currentGraph = node?.graph || graph();
  const input = node?.inputs?.find((item) => item?.name === inputName);
  const link = input?.link != null ? currentGraph?.links?.[input.link] : null;
  return link ? currentGraph?.getNodeById?.(link.origin_id) : null;
}

function applyLinkedNodeBypass(source, shouldBypass, stateKey) {
  if (!source) return;
  if (shouldBypass) {
    if (source[stateKey] === undefined) source[stateKey] = source.mode;
    source.mode = 2; // LiteGraph.NEVER: the node is visibly grey and skipped.
  } else if (source[stateKey] !== undefined) {
    source.mode = source[stateKey];
    delete source[stateKey];
  } else if (source.mode === 2) {
    // A workflow could be saved while this board was bypassed.  Restore an
    // enabled board after reload rather than leaving it grey forever.
    source.mode = 0;
  }
}

function applySamplingControlBranches(node) {
  if (!node || node.type !== "Ref2VASamplerVideoOutputPanel") return;
  const initialEnabled = isEnabled(node.widgets?.find((widget) => widget?.name === "enable_initial_video")?.value);
  const finalEnabled = isEnabled(node.widgets?.find((widget) => widget?.name === "enable_final_video")?.value);
  const primaryPanel = findLinkedInputSource(node, "primary_settings");
  const refinementPanel = findLinkedInputSource(node, "refinement_settings");

  // No output selected means the combined panel returns before sampling, so
  // both setting nodes can be truthfully skipped.  If final is enabled the
  // primary settings must remain live even when the initial preview is off.
  applyLinkedNodeBypass(primaryPanel, !initialEnabled && !finalEnabled, "_ref2vaPrimaryModeBeforeMute");
  applyLinkedNodeBypass(refinementPanel, !finalEnabled, "_ref2vaRefinementModeBeforeMute");
  app.canvas?.setDirty?.(true, true);
}

function attachVideoOutputPanelListener(node) {
  if (!node || node.type !== "Ref2VASamplerVideoOutputPanel" || node._ref2vaVideoOutputListenerAttached) return;
  node._ref2vaVideoOutputListenerAttached = true;
  for (const widget of (node.widgets || []).filter((item) => item?.name === "enable_initial_video" || item?.name === "enable_final_video")) {
    const previousCallback = widget.callback;
    widget.callback = function (...args) {
      const result = previousCallback?.apply(this, args);
      setTimeout(() => { applyInitialVideoBranch(node); applyFinalVideoBranch(node); applySamplingControlBranches(node); }, 0);
      return result;
    };
  }
  const previousWidgetChanged = node.onWidgetChanged;
  node.onWidgetChanged = function (name, ...args) {
    const result = previousWidgetChanged?.call(this, name, ...args);
    if (name === "enable_initial_video" || name === "enable_final_video") {
      setTimeout(() => { applyInitialVideoBranch(node); applyFinalVideoBranch(node); applySamplingControlBranches(node); }, 0);
    }
    return result;
  };
  setTimeout(() => { applyInitialVideoBranch(node); applyFinalVideoBranch(node); applySamplingControlBranches(node); }, 0);
}

function applyRefinementTemplate(node) {
  const widgets = node.widgets || [];
  // Widget array positions are not stable after a legacy ComfyUI widget is
  // hidden: the frontend can insert/remove proxy widgets.  Always resolve by
  // the backend input name, never by widgets[0]/widgets[1] positions.
  const modeWidget = widgets.find((widget) => widget?.name === "second_sampling_mode");
  // The mode is a normal two-choice COMBO.  Keep one canonical value owned by
  // this panel rather than reading stale values left by a previously hidden
  // widget.  This is what makes H3 -> same-resolution -> H3 repeatable.
  if (modeWidget && !modeWidget.ref2vaTemplateCallbackInstalled) {
    const originalCallback = modeWidget.callback;
    modeWidget.callback = function (value) {
      if (originalCallback) originalCallback.call(this, value);
      node._ref2vaCurrentMode = value;
      node.widgets_values = node.widgets_values || [];
      node.widgets_values[0] = value;
      setTimeout(() => applyRefinementTemplate(node), 0);
    };
    modeWidget.ref2vaTemplateCallbackInstalled = true;
  }
  if (modeWidget && !modeWidget.ref2vaValueWriteListenerInstalled) {
    // On this legacy canvas, clicking the drop-down can repaint its label
    // without calling `callback` or `onWidgetChanged`.  Observe the actual
    // value write as well, and make that value the only panel state.  This
    // handles both menu choices repeatedly, including same -> H3.
    const ownDescriptor = Object.getOwnPropertyDescriptor(modeWidget, "value");
    if (!ownDescriptor || ownDescriptor.configurable !== false) {
      let currentValue = modeWidget.value;
      Object.defineProperty(modeWidget, "value", {
        configurable: true,
        enumerable: ownDescriptor?.enumerable ?? true,
        get() { return currentValue; },
        set(value) {
          currentValue = value;
          node._ref2vaCurrentMode = value;
          node.widgets_values = node.widgets_values || [];
          node.widgets_values[0] = value;
          setTimeout(() => applyRefinementTemplate(node), 0);
        },
      });
    }
    modeWidget.ref2vaValueWriteListenerInstalled = true;
  }
  const comboValues = modeWidget?.options?.values || modeWidget?.options?.options || [];
  // The visible COMBO value is the source of truth. Its callback is not
  // consistently emitted by this legacy canvas, so a previous callback value
  // must never override the user's latest menu selection.
  let rawMode = modeWidget?.value ?? node._ref2vaCurrentMode ?? node.widgets_values?.[0] ?? "同分辨率细化";
  if (typeof rawMode === "number" && Array.isArray(comboValues) && comboValues[rawMode] != null) rawMode = comboValues[rawMode];
  const modeText = String(rawMode);
  const latentMode = modeText.includes("Latent") || modeText.includes("latent") || modeText.includes("超分");
  // 0 is the mode selector. This is a real two-page control panel:
  // - 同分辨率细化: only the sampling/refinement controls are rendered.
  // - H3 Latent 超分: the H3 model, target resolution and upscale-count
  //   controls are rendered before those same refinement controls.
  // The backend already follows the selected mode; hiding the inactive
  // widgets here prevents unrelated H3 settings from looking active while
  // the workflow is in same-resolution mode.
  const latentOnlyNames = new Set([
    "latent_upscale_model",
    "second_aspect_ratio",
    "second_megapixels",
    "upscale_passes",
  ]);
  // Keep the original list once.  Do not rebuild node.widgets while changing
  // pages: this ComfyUI canvas retains draw metadata for each widget object,
  // and replacing the array is exactly what caused the H3 label to switch
  // without bringing the H3-only controls back.
  if (!node._ref2vaAllTemplateWidgets) {
    node._ref2vaAllTemplateWidgets = [...widgets];
  }
  const templateWidgets = node._ref2vaAllTemplateWidgets;
  for (const widget of templateWidgets) {
    if (!widget || widget === modeWidget) continue;
    // Use the same converted-widget pattern used by the installed ComfyUI
    // extensions.  It hides a widget without deleting it from the canvas;
    // restoring its original type and size is then reliable on every switch.
    if (widget._ref2vaOriginalType === undefined) widget._ref2vaOriginalType = widget.type;
    // These four named inputs belong only to the H3 Latent template.
    const shouldHide = !latentMode && latentOnlyNames.has(widget.name);
    if (shouldHide) {
      widget.type = "converted-widget:ref2va-template";
      widget.computeSize = () => [0, -4];
    } else {
      widget.type = widget._ref2vaOriginalType;
      // These are ordinary COMBO/NUMBER controls.  A hidden page leaves a
      // zero-height closure and a shared last_y on the legacy canvas.  Restore
      // the native layout path instead of reusing that closure; otherwise the
      // H3 controls exist but remain invisible after the first switch.
      widget.computeSize = undefined;
      delete widget.y;
      delete widget.last_y;
    }
  }
  // Let LiteGraph calculate the exact content height after the widgets have
  // changed.  Fixed heights made the panel leave a purple tail and the 180ms
  // reconciliation would undo a manual resize.  This keeps the existing
  // width, but lets each template end exactly below its final visible row.
  const naturalSize = node.computeSize?.() || node.size || [340, 180];
  const width = Math.max(Number(node.size?.[0]) || 340, Number(naturalSize[0]) || 340);
  const height = Number(naturalSize[1]) || Number(node.size?.[1]) || 180;
  node.size = [width, height];
  node.setSize?.([width, height]);
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
}

function attachRefinementTemplateListener(node) {
  if (!node || node.type !== "Ref2VARefinementPanel" || node._ref2vaRefinementTemplateListenerAttached) return;
  node._ref2vaRefinementTemplateListenerAttached = true;
  const previous = node.onWidgetChanged;
  node.onWidgetChanged = function (name, ...args) {
    const result = previous?.call(this, name, ...args);
    if (name === "second_sampling_mode") {
      // This is intentionally deferred: at this point the new value is
      // committed and the relevant page can be rendered accurately.
      setTimeout(() => applyRefinementTemplate(node), 0);
    }
    return result;
  };
}

function applyRefinementBypassVisual(node) {
  // The switch is part of the refinement panel and is always left clickable.
}

function graph() {
  return app.graph || app.canvas?.graph;
}

function isEnabled(value) {
  if (typeof value === "string") {
    return !["", "0", "false", "no", "off", "关闭", "跳过", "已关闭：绕过二次采样"].includes(value.trim().toLowerCase());
  }
  return Boolean(value);
}

function applyZone(zone) {
  const currentGraph = graph();
  if (!currentGraph) return;
  const switchNode = currentGraph.getNodeById?.(zone.controllerId);
  const widget = switchNode?.widgets?.find((item) => item.name === zone.widgetName);
  const enabled = isEnabled(widget?.value);

  const groups = currentGraph._groups || currentGraph.groups || [];
  const zoneGroup = groups.find((group) => String(group.title || "").startsWith(zone.groupPrefix));
  if (zoneGroup) {
    if (!zoneGroup._ref2vaBaseTitle) zoneGroup._ref2vaBaseTitle = zoneGroup.title;
    zoneGroup.color = zone.color;
    zoneGroup.title = zoneGroup._ref2vaBaseTitle.replace(/（已关闭）$/, "");
    zoneGroup.recomputeInsideNodes?.();
  }

  // Use ComfyUI/LiteGraph's real bypass mode (4), exactly like rgthree's
  // Fast Groups Bypasser.  This produces the purple overlay on all nodes and
  // noodles inside a disabled group instead of merely recolouring them.
  const members = zoneGroup?._nodes?.length
    ? zoneGroup._nodes
    : zone.nodeIds.map((nodeId) => currentGraph.getNodeById?.(nodeId));
  for (const node of members) {
    if (!node) continue;
    if (!enabled) {
      if (node._ref2vaModeBeforeIgnore === undefined) {
        node._ref2vaModeBeforeIgnore = node.mode;
      }
      node.mode = 4;
    } else if (node._ref2vaModeBeforeIgnore !== undefined) {
      node.mode = node._ref2vaModeBeforeIgnore;
      delete node._ref2vaModeBeforeIgnore;
    }
  }
}

function refreshZones() {
  for (const zone of ZONES) applyZone(zone);
  app.canvas?.setDirty?.(true, true);
}

// Existing nodes are restored from a workflow through `loadedGraphNode`, not
// `nodeCreated`.  Attach the listener in both paths so toggles keep working
// after opening a saved workflow.
function attachSwitchListener(node) {
  const zone = ZONES.find((item) => item.controllerId === node?.id);
  if (!zone || node._ref2vaVisualListenerAttached) return;
  node._ref2vaVisualListenerAttached = true;

  const widget = node.widgets?.find((item) => item.name === zone.widgetName);
  if (widget) {
    const previousCallback = widget.callback;
    widget.callback = function (...args) {
      const result = previousCallback?.apply(this, args);
      setTimeout(refreshZones, 0);
      return result;
    };
  }

  const previousWidgetChanged = node.onWidgetChanged;
  node.onWidgetChanged = function (name, ...args) {
    const result = previousWidgetChanged?.call(this, name, ...args);
    if (ZONES.some((item) => item.widgetName === name)) setTimeout(refreshZones, 0);
    return result;
  };
}

function attachAllSwitchListeners() {
  const currentGraph = graph();
  const nodes = currentGraph?._nodes || currentGraph?.nodes || [];
  for (const node of nodes) attachSwitchListener(node);
}

function zoneStateSignature() {
  const currentGraph = graph();
  if (!currentGraph) return "";
  return ZONES.map((zone) => {
    const node = currentGraph.getNodeById?.(zone.controllerId);
    const widget = node?.widgets?.find((item) => item.name === zone.widgetName);
    const groups = currentGraph._groups || currentGraph.groups || [];
    const group = groups.find((item) => String(item.title || "").startsWith(zone.groupPrefix));
    group?.recomputeInsideNodes?.();
    const memberIds = (group?._nodes || []).map((item) => item.id).join(",");
    return `${zone.widgetName}:${isEnabled(widget?.value)}:${memberIds}`;
  }).join("|");
}

app.registerExtension({
  name: "Ref2VA.ZoneVisualSwitches",
  nodeCreated(node) {
    setTimeout(() => labelSamplingPanel(node), 0);
    attachGenerationModeListener(node);
    setTimeout(() => applyGenerationModeTemplate(node), 0);
    setTimeout(() => applyRefinementBypassVisual(node), 0);
    attachSwitchListener(node);
    attachVideoOutputPanelListener(node);
    setTimeout(refreshZones, 0);
  },
  loadedGraphNode(node) {
    setTimeout(() => labelSamplingPanel(node), 0);
    attachGenerationModeListener(node);
    setTimeout(() => applyGenerationModeTemplate(node), 0);
    setTimeout(() => applyRefinementBypassVisual(node), 0);
    attachSwitchListener(node);
    attachVideoOutputPanelListener(node);
    setTimeout(refreshZones, 0);
  },
  afterConfigureGraph() {
    // Some ComfyUI builds restore widgets after loadedGraphNode; one delayed
    // pass guarantees the Boolean widgets have their final values.
    setTimeout(() => {
      attachAllSwitchListeners();
      const currentGraph = graph();
      for (const node of currentGraph?._nodes || currentGraph?.nodes || []) {
        labelSamplingPanel(node);
        attachGenerationModeListener(node);
        if (node.type === "Ref2VAAllInOne") applyGenerationModeTemplate(node);
        applyRefinementBypassVisual(node);
        attachVideoOutputPanelListener(node);
        applyInitialVideoBranch(node);
        applyFinalVideoBranch(node);
        applySamplingControlBranches(node);
      }
      refreshZones();
    }, 150);
  },
  setup() {
    setTimeout(() => {
      attachAllSwitchListeners();
      refreshZones();
    }, 300);

    // Some ComfyUI frontends do not emit onWidgetChanged for legacy Boolean
    // widgets restored from a workflow.  Polling the three tiny controls keeps
    // the canvas feedback reliable without affecting generation performance.
    let lastState = "";
    setInterval(() => {
      attachAllSwitchListeners();
      const currentGraph = graph();
      for (const node of currentGraph?._nodes || currentGraph?.nodes || []) {
        if (node.type === "Ref2VAAllInOne") {
          attachGenerationModeListener(node);
          applyGenerationModeTemplate(node);
        }
        if (node.type === "Ref2VARefinementPanel") {
          attachRefinementTemplateListener(node);
          // A short state reconciliation is deliberately kept here because
          // some ComfyUI widget implementations do not emit a callback when
          // switching a restored combo widget. It makes the two pages
          // reversible rather than a one-way hide operation.
          applyRefinementTemplate(node);
        }
        if (node.type === "Ref2VASamplerVideoOutputPanel") {
          attachVideoOutputPanelListener(node);
          applyInitialVideoBranch(node);
          applyFinalVideoBranch(node);
          applySamplingControlBranches(node);
        }
      }
      const state = zoneStateSignature();
      if (state && state !== lastState) {
        lastState = state;
        refreshZones();
      }
    }, 180);
  },
});


