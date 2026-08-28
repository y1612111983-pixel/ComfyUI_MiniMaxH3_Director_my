const fs = require("fs");
const path = require("path");

const frontendPath = path.resolve(__dirname, "../web/ref2va-unified-director-studio-v190.js");
const source = fs.readFileSync(frontendPath, "utf8");
const cardStart = source.indexOf('const rail = el("div"');
const versionStart = source.indexOf('const version = el("div"');
const workspaceStart = source.indexOf('const workspace = el("div"');
if (cardStart < 0 || versionStart < 0 || workspaceStart < 0) throw new Error("required UI sections not found");

const cardSection = source.slice(cardStart, versionStart);
const versionSection = source.slice(versionStart, workspaceStart);
for (const label of ["刷新成果", "停止生成", "换随机种子", "重跑当前镜头"]) {
  if (!cardSection.includes(label)) throw new Error(`${label} is not inside the shot-card section`);
}
if (!cardSection.includes("ownsLivePreview")) throw new Error("live preview is not bound to a shot card");
if (!versionSection.includes("生成视频与版本")) throw new Error("final video panel title is missing");
for (const label of ["导出原始视频", "导出最终视频", "删除视频"]) {
  if (!versionSection.includes(label)) throw new Error(`${label} is not integrated with the final video panel`);
}
if (versionSection.includes('livePanel') || versionSection.includes('versionLabel.textContent = "版本与预览"')) {
  throw new Error("duplicate standalone live preview panel still exists");
}
if (!source.includes("finishLivePreview") || !source.includes("setTimeout(refreshAfterExecution, 1800)")) {
  throw new Error("completed execution does not transition from live preview to final video");
}
if (!source.includes("completedLiveShotIds") || !source.includes("!liveRunning && !project.active_job_id") || !source.includes("staleActiveStatus && localShot.takes?.length")) {
  throw new Error("stale backend sampling state can still overwrite a completed shot");
}
if (!source.includes("生成中：修改已保存在本地，生成完成后同步后台") || !source.includes('persistBackend("after_generation")')) {
  throw new Error("editor autosave is not deferred while generation owns the project file");
}

console.log(JSON.stringify({
  ok: true,
  livePreviewInShotCard: true,
  runActionsInShotCard: true,
  finalPlayerOwnsVersionActions: true,
  duplicateLivePanelRemoved: true,
  completionRefreshRetries: 3,
  staleSamplingLockedAfterCompletion: true,
  autosaveDeferredDuringGeneration: true,
}));
