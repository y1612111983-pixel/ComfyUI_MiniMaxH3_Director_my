const fs = require("fs");
const path = require("path");

const frontendPath = path.resolve(__dirname, "../web/ref2va-unified-director-studio-v190.js");
const source = fs.readFileSync(frontendPath, "utf8");
if (!source.includes("localEditSerial !== editSerialAtRequest")) throw new Error("initial backend-load race guard is missing");
if (!source.includes("已保护当前内容不被覆盖")) throw new Error("initial load race has no visible protection state");
const match = source.match(/const mergeSaveAcknowledgement = \(backendProject\) => \{([\s\S]*?)\n      \};\n      const persistBackend/);
if (!match) throw new Error("mergeSaveAcknowledgement implementation not found");

const merge = new Function("project", "backendProject", "completedLiveShotIds", "liveRunning", `${match[1]}\nreturn project;`);
const shot2 = { id: "shot-2", prompt: "镜头2刚输入", takes: [], status: "draft" };
const shot3 = { id: "shot-3", prompt: "", takes: [], status: "draft" };
const project = { revision: 1, shots: [shot2, shot3], jobs: [] };
const identity = project;

// This response represents an earlier request returning after the user has
// already switched to shot 3. It must acknowledge server-owned fields without
// detaching the textarea-bound shot objects.
merge(project, {
  revision: 2,
  updated_at: "test",
  jobs: [],
  active_job_id: null,
  shots: [
    { id: "shot-2", prompt: "older server copy", takes: [{ take_id: "take-1" }], selected_take_id: "take-1", status: "generated" },
    { id: "shot-3", prompt: "", takes: [], selected_take_id: null, status: "draft" },
  ],
}, new Set(), false);
shot3.prompt = "镜头3刚输入";

if (project !== identity) throw new Error("save acknowledgement replaced the project root");
if (project.shots[0] !== shot2 || project.shots[1] !== shot3) throw new Error("save acknowledgement detached shot objects");
if (project.shots[0].prompt !== "镜头2刚输入") throw new Error("shot 2 prompt was overwritten");
if (project.shots[1].prompt !== "镜头3刚输入") throw new Error("shot 3 prompt was lost after switching");
if (project.shots[0].takes[0]?.take_id !== "take-1") throw new Error("server-owned take state was not merged");
if (project.revision !== 2) throw new Error("server revision was not acknowledged");

console.log(JSON.stringify({ ok: true, shot2: project.shots[0].prompt, shot3: project.shots[1].prompt, revision: project.revision, takeMerged: true }));
