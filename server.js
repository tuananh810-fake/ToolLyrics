const express = require("express");
const multer = require("multer");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const { spawn } = require("child_process");

const app = express();
const rootDir = __dirname;
const jobsDir = path.join(rootDir, "jobs");
const uploadsDir = path.join(jobsDir, "uploads");
const outputsDir = path.join(jobsDir, "outputs");
const debugDir = path.join(jobsDir, "debug");
const publicDir = path.join(rootDir, "public");
const staticDir = path.join(rootDir, "static");
const workerScript = path.join(rootDir, "python_worker.py");
const bundledPython = "C:\\Users\\admin\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";

for (const directory of [jobsDir, uploadsDir, outputsDir, debugDir, publicDir, staticDir]) {
  fs.mkdirSync(directory, { recursive: true });
}

function resolvePythonExecutable() {
  const override = process.env.TOOLLYRIC_PYTHON;
  if (override) {
    return override;
  }
  if (fs.existsSync(bundledPython)) {
    return bundledPython;
  }
  return "python";
}

function makeJobId() {
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function makeMessageId() {
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function sanitizeFilename(filename) {
  return path.basename(filename).replace(/[^\w.\-]+/g, "_");
}

function makeOutputFilename(songTitle) {
  const baseName = String(songTitle || "song")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D")
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_")
    .toLowerCase();

  return `${baseName || "song"}_lyrics_pro.json`;
}

function getLyricsMode(filename) {
  const extension = path.extname(filename || "").toLowerCase();
  if (extension === ".txt") {
    return "plain_text";
  }
  return "lrc";
}

function hasAllowedExtension(filename, extensions) {
  return extensions.includes(path.extname(filename || "").toLowerCase());
}

function summarizePythonError(message) {
  const text = (message || "").trim();
  if (!text) {
    return "Python aligner failed with no error details.";
  }

  if (text.includes("stable-ts refine() failed")) {
    return "stable-ts refine() failed internally. The tool will continue with unrefined timings after restart.";
  }

  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  return lines[lines.length - 1] || "Python aligner failed.";
}

const workerState = {
  process: null,
  buffer: "",
  pending: new Map(),
  readyPromise: null,
  resolveReady: null,
  rejectReady: null,
  runtime: null,
  warming: null
};

const jobState = {
  jobs: new Map(),
  queue: [],
  activeJobId: null
};

function rejectAllPending(error) {
  for (const pending of workerState.pending.values()) {
    pending.reject(error);
  }
  workerState.pending.clear();
}

function createHistoryEntry(stage, message, percent, details = {}) {
  return {
    stage,
    message,
    percent,
    details,
    timestamp: new Date().toISOString()
  };
}

function pushJobHistory(job, stage, message, percent, details = {}) {
  const historyEntry = createHistoryEntry(stage, message, percent, details);
  const lastEntry = job.history[job.history.length - 1];
  if (
    lastEntry
    && lastEntry.stage === historyEntry.stage
    && lastEntry.message === historyEntry.message
    && lastEntry.percent === historyEntry.percent
  ) {
    job.history[job.history.length - 1] = historyEntry;
  } else {
    job.history.push(historyEntry);
  }
  if (job.history.length > 20) {
    job.history = job.history.slice(-20);
  }
  job.stage = stage;
  job.message = message;
  job.progressPercent = percent;
  job.details = details;
  job.updatedAt = historyEntry.timestamp;
}

function refreshQueuePositions() {
  for (const [index, queuedJobId] of jobState.queue.entries()) {
    const queuedJob = jobState.jobs.get(queuedJobId);
    if (queuedJob && queuedJob.status === "queued") {
      queuedJob.queuePosition = index + 1;
      queuedJob.updatedAt = new Date().toISOString();
    }
  }
}

function serializeJob(job) {
  return {
    jobId: job.jobId,
    status: job.status,
    stage: job.stage,
    message: job.message,
    progressPercent: job.progressPercent,
    queuePosition: job.queuePosition,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    requestedModel: job.requestedModel,
    resolvedModel: job.resolvedModel,
    requestedDevice: job.requestedDevice,
    deviceName: job.deviceName,
    lyricsMode: job.lyricsMode,
    strategy: job.strategy,
    skipRefine: job.skipRefine,
    useVad: job.useVad,
    debugEnabled: job.debugEnabled,
    songTitle: job.songTitle,
    totalLines: job.totalLines,
    totalWords: job.totalWords,
    outputSizeKb: job.outputSizeKb,
    outputFilename: job.outputFilename,
    downloadUrl: job.downloadUrl,
    error: job.error,
    history: job.history
  };
}

function handleWorkerMessage(message) {
  if (message.event === "ready") {
    workerState.runtime = message.runtime || null;
    if (workerState.resolveReady) {
      workerState.resolveReady(message.runtime || null);
      workerState.resolveReady = null;
      workerState.rejectReady = null;
    }
    return;
  }

  const pending = workerState.pending.get(message.id);
  if (!pending) {
    return;
  }

  if (message.event === "progress") {
    if (typeof pending.onProgress === "function") {
      pending.onProgress(message.progress || {});
    }
    return;
  }

  workerState.pending.delete(message.id);
  if (message.ok) {
    pending.resolve(message);
    return;
  }

  pending.reject(new Error(summarizePythonError(message.error || message.traceback || "Worker request failed.")));
}

function startWorker() {
  if (workerState.process) {
    return workerState.readyPromise;
  }

  const pythonExecutable = resolvePythonExecutable();
  const child = spawn(pythonExecutable, [workerScript], {
    cwd: rootDir,
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8"
    },
    stdio: ["pipe", "pipe", "pipe"]
  });

  workerState.process = child;
  workerState.buffer = "";
  workerState.readyPromise = new Promise((resolve, reject) => {
    workerState.resolveReady = resolve;
    workerState.rejectReady = reject;
  });

  child.stdout.on("data", (chunk) => {
    workerState.buffer += chunk.toString();
    const lines = workerState.buffer.split(/\r?\n/);
    workerState.buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      try {
        handleWorkerMessage(JSON.parse(line));
      } catch (error) {
        console.error("Failed to parse worker message:", line, error);
      }
    }
  });

  child.stderr.on("data", (chunk) => {
    const text = chunk.toString().trim();
    if (text) {
      console.error("[python-worker]", text);
    }
  });

  child.on("error", (error) => {
    if (workerState.rejectReady) {
      workerState.rejectReady(error);
    }
    rejectAllPending(error);
    workerState.process = null;
  });

  child.on("exit", (code, signal) => {
    const error = new Error(`Python worker exited with code ${code ?? "unknown"} signal ${signal ?? "none"}`);
    if (workerState.rejectReady) {
      workerState.rejectReady(error);
    }
    rejectAllPending(error);
    workerState.process = null;
    workerState.readyPromise = null;
    workerState.resolveReady = null;
    workerState.rejectReady = null;
    workerState.runtime = null;
    workerState.warming = null;
  });

  return workerState.readyPromise;
}

async function ensureWorkerReady() {
  const readyPromise = startWorker();
  await readyPromise;
  return workerState.runtime;
}

async function requestWorker(command, payload = {}, onProgress = null) {
  await ensureWorkerReady();

  const child = workerState.process;
  if (!child || !child.stdin.writable) {
    throw new Error("Python worker is not available.");
  }

  const id = makeMessageId();
  const message = JSON.stringify({ id, command, payload });

  return new Promise((resolve, reject) => {
    workerState.pending.set(id, { resolve, reject, onProgress });
    child.stdin.write(`${message}\n`, "utf8", (error) => {
      if (error) {
        workerState.pending.delete(id);
        reject(error);
      }
    });
  });
}

async function warmupRecommendedModel() {
  if (workerState.warming) {
    return workerState.warming;
  }

  workerState.warming = requestWorker("warmup", { model: "recommended", device: "auto" })
    .catch((error) => {
      console.error("Worker warmup failed:", error.message);
    })
    .finally(() => {
      workerState.warming = null;
    });

  return workerState.warming;
}

function createJobRecord({
  jobId,
  audioPath,
  lrcPath,
  outputPath,
  outputFilename,
  debugPath,
  songTitle,
  requestedModel,
  requestedDevice,
  lyricsMode,
  language,
  strategy,
  skipRefine,
  useVad,
  debugEnabled
}) {
  const now = new Date().toISOString();
  return {
    jobId,
    audioPath,
    lrcPath,
    outputPath,
    debugPath,
    songTitle,
    requestedModel,
    requestedDevice,
    resolvedModel: null,
    deviceName: null,
    lyricsMode,
    language,
    strategy,
    skipRefine,
    useVad,
    debugEnabled,
    outputFilename: outputFilename || "lyrics_pro.json",
    outputSizeKb: 0,
    downloadUrl: null,
    totalLines: 0,
    totalWords: 0,
    status: "queued",
    stage: "queued",
    message: "Waiting in queue.",
    progressPercent: 1,
    queuePosition: 1,
    createdAt: now,
    updatedAt: now,
    history: [createHistoryEntry("queued", "Waiting in queue.", 1)],
    details: {},
    error: null
  };
}

async function processNextJob() {
  if (jobState.activeJobId || jobState.queue.length === 0) {
    return;
  }

  const nextJobId = jobState.queue.shift();
  refreshQueuePositions();
  const job = jobState.jobs.get(nextJobId);
  if (!job) {
    setImmediate(processNextJob);
    return;
  }

  jobState.activeJobId = nextJobId;
  job.status = "running";
  job.queuePosition = 0;
  pushJobHistory(job, "starting", "Starting the worker job.", 4);

  try {
    const workerResponse = await requestWorker(
      "align",
      {
        audioPath: job.audioPath,
        lrcPath: job.lrcPath,
        outputPath: job.outputPath,
        song: job.songTitle || undefined,
        language: job.language || undefined,
        model: job.requestedModel,
        device: job.requestedDevice,
        lyricsMode: job.lyricsMode,
        strategy: job.strategy,
        skipRefine: job.skipRefine,
        useVad: job.useVad,
        debugDir: job.debugEnabled ? job.debugPath : undefined
      },
      (progress) => {
        pushJobHistory(
          job,
          progress.stage || "running",
          progress.message || "Processing...",
          Number.isFinite(progress.percent) ? progress.percent : job.progressPercent,
          progress.details || {}
        );
      }
    );

    const payload = workerResponse.payload || JSON.parse(await fsp.readFile(job.outputPath, "utf8"));
    job.status = "completed";
    job.songTitle = payload.song || job.songTitle;
    job.resolvedModel = workerResponse.resolvedModel || job.requestedModel;
    job.deviceName = workerResponse.resolvedDevice || "auto";
    job.totalLines = Array.isArray(payload.data) ? payload.data.length : 0;
    job.totalWords = Array.isArray(payload.data)
      ? payload.data.reduce((sum, line) => sum + (Array.isArray(line.words) ? line.words.length : 0), 0)
      : 0;
    job.outputSizeKb = Math.max(1, Math.round(fs.statSync(job.outputPath).size / 1024));
    job.downloadUrl = `/download/${job.jobId}/${job.outputFilename}`;
    pushJobHistory(job, "completed", "Alignment completed. Export is ready.", 100, {
      totalLines: job.totalLines,
      totalWords: job.totalWords,
      lyricsMode: job.lyricsMode
    });
  } catch (error) {
    job.status = "failed";
    job.error = error.message || "Processing failed.";
    pushJobHistory(job, "failed", job.error, job.progressPercent || 0);
  } finally {
    jobState.activeJobId = null;
    refreshQueuePositions();
    setImmediate(processNextJob);
  }
}

function enqueueJob(job) {
  jobState.jobs.set(job.jobId, job);
  jobState.queue.push(job.jobId);
  refreshQueuePositions();
  setImmediate(processNextJob);
}

const storage = multer.diskStorage({
  destination(req, file, cb) {
    const jobId = req.jobId;
    const jobUploadDir = path.join(uploadsDir, jobId);
    fs.mkdirSync(jobUploadDir, { recursive: true });
    cb(null, jobUploadDir);
  },
  filename(req, file, cb) {
    cb(null, sanitizeFilename(file.originalname));
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 512 * 1024 * 1024
  }
});

app.use((req, res, next) => {
  req.jobId = makeJobId();
  next();
});

app.use("/static", express.static(staticDir));
app.use(express.static(publicDir));

app.get("/", (req, res) => {
  res.sendFile(path.join(publicDir, "index.html"));
});

app.get("/runtime", async (req, res) => {
  try {
    const runtime = await ensureWorkerReady();
    res.json({ ok: true, runtime });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message || "Worker startup failed." });
  }
});

app.post(
  "/process",
  upload.fields([
    { name: "audio", maxCount: 1 },
    { name: "lrc", maxCount: 1 }
  ]),
  async (req, res) => {
    try {
      const body = req.body || {};
      const audioFile = req.files?.audio?.[0];
      const lrcFile = req.files?.lrc?.[0];
      const song = (body.song || "").trim();
      const language = (body.language || "").trim();
      const model = (body.model || "recommended").trim();
      const device = (body.device || "auto").trim();
      const lyricsMode = lrcFile ? getLyricsMode(lrcFile.originalname) : "lrc";
      const strategy = lyricsMode === "plain_text" ? "transcribe" : (body.qualityFallback === "on" ? "auto" : "direct");
      const skipRefine = body.refineTimestamps === "on" ? false : true;
      const useVad = lyricsMode === "plain_text" ? true : body.useVad === "on";
      const debugEnabled = body.debugOutput === "on";

      if (!audioFile) {
        return res.status(400).json({ error: "Please choose an MP3 file." });
      }
      if (!lrcFile) {
        return res.status(400).json({ error: "Please choose an LRC or TXT lyric file." });
      }
      if (!hasAllowedExtension(audioFile.originalname, [".mp3"])) {
        return res.status(400).json({ error: "Audio file must be .mp3." });
      }
      if (!hasAllowedExtension(lrcFile.originalname, [".lrc", ".txt"])) {
        return res.status(400).json({ error: "Lyric file must be .lrc or .txt." });
      }
      if (!["recommended", "large-v3", "medium"].includes(model)) {
        return res.status(400).json({ error: "Invalid model." });
      }
      if (!["auto", "cuda", "cpu"].includes(device)) {
        return res.status(400).json({ error: "Invalid device." });
      }

      const jobId = req.jobId;
      const jobOutputDir = path.join(outputsDir, jobId);
      const jobDebugDir = path.join(debugDir, jobId);
      const songTitle = song || path.parse(audioFile.originalname).name;
      const outputFilename = makeOutputFilename(songTitle);
      const outputPath = path.join(jobOutputDir, outputFilename);
      await fsp.mkdir(jobOutputDir, { recursive: true });
      if (debugEnabled) {
        await fsp.mkdir(jobDebugDir, { recursive: true });
      }

      const job = createJobRecord({
        jobId,
        audioPath: audioFile.path,
        lrcPath: lrcFile.path,
        outputPath,
        outputFilename,
        debugPath: jobDebugDir,
        songTitle,
        requestedModel: model,
        requestedDevice: device,
        lyricsMode,
        language,
        strategy,
        skipRefine,
        useVad,
        debugEnabled
      });

      enqueueJob(job);

      return res.status(202).json({
        ok: true,
        job: serializeJob(job)
      });
    } catch (error) {
      return res.status(500).json({
        error: error.message || "Processing failed."
      });
    }
  }
);

app.get("/jobs/:jobId", (req, res) => {
  const jobId = sanitizeFilename(req.params.jobId);
  const job = jobState.jobs.get(jobId);
  if (!job) {
    return res.status(404).json({ ok: false, error: "Job was not found." });
  }
  return res.json({ ok: true, job: serializeJob(job) });
});

app.get("/download/:jobId/:filename", (req, res) => {
  const jobId = sanitizeFilename(req.params.jobId);
  const filename = sanitizeFilename(req.params.filename);
  const filePath = path.join(outputsDir, jobId, filename);

  if (!fs.existsSync(filePath)) {
    res.status(404).send("Export file was not found.");
    return;
  }

  res.download(filePath, filename);
});

app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    res.status(400).json({ error: `Upload error: ${error.message}` });
    return;
  }

  if (error) {
    res.status(500).json({ error: error.message || "Server error." });
    return;
  }

  next();
});

const port = Number.parseInt(process.env.PORT || "5000", 10);
const host = process.env.HOST || "127.0.0.1";

const server = app.listen(port, host, async () => {
  console.log(`ToolLyric Node server listening at http://${host}:${port}`);
  try {
    const runtime = await ensureWorkerReady();
    console.log(`Python worker ready on ${runtime?.resolvedDevice || "auto"} with recommended model ${runtime?.recommendedModel || "unknown"}`);
    if (process.env.TOOLLYRIC_WARMUP === "1") {
      warmupRecommendedModel().catch(() => {});
    }
  } catch (error) {
    console.error("Python worker failed to start:", error.message);
  }
});

server.on("error", (error) => {
  if (error && error.code === "EADDRINUSE") {
    console.error(`Port ${port} is already in use on ${host}. Set a different PORT and try again.`);
    process.exit(1);
  }
  console.error("Server startup failed:", error);
  process.exit(1);
});
