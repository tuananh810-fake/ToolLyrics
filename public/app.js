const uploadForm = document.getElementById("uploadForm");
const noticeStack = document.getElementById("noticeStack");
const progressPanel = document.getElementById("progressPanel");
const exportPanel = document.getElementById("exportPanel");
const submitButton = document.getElementById("submitButton");
const actionHint = document.getElementById("actionHint");
const resetButton = document.getElementById("resetButton");
const downloadLink = document.getElementById("downloadLink");
const runtimeHint = document.getElementById("runtimeHint");
const audioInput = uploadForm.elements.namedItem("audio");
const lrcInput = uploadForm.elements.namedItem("lrc");

let currentJobId = null;
let pollTimer = null;

function applyFastDefaults() {
  for (const name of ["qualityFallback", "refineTimestamps", "useVad", "debugOutput"]) {
    const field = uploadForm.elements.namedItem(name);
    if (field) {
      field.checked = false;
    }
  }
}

function showNotice(message) {
  noticeStack.hidden = false;
  noticeStack.innerHTML = `<div class="notice notice-error">${message}</div>`;
}

function clearNotice() {
  noticeStack.hidden = true;
  noticeStack.innerHTML = "";
}

function stopPolling() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function resetProgressPanel() {
  document.getElementById("progressTitle").textContent = "No active job";
  document.getElementById("jobStatusChip").textContent = "idle";
  document.getElementById("jobQueueChip").textContent = "queue: -";
  document.getElementById("jobId").textContent = "-";
  document.getElementById("jobStage").textContent = "-";
  document.getElementById("jobPercent").textContent = "0%";
  document.getElementById("jobMessage").textContent = "No active job yet.";
  document.getElementById("progressBar").style.width = "0%";
  document.getElementById("historyList").innerHTML = "";
  progressPanel.hidden = true;
}

function resetExportPanel() {
  document.getElementById("exportTitle").textContent = "Export is not ready yet";
  document.getElementById("songTitle").textContent = "-";
  document.getElementById("modelName").textContent = "-";
  document.getElementById("deviceName").textContent = "-";
  document.getElementById("totalLines").textContent = "0";
  document.getElementById("totalWords").textContent = "0";
  document.getElementById("outputSize").textContent = "0 KB";
  downloadLink.href = "#";
  downloadLink.setAttribute("aria-disabled", "true");
  exportPanel.hidden = true;
}

function setBusyState(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Uploading..." : "Confirm upload and process";
  actionHint.textContent = isBusy
    ? "Uploading files and creating a background job."
    : "Mac dinh: RTX GPU neu co, model medium, direct align, skip refine, no VAD, no debug.";
}

function renderHistory(history) {
  const list = document.getElementById("historyList");
  if (!history.length) {
    list.innerHTML = "<li class=\"history-item\">No progress events yet.</li>";
    return;
  }
  list.innerHTML = history
    .slice()
    .reverse()
    .map((entry) => {
      const percent = Number.isFinite(entry.percent) ? `${entry.percent}%` : "-";
      return `<li class="history-item"><span class="history-meta">${percent}</span><span>${entry.message}</span></li>`;
    })
    .join("");
}

function renderProgress(job) {
  progressPanel.hidden = false;
  document.getElementById("progressTitle").textContent = job.songTitle || "Background job";
  document.getElementById("jobStatusChip").textContent = job.status;
  document.getElementById("jobQueueChip").textContent = job.queuePosition > 0 ? `queue: ${job.queuePosition}` : "queue: running";
  document.getElementById("jobId").textContent = job.jobId;
  document.getElementById("jobStage").textContent = job.stage || "-";
  document.getElementById("jobPercent").textContent = `${job.progressPercent || 0}%`;
  document.getElementById("jobMessage").textContent = job.message || "Processing...";
  document.getElementById("progressBar").style.width = `${job.progressPercent || 0}%`;
  renderHistory(job.history || []);
}

function renderExport(job) {
  exportPanel.hidden = false;
  document.getElementById("exportTitle").textContent = job.status === "completed"
    ? "Export file is ready"
    : "Export is not ready yet";
  document.getElementById("songTitle").textContent = job.songTitle || "-";
  document.getElementById("modelName").textContent = job.resolvedModel || job.requestedModel || "-";
  document.getElementById("deviceName").textContent = job.deviceName || "-";
  document.getElementById("totalLines").textContent = String(job.totalLines || 0);
  document.getElementById("totalWords").textContent = String(job.totalWords || 0);
  document.getElementById("outputSize").textContent = `${job.outputSizeKb || 0} KB`;
  if (job.status === "completed" && job.downloadUrl) {
    downloadLink.href = job.downloadUrl;
    downloadLink.removeAttribute("aria-disabled");
  } else {
    downloadLink.href = "#";
    downloadLink.setAttribute("aria-disabled", "true");
  }
}

async function loadRuntimeInfo() {
  try {
    const response = await fetch("/runtime");
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Runtime info request failed.");
    }
    const runtime = payload.runtime || {};
    const gpuName = runtime.cudaDeviceName ? ` | GPU: ${runtime.cudaDeviceName}` : "";
    const cudaVersion = runtime.cudaVersion ? ` | CUDA: ${runtime.cudaVersion}` : "";
    runtimeHint.textContent = `Runtime: ${runtime.resolvedDevice || "auto"} | Recommended model: ${runtime.recommendedModel || "recommended"}${gpuName}${cudaVersion}`;
  } catch (error) {
    runtimeHint.textContent = "Runtime info is unavailable. The server can still process files.";
  }
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/jobs/${encodeURIComponent(jobId)}`);
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Job lookup failed.");
    }

    const job = payload.job;
    renderProgress(job);
    renderExport(job);

    if (job.status === "queued" || job.status === "running") {
      pollTimer = window.setTimeout(() => {
        pollJob(jobId).catch(() => {});
      }, 1200);
      return;
    }

    if (job.status === "failed") {
      showNotice(job.error || "Processing failed.");
    }
  } catch (error) {
    showNotice(error.message || "Could not refresh job progress.");
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!uploadForm.reportValidity()) {
    return;
  }
  if (!audioInput.files.length) {
    clearNotice();
    resetProgressPanel();
    resetExportPanel();
    showNotice("Please choose an MP3 file.");
    return;
  }
  if (!lrcInput.files.length) {
    clearNotice();
    resetProgressPanel();
    resetExportPanel();
    showNotice("Please choose an LRC file.");
    return;
  }

  clearNotice();
  stopPolling();
  resetProgressPanel();
  resetExportPanel();
  setBusyState(true);

  try {
    const formData = new FormData(uploadForm);
    const response = await fetch("/process", {
      method: "POST",
      body: formData
    });
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Processing failed.");
    }

    currentJobId = payload.job.jobId;
    renderProgress(payload.job);
    renderExport(payload.job);
    setBusyState(false);
    pollJob(currentJobId).catch(() => {});
  } catch (error) {
    setBusyState(false);
    resetProgressPanel();
    resetExportPanel();
    showNotice(error.message || "Processing failed.");
  }
});

resetButton.addEventListener("click", () => {
  stopPolling();
  currentJobId = null;
  uploadForm.reset();
  clearNotice();
  resetProgressPanel();
  resetExportPanel();
});

resetProgressPanel();
resetExportPanel();
applyFastDefaults();
loadRuntimeInfo();
