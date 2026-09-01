document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("image-input");
  const dzEmpty = document.getElementById("dz-empty");
  const dzFile = document.getElementById("dz-file");
  const dzThumb = document.getElementById("dz-thumb");
  const dzFileName = document.getElementById("dz-file-name");
  const dzFileSize = document.getElementById("dz-file-size");
  const dzClear = document.getElementById("dz-clear");
  const submitBtn = document.getElementById("submit-btn");
  const loadingState = document.getElementById("loading-state");
  const errorState = document.getElementById("error-state");
  const previewSection = document.getElementById("preview-section");
  const results = document.getElementById("results");
  const emptyState = document.getElementById("empty-state");
  const resultsSkeleton = document.getElementById("results-skeleton");
  const dzAdjust = document.getElementById("dz-adjust");
  const historySection = document.getElementById("history-section");
  const historyRow = document.getElementById("history-row");
  const historyClear = document.getElementById("history-clear");
  const feedbackUp = document.getElementById("feedback-up");
  const feedbackDown = document.getElementById("feedback-down");

  if (!form) return;

  const MAX_BYTES = parseInt(form.dataset.maxBytes, 10) || Infinity;
  const ALLOWED_EXTENSIONS = (form.dataset.allowedExtensions || "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  const REQUEST_TIMEOUT_MS = 90000;

  function showError(message) {
    errorState.textContent = message;
    errorState.hidden = false;
  }

  // Mirrors the server-side checks in app/routes.py (_allowed_file,
  // MAX_CONTENT_LENGTH) so the user gets an immediate answer instead of a
  // round trip that ends in a 413/400.
  function validateFile(file) {
    const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return `Unsupported file type ".${ext || "?"}". Allowed: ${ALLOWED_EXTENSIONS.join(", ")}.`;
    }
    if (file.size > MAX_BYTES) {
      const maxMb = (MAX_BYTES / (1024 * 1024)).toFixed(1);
      const fileMb = (file.size / (1024 * 1024)).toFixed(1);
      return `File is too large (${fileMb} MB). Maximum upload size is ${maxMb} MB.`;
    }
    return null;
  }

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // Simple LCS-based token diff: returns the beam tokens as an array of
  // {text, changed} so differing tokens (insert/replace relative to greedy)
  // can be highlighted. Tokens are space-separated, matching how the
  // model's vocabulary is joined server-side (see tokens_to_latex).
  function diffTokens(greedyLatex, beamLatex) {
    const a = greedyLatex.split(" ");
    const b = beamLatex.split(" ");
    const n = a.length, m = b.length;
    const lcs = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
      }
    }
    const result = [];
    let i = 0, j = 0;
    while (j < m) {
      if (i < n && a[i] === b[j]) {
        result.push({ text: b[j], changed: false });
        i++; j++;
      } else if (i < n && lcs[i + 1][j] >= lcs[i][j + 1]) {
        i++;
      } else {
        result.push({ text: b[j], changed: true });
        j++;
      }
    }
    return result;
  }

  function renderBeamWithDiff(greedyLatex, beamLatex) {
    const target = document.getElementById("beam-latex");
    const looksLikeError = (s) => s.startsWith("[Prediction error") || s.startsWith("[Conversion error");
    if (!greedyLatex || !beamLatex || looksLikeError(greedyLatex) || looksLikeError(beamLatex)) {
      target.textContent = beamLatex;
      return;
    }
    const tokens = diffTokens(greedyLatex, beamLatex);
    target.innerHTML = tokens
      .map((t) => (t.changed ? `<span class="diff-token">${escapeHtml(t.text)}</span>` : escapeHtml(t.text)))
      .join(" ");
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function showSelectedFile(file) {
    dzEmpty.hidden = true;
    dzFile.hidden = false;
    dzFileName.textContent = file.name;
    dzFileSize.textContent = formatBytes(file.size);
    dzThumb.src = URL.createObjectURL(file);
    dropzone.classList.add("has-file");
    submitBtn.disabled = false;
  }

  function clearSelectedFile() {
    fileInput.value = "";
    dzEmpty.hidden = false;
    dzFile.hidden = true;
    dropzone.classList.remove("has-file");
    submitBtn.disabled = true;
  }

  // Shared entry point for both the native file picker and drag-and-drop:
  // validates before accepting, so an oversized/wrong-type file never
  // reaches the network layer.
  function handleFileSelection(file) {
    const validationError = validateFile(file);
    if (validationError) {
      clearSelectedFile();
      showError(validationError);
      return;
    }
    errorState.hidden = true;
    showSelectedFile(file);
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFileSelection(fileInput.files[0]);
  });

  // The native <label for="..."> click-to-open behavior only fires on
  // pointer clicks, and the file input itself is `hidden` so it can never
  // receive keyboard focus. Make the dropzone a keyboard-operable button.
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !fileInput.disabled) {
      e.preventDefault();
      fileInput.click();
    }
  });

  dzClear.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearSelectedFile();
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("is-dragover");
    if (fileInput.disabled) return;
    const files = e.dataTransfer.files;
    if (files.length) {
      fileInput.files = files;
      handleFileSelection(files[0]);
    }
  });

  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = document.getElementById(btn.dataset.target);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent);
        btn.classList.add("is-copied");
        setTimeout(() => btn.classList.remove("is-copied"), 1200);
        if (window.showToast) window.showToast("Copied to clipboard");
      } catch (err) {
        if (window.showToast) window.showToast("Could not copy — clipboard access was denied");
      }
    });
  });

  // Quick-start samples: fetch the sample image and feed it through the
  // exact same validation/preview path as a real upload, so nothing about
  // the rest of the flow needs to know the file didn't come from disk.
  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const filename = chip.dataset.filename;
      const img = chip.querySelector("img");
      try {
        const res = await fetch(img.getAttribute("src"));
        if (!res.ok) throw new Error("Could not load sample image.");
        const blob = await res.blob();
        const file = new File([blob], filename, { type: blob.type || "image/png" });
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        handleFileSelection(file);
      } catch (err) {
        showError("Could not load that sample image. Please try uploading your own.");
      }
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    const validationError = validateFile(fileInput.files[0]);
    if (validationError) {
      showError(validationError);
      return;
    }

    errorState.hidden = true;
    previewSection.hidden = true;
    results.hidden = true;
    loadingState.hidden = false;
    submitBtn.disabled = true;
    if (emptyState) emptyState.hidden = true;
    if (resultsSkeleton) resultsSkeleton.hidden = false;

    const sourceName = fileInput.files[0].name;
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData, signal: controller.signal });

      let data;
      try {
        data = await res.json();
      } catch (parseErr) {
        throw new Error(`Server returned an unexpected response (status ${res.status}).`);
      }

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      document.getElementById("preview-img").src = data.preview_image;
      document.getElementById("greedy-latex").textContent = data.greedy_latex;
      document.getElementById("greedy-mathml").textContent = data.greedy_mathml;
      renderBeamWithDiff(data.greedy_latex, data.beam_latex);
      document.getElementById("beam-mathml").textContent = data.beam_mathml;

      const greedyRender = document.getElementById("greedy-render");
      const beamRender = document.getElementById("beam-render");
      greedyRender.textContent = "\\[" + data.greedy_latex + "\\]";
      beamRender.textContent = "\\[" + data.beam_latex + "\\]";

      previewSection.hidden = false;
      results.hidden = false;
      resetFeedbackButtons();
      const mathjaxNote = document.getElementById("mathjax-fallback-note");
      if (window.MathJax && window.MathJax.typesetPromise) {
        if (mathjaxNote) mathjaxNote.hidden = true;
        window.MathJax.typesetPromise([greedyRender, beamRender]);
      } else if (mathjaxNote) {
        mathjaxNote.hidden = false;
      }

      pushHistory({
        name: sourceName,
        previewImage: data.preview_image,
        greedyLatex: data.greedy_latex,
        greedyMathml: data.greedy_mathml,
        beamLatex: data.beam_latex,
        beamMathml: data.beam_mathml,
      });
    } catch (err) {
      if (err.name === "AbortError") {
        showError("Request timed out. Please try again — CPU-only inference can be slow for large or complex images.");
      } else if (err instanceof TypeError) {
        showError("Could not reach the server. Check your connection and try again.");
      } else {
        showError(err.message);
      }
      if (results.hidden && emptyState) emptyState.hidden = false;
    } finally {
      clearTimeout(timeoutId);
      loadingState.hidden = true;
      if (resultsSkeleton) resultsSkeleton.hidden = true;
      submitBtn.disabled = false;
    }
  });

  // ---------- Result history (recent uploads, kept client-side) ----------

  const HISTORY_KEY = "eqocr_history_v1";
  const HISTORY_MAX = 8;

  function loadHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    } catch (err) {
      return [];
    }
  }

  function saveHistory(list) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
    } catch (err) {
      // Storage full/unavailable (e.g. private browsing) -- history is a
      // nice-to-have, so fail silently rather than interrupt the user.
    }
  }

  function renderHistory() {
    if (!historySection || !historyRow) return;
    const list = loadHistory();
    historySection.hidden = list.length === 0;
    historyRow.innerHTML = "";
    list.forEach((entry) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "history-chip";
      const name = entry.name || "image";
      chip.title = name;
      chip.innerHTML = `<img src="${entry.previewImage}" alt=""><span>${escapeHtml(name)}</span>`;
      chip.addEventListener("click", () => applyHistoryEntry(entry));
      historyRow.appendChild(chip);
    });
  }

  function pushHistory(entry) {
    const list = loadHistory();
    list.unshift(entry);
    saveHistory(list.slice(0, HISTORY_MAX));
    renderHistory();
  }

  function applyHistoryEntry(entry) {
    errorState.hidden = true;
    document.getElementById("preview-img").src = entry.previewImage;
    document.getElementById("greedy-latex").textContent = entry.greedyLatex;
    document.getElementById("greedy-mathml").textContent = entry.greedyMathml;
    renderBeamWithDiff(entry.greedyLatex, entry.beamLatex);
    document.getElementById("beam-mathml").textContent = entry.beamMathml;

    const greedyRender = document.getElementById("greedy-render");
    const beamRender = document.getElementById("beam-render");
    greedyRender.textContent = "\\[" + entry.greedyLatex + "\\]";
    beamRender.textContent = "\\[" + entry.beamLatex + "\\]";

    if (emptyState) emptyState.hidden = true;
    previewSection.hidden = false;
    results.hidden = false;
    resetFeedbackButtons();
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([greedyRender, beamRender]);
    }
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  if (historyClear) {
    historyClear.addEventListener("click", () => {
      saveHistory([]);
      renderHistory();
    });
  }

  renderHistory();

  // ---------- Feedback (was this correct?) ----------
  // Client-side only tally (localStorage) -- there is no feedback API on
  // the backend, so this is a lightweight UX signal, not telemetry.

  function resetFeedbackButtons() {
    if (feedbackUp) { feedbackUp.classList.remove("is-selected"); feedbackUp.disabled = false; }
    if (feedbackDown) { feedbackDown.classList.remove("is-selected"); feedbackDown.disabled = false; }
  }

  function recordFeedback(vote) {
    try {
      const key = "eqocr_feedback_tally";
      const tally = JSON.parse(localStorage.getItem(key) || '{"up":0,"down":0}');
      tally[vote] = (tally[vote] || 0) + 1;
      localStorage.setItem(key, JSON.stringify(tally));
    } catch (err) {
      // best-effort only
    }
    if (feedbackUp) feedbackUp.disabled = true;
    if (feedbackDown) feedbackDown.disabled = true;
    if (vote === "up" && feedbackUp) feedbackUp.classList.add("is-selected");
    if (vote === "down" && feedbackDown) feedbackDown.classList.add("is-selected");
    if (window.showToast) window.showToast("Thanks for the feedback");
  }

  if (feedbackUp) feedbackUp.addEventListener("click", () => recordFeedback("up"));
  if (feedbackDown) feedbackDown.addEventListener("click", () => recordFeedback("down"));

  // ---------- Adjust image: rotate + crop before submitting ----------

  const cropModal = document.getElementById("crop-modal");
  const cropStage = document.getElementById("crop-stage");
  const cropCanvas = document.getElementById("crop-canvas");
  const cropRect = document.getElementById("crop-rect");
  const cropRotateLeft = document.getElementById("crop-rotate-left");
  const cropRotateRight = document.getElementById("crop-rotate-right");
  const cropReset = document.getElementById("crop-reset");
  const cropCancel = document.getElementById("crop-cancel");
  const cropApply = document.getElementById("crop-apply");

  if (dzAdjust && cropModal && cropStage && cropCanvas && cropRect) {
    const cropState = { rotation: 0, img: null, rectPct: { x: 5, y: 5, w: 90, h: 90 } };
    let dragMode = null;
    let dragStart = null;

    function loadImageForCrop(file) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
      });
    }

    // The modal itself is sized in CSS (max-width: 560px, space-5 padding on
    // both the backdrop and the modal), so on a narrow phone viewport a fixed
    // 480px canvas would overflow it. Cap the display size to what the
    // viewport can actually show instead.
    function maxCropDisplaySize() {
      const horizontalReserve = 110; // backdrop + modal padding, both sides
      const verticalReserve = 260; // heading + toolbar + backdrop padding
      return {
        width: Math.max(200, Math.min(480, window.innerWidth - horizontalReserve)),
        height: Math.max(200, Math.min(420, window.innerHeight - verticalReserve)),
      };
    }

    function drawCropCanvas() {
      const img = cropState.img;
      const rot = cropState.rotation;
      const swapped = rot % 180 !== 0;
      const iw = img.naturalWidth, ih = img.naturalHeight;
      const cw = swapped ? ih : iw, ch = swapped ? iw : ih;
      cropCanvas.width = cw;
      cropCanvas.height = ch;
      const ctx = cropCanvas.getContext("2d");
      ctx.save();
      ctx.translate(cw / 2, ch / 2);
      ctx.rotate((rot * Math.PI) / 180);
      ctx.drawImage(img, -iw / 2, -ih / 2);
      ctx.restore();

      const { width: maxDisplayWidth, height: maxDisplayHeight } = maxCropDisplaySize();
      const scale = Math.min(1, maxDisplayWidth / cw, maxDisplayHeight / ch);
      const displayW = Math.round(cw * scale);
      const displayH = Math.round(ch * scale);
      cropCanvas.style.width = displayW + "px";
      cropCanvas.style.height = displayH + "px";
      cropStage.style.width = displayW + "px";
      cropStage.style.height = displayH + "px";
    }

    function renderCropRect() {
      const { x, y, w, h } = cropState.rectPct;
      cropRect.style.left = x + "%";
      cropRect.style.top = y + "%";
      cropRect.style.width = w + "%";
      cropRect.style.height = h + "%";
    }

    function resetCropRect() {
      cropState.rectPct = { x: 5, y: 5, w: 90, h: 90 };
      renderCropRect();
    }

    function pctFromEvent(e) {
      const rect = cropStage.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;
      return { x: Math.min(100, Math.max(0, x)), y: Math.min(100, Math.max(0, y)) };
    }

    cropRect.addEventListener("pointerdown", (e) => {
      dragMode = "move";
      dragStart = { ...pctFromEvent(e), rect: { ...cropState.rectPct } };
      cropRect.setPointerCapture(e.pointerId);
      e.stopPropagation();
    });

    cropRect.querySelectorAll(".crop-handle").forEach((handle) => {
      handle.addEventListener("pointerdown", (e) => {
        dragMode = handle.classList[1];
        dragStart = { ...pctFromEvent(e), rect: { ...cropState.rectPct } };
        handle.setPointerCapture(e.pointerId);
        e.stopPropagation();
      });
    });

    cropStage.addEventListener("pointermove", (e) => {
      if (!dragMode) return;
      const cur = pctFromEvent(e);
      const dx = cur.x - dragStart.x, dy = cur.y - dragStart.y;
      const r0 = dragStart.rect;
      const MIN = 8;
      let r = { ...r0 };
      if (dragMode === "move") {
        r.x = Math.min(100 - r0.w, Math.max(0, r0.x + dx));
        r.y = Math.min(100 - r0.h, Math.max(0, r0.y + dy));
      } else {
        if (dragMode.includes("w")) {
          const newX = Math.min(r0.x + r0.w - MIN, Math.max(0, r0.x + dx));
          r.w = r0.x + r0.w - newX;
          r.x = newX;
        }
        if (dragMode.includes("e")) {
          r.w = Math.min(100 - r0.x, Math.max(MIN, r0.w + dx));
        }
        if (dragMode.includes("n")) {
          const newY = Math.min(r0.y + r0.h - MIN, Math.max(0, r0.y + dy));
          r.h = r0.y + r0.h - newY;
          r.y = newY;
        }
        if (dragMode.includes("s")) {
          r.h = Math.min(100 - r0.y, Math.max(MIN, r0.h + dy));
        }
      }
      cropState.rectPct = r;
      renderCropRect();
    });

    ["pointerup", "pointercancel"].forEach((evt) => {
      cropStage.addEventListener(evt, () => { dragMode = null; dragStart = null; });
    });

    function openCropTool(reRotate) {
      cropState.rotation = reRotate;
      drawCropCanvas();
      resetCropRect();
    }

    cropRotateLeft.addEventListener("click", () => openCropTool((cropState.rotation + 270) % 360));
    cropRotateRight.addEventListener("click", () => openCropTool((cropState.rotation + 90) % 360));
    cropReset.addEventListener("click", () => openCropTool(0));

    dzAdjust.addEventListener("click", async (e) => {
      // dzAdjust lives inside the <label for="image-input"> that wraps the
      // hidden file input -- without stopping propagation, this click would
      // also bubble to the label and pop the native file picker on top of
      // the crop modal (dzClear has the same guard for the same reason).
      e.preventDefault();
      e.stopPropagation();
      if (!fileInput.files.length) return;
      try {
        cropState.img = await loadImageForCrop(fileInput.files[0]);
      } catch (err) {
        showError("Could not load image for adjusting.");
        return;
      }
      openCropTool(0);
      cropModal.hidden = false;
      // Lock background scroll -- the backdrop covers the viewport but
      // doesn't stop wheel/touch scroll from reaching the page underneath.
      document.body.style.overflow = "hidden";
      // Move focus into the dialog so keyboard/screen-reader users land
      // somewhere sensible instead of staying on the (now-hidden-behind-
      // the-modal) Adjust button.
      cropRotateLeft.focus();
    });

    function closeCropModal() {
      cropModal.hidden = true;
      document.body.style.overflow = "";
      dzAdjust.focus();
    }

    cropCancel.addEventListener("click", closeCropModal);
    cropModal.addEventListener("click", (e) => { if (e.target === cropModal) closeCropModal(); });
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" && e.key !== "Tab") return;
      if (cropModal.hidden) return;
      if (e.key === "Escape") {
        closeCropModal();
        return;
      }
      // Basic focus trap: keep Tab cycling within the dialog while it's open.
      const focusable = cropModal.querySelectorAll("button, [tabindex]:not([tabindex='-1'])");
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    });

    cropApply.addEventListener("click", () => {
      const { x, y, w, h } = cropState.rectPct;
      const sx = (x / 100) * cropCanvas.width;
      const sy = (y / 100) * cropCanvas.height;
      const sw = (w / 100) * cropCanvas.width;
      const sh = (h / 100) * cropCanvas.height;
      const outCanvas = document.createElement("canvas");
      outCanvas.width = Math.max(1, Math.round(sw));
      outCanvas.height = Math.max(1, Math.round(sh));
      outCanvas.getContext("2d").drawImage(cropCanvas, sx, sy, sw, sh, 0, 0, outCanvas.width, outCanvas.height);
      outCanvas.toBlob((blob) => {
        if (!blob) {
          showError("Could not process the adjusted image.");
          return;
        }
        const originalName = (fileInput.files[0] && fileInput.files[0].name) || "image.png";
        const baseName = originalName.replace(/\.[^.]+$/, "");
        const newFile = new File([blob], `${baseName}-adjusted.png`, { type: "image/png" });
        const dt = new DataTransfer();
        dt.items.add(newFile);
        fileInput.files = dt.files;
        handleFileSelection(newFile);
        closeCropModal();
        if (window.showToast) window.showToast("Image adjusted");
      }, "image/png");
    });
  }
});
