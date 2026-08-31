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
      } catch (err) {
        /* clipboard API unavailable or denied — silently ignore */
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
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([greedyRender, beamRender]);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        showError("Request timed out. Please try again — CPU-only inference can be slow for large or complex images.");
      } else if (err instanceof TypeError) {
        showError("Could not reach the server. Check your connection and try again.");
      } else {
        showError(err.message);
      }
    } finally {
      clearTimeout(timeoutId);
      loadingState.hidden = true;
      submitBtn.disabled = false;
    }
  });
});
