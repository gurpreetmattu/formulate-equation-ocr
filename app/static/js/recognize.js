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
  const results = document.getElementById("results");

  if (!form) return;

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

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) showSelectedFile(fileInput.files[0]);
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
      showSelectedFile(files[0]);
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

    errorState.hidden = true;
    results.hidden = true;
    loadingState.hidden = false;
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      document.getElementById("preview-img").src = data.preview_image;
      document.getElementById("greedy-latex").textContent = data.greedy_latex;
      document.getElementById("greedy-mathml").textContent = data.greedy_mathml;
      document.getElementById("beam-latex").textContent = data.beam_latex;
      document.getElementById("beam-mathml").textContent = data.beam_mathml;

      const greedyRender = document.getElementById("greedy-render");
      const beamRender = document.getElementById("beam-render");
      greedyRender.textContent = "\\[" + data.greedy_latex + "\\]";
      beamRender.textContent = "\\[" + data.beam_latex + "\\]";

      results.hidden = false;
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([greedyRender, beamRender]);
      }
    } catch (err) {
      errorState.textContent = err.message;
      errorState.hidden = false;
    } finally {
      loadingState.hidden = true;
      submitBtn.disabled = false;
    }
  });
});
