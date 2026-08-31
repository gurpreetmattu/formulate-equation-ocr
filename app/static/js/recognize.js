document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const fileInput = document.getElementById("image-input");
  const submitBtn = document.getElementById("submit-btn");
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const errorState = document.getElementById("error-state");
  const results = document.getElementById("results");

  if (!form) return;

  fileInput.addEventListener("change", () => {
    emptyState.hidden = fileInput.files.length > 0;
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
