// Site-wide behavior shared across every page (theme toggle, mobile nav,
// toast notifications). Page-specific logic (the upload flow) lives in
// recognize.js.
document.addEventListener("DOMContentLoaded", () => {
  // ---------- Theme toggle ----------
  const themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const resolvedCurrent = current === "light" || current === "dark"
        ? current
        : (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = resolvedCurrent === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      document.cookie = `theme=${next}; path=/; max-age=31536000; samesite=lax`;
    });
  }

  // ---------- Mobile nav ----------
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      const isOpen = navLinks.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
    });
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navLinks.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  // ---------- Toasts ----------
  const toastRegion = document.getElementById("toast-region");
  window.showToast = function showToast(message, duration = 1800) {
    if (!toastRegion) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    toastRegion.appendChild(toast);
    // Force a reflow so the transition on the next class change actually runs.
    void toast.offsetWidth;
    toast.classList.add("is-visible");
    setTimeout(() => {
      toast.classList.remove("is-visible");
      setTimeout(() => toast.remove(), 250);
    }, duration);
  };
});
