// js/session_handler.js
document.addEventListener("DOMContentLoaded", async () => {
  const API_URL = "http://localhost:5000/api";
  const userNameSpan = document.querySelector(".user-name");
  const logoutBtn = document.getElementById("logoutBtn");

  async function checkAuth() {
    try {
      const res = await fetch(`${API_URL}/check-auth`, { credentials: "include" });
      if (!res.ok) throw new Error("not ok");
      const data = await res.json();
      if (!data.authenticated) window.location.href = "../login.html";
      if (userNameSpan) userNameSpan.textContent = data.username || "User";
    } catch {
      window.location.href = "../login.html";
    }
  }

  async function handleLogout() {
    const res = await fetch(`${API_URL}/logout`, { method: "POST", credentials: "include" });
    sessionStorage.clear();
    localStorage.clear();
    window.location.href = "../login.html";
  }

  if (logoutBtn) logoutBtn.addEventListener("click", handleLogout);
  await checkAuth();
});
