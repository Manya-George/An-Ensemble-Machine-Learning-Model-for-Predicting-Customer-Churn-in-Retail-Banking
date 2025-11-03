document.addEventListener("DOMContentLoaded", () => {
    const API_URL = "http://localhost:5000/api";
    const refreshBtn = document.getElementById('refreshBtn');
    const refreshSpinner = document.getElementById('refreshSpinner');
    const refreshModelBtn = document.getElementById('refreshModelBtn');
    const alertsContainer = document.getElementById('alertsContainer');
    const notificationBadge = document.getElementById('notificationBadge');
    const userNameSpan = document.querySelector('.user-name');
    const logoutBtn = document.getElementById('logoutBtn');

    // ---------- Helper Functions ---------- //

    function showSpinner(show = true) {
        if (refreshSpinner) {
            refreshSpinner.style.display = show ? 'inline-block' : 'none';
        }
    }

    function showAlert(type, message, timeout = 8000) {
        if (!alertsContainer) return;
        const el = document.createElement('div');
        el.className = `alert ${type}`;
        el.innerHTML = `<strong>${type === 'error' ? 'Error' : type === 'warn' ? 'Warning' : 'OK'}</strong>&nbsp;&nbsp;<span>${message}</span>`;
        alertsContainer.prepend(el);
        if (timeout) setTimeout(() => el.remove(), timeout);
    }

    async function checkAuth() {
        try {
            const res = await fetch(`${API_URL}/check-auth`, {
                credentials: 'include'
            });
            if (!res.ok) {
                window.location.href = "../login.html";
                return false;
            }
            const j = await res.json();
            if (!j.authenticated) {
                window.location.href = "../login.html";
                return false;
            }
            // Update user name on dashboard
            if (userNameSpan) userNameSpan.textContent = j.username || "User";
            return true;
        } catch (err) {
            console.error("checkAuth error:", err);
            window.location.href = "../login.html";
            return false;
        }
    }

    async function handleLogout() {
        try {
            const res = await fetch(`${API_URL}/logout`, {
                method: 'POST',
                credentials: 'include'
            });
            if (res.ok) {
                sessionStorage.clear();
                localStorage.clear();
                window.location.href = "../login.html";
            } else {
                showAlert('warn', 'Logout failed. Please try again.');
            }
        } catch (err) {
            console.error('Logout error:', err);
            showAlert('error', 'Unable to log out.');
        }
    }

    // ---------- Dashboard Data Fetching ---------- //

    async function fetchAdminStats() {
        try {
            const res = await fetch(`${API_URL}/admin-stats`, {
                credentials: 'include'
            });
            const j = await res.json();
            if (j.success) {
                document.getElementById("totalUsers").textContent = j.data.total_users;
                document.getElementById("activeUsers").textContent = j.data.active_users;
                document.getElementById("revokedUsers").textContent = j.data.revoked_users;
                document.getElementById("totalLogs").textContent = j.data.total_logs;
            } else {
                showAlert('warn', j.message || 'Failed to fetch admin stats');
            }
        } catch (err) {
            console.error('fetchAdminStats', err);
            showAlert('warn', 'Could not fetch admin stats');
        }
    }

    async function fetchRecentLogs() {
        try {
            const res = await fetch(`${API_URL}/recent-logs`, {
                credentials: 'include'
            });
            if (res.status === 401) {
                showAlert('error', 'Session expired. Redirecting...');
                setTimeout(() => (window.location.href = "../login.html"), 2000);
                return;
            }

            const j = await res.json();
            if (j.success) {
                const tbody = document.querySelector("#logsTable tbody");
                if (tbody) {
                    tbody.innerHTML = "";
                    j.logs.forEach(log => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${new Date(log.timestamp).toLocaleString()}</td>
                            <td>${log.username}</td>
                            <td>${log.action || log.endpoint}</td>
                            <td>${log.status_code || log.status}</td>`;
                        tbody.appendChild(row);
                    });
                }
            } else {
                showAlert('warn', j.message || 'Could not fetch logs');
            }
        } catch (err) {
            console.error('fetchRecentLogs', err);
            showAlert('warn', 'Failed to fetch recent logs');
        }
    }

    async function fetchSystemHealth() {
        try {
            const res = await fetch(`${API_URL}/system-health`, {
                credentials: 'include'
            });
            const j = await res.json();

            if (!j.success) {
                showAlert('warn', 'System health returned false');
                return;
            }

            const d = j.data;

            // DB
            const dbStatusEl = document.getElementById("dbStatus");
            const dbErrorEl = document.getElementById("dbError");
            if (d.db.status === 'active') {
                dbStatusEl.textContent = 'Active';
                dbStatusEl.classList.remove('down');
                dbStatusEl.classList.add('up');
                dbErrorEl.style.display = 'none';
            } else {
                dbStatusEl.textContent = 'Down';
                dbStatusEl.classList.remove('up');
                dbStatusEl.classList.add('down');
                dbErrorEl.style.display = 'block';
                dbErrorEl.textContent = d.db.error || 'DB error';
                showAlert('error', 'Database connection down.');
            }

            // Model
            const modelStatusEl = document.getElementById("modelStatus");
            const modelInfoEl = document.getElementById("modelInfo");
            if (d.model.loaded) {
                modelStatusEl.textContent = 'Deployed & Ready';
                modelStatusEl.classList.add('up');
                modelInfoEl.textContent = `${d.model.model_path || ''} ${d.model.file_exists ? '(present)' : '(missing)'}`;
                if (d.model.file_mtime_iso) modelInfoEl.textContent += ` • ${new Date(d.model.file_mtime_iso).toLocaleString()}`;
            } else {
                modelStatusEl.textContent = 'Not loaded';
                modelStatusEl.classList.add('down');
                modelInfoEl.textContent = d.model.model_path ? `File: ${d.model.model_path}` : '';
                showAlert('warn', 'Model is not loaded in memory.');
            }

            // API uptime
            const apiUptimeEl = document.getElementById("apiUptime");
            const apiStartEl = document.getElementById("apiStart");
            const seconds = d.api.uptime_seconds || 0;
            const days = Math.floor(seconds / 86400);
            const hours = Math.floor((seconds % 86400) / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            const parts = [];
            if (days) parts.push(`${days}d`);
            if (hours) parts.push(`${hours}h`);
            parts.push(`${mins}m`);
            apiUptimeEl.textContent = parts.join(' ');
            apiStartEl.textContent = `Started: ${new Date(d.api.app_start_time).toLocaleString()}`;

            // API calls
            document.getElementById("apiCalls24h").textContent = d.api.api_calls_last_24h ?? 0;

        } catch (err) {
            console.error('fetchSystemHealth', err);
            showAlert('warn', 'Could not fetch system health');
        }
    }

    async function fetchModelPerformance() {
        try {
            const res = await fetch(`${API_URL}/model-performance`, {
                credentials: 'include'
            });
            const j = await res.json();
            if (!j.success) {
                showAlert('warn', 'Model performance returned false');
                return;
            }

            const m = j.data;
            document.getElementById('modelFile').textContent = m.model_path || '—';
            document.getElementById('modelFileMtime').textContent = m.file_mtime_iso ? new Date(m.file_mtime_iso).toLocaleString() : '';
            document.getElementById('modelAcc').textContent = m.metrics?.accuracy ? m.metrics.accuracy.toFixed(3) : '—';
            document.getElementById('modelPrecision').textContent = m.metrics?.precision ? m.metrics.precision.toFixed(3) : '—';
            document.getElementById('modelRecall').textContent = m.metrics?.recall ? m.metrics.recall.toFixed(3) : '—';
        } catch (err) {
            console.error('fetchModelPerformance', err);
            showAlert('warn', 'Could not fetch model performance');
        }
    }

    async function fetchApiMetrics() {
        try {
            const res = await fetch(`${API_URL}/api-metrics`, {
                credentials: 'include'
            });
            const j = await res.json();
            if (!j.success) return;

            const newCount = j.data.requests_last_hour || 0;
            if (newCount > 0) {
                notificationBadge.style.display = 'inline-block';
                notificationBadge.textContent = newCount;
            } else {
                notificationBadge.style.display = 'none';
            }
        } catch (err) {
            console.error('fetchApiMetrics', err);
        }
    }

    // ---------- Combined Refresh ---------- //
    async function refreshAll() {
        showSpinner(true);
        await Promise.all([
            fetchAdminStats(),
            fetchRecentLogs(),
            fetchSystemHealth(),
            fetchModelPerformance(),
            fetchApiMetrics()
        ]);
        setTimeout(() => showSpinner(false), 300);
    }

    // ---------- Event Handlers ---------- //
    if (refreshBtn) refreshBtn.addEventListener('click', refreshAll);
    if (refreshModelBtn) refreshModelBtn.addEventListener('click', async () => {
        showSpinner(true);
        await fetchModelPerformance();
        setTimeout(() => showSpinner(false), 300);
    });
    if (logoutBtn) logoutBtn.addEventListener('click', handleLogout);

    // ---------- Initialize ---------- //
    (async () => {
        const authed = await checkAuth();
        if (authed) {
            refreshAll();
            setInterval(async () => {
                await Promise.all([fetchSystemHealth(), fetchApiMetrics()]);
            }, 60000);
        }
    })();
});
