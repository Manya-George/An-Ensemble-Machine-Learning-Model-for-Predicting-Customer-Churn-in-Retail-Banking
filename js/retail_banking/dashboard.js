// js/retail_dashboard.js
document.addEventListener('DOMContentLoaded', () => {
  const API_URL = 'http://localhost:5000/api';
  const refreshBtn = document.getElementById('refreshBtn');
  const refreshSpinner = document.getElementById('refreshSpinner');
  const newRequestToggle = document.getElementById('newRequestToggle');
  const newRequestPanel = document.getElementById('newRequestPanel');
  const requestForm = document.getElementById('requestForm');
  const cancelRequestBtn = document.getElementById('cancelRequestBtn');
  const statusFilter = document.getElementById('statusFilter');
  const searchRequest = document.getElementById('searchRequest');
  const requestsTableBody = document.querySelector('#requestsTable tbody');
  const pendingCount = document.getElementById('pendingCount');
  const resolvedCount = document.getElementById('resolvedCount');
  const totalCount = document.getElementById('totalCount');
  const avgResolution = document.getElementById('avgResolution');
  const requestsCount = document.getElementById('requestsCount');
  const notificationBadge = document.getElementById('notificationBadge');
  const toastContainer = document.getElementById('toastContainer');
  let pollingIntervalId = null;
  let requestsCache = [];

  function showSpinner(show = true) {
    if (!refreshSpinner) return;
    refreshSpinner.style.display = show ? 'inline-block' : 'none';
  }

  function toast(message, type = 'info', timeout = 5000) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.style.background = type === 'error' ? '#fee2e2' : type === 'success' ? '#d1fae5' : '#fff';
    el.style.border = '1px solid rgba(0,0,0,0.06)';
    el.style.padding = '10px 14px';
    el.style.marginBottom = '10px';
    el.style.borderRadius = '8px';
    el.textContent = message;
    toastContainer.prepend(el);
    setTimeout(() => el.remove(), timeout);
  }

  // Check auth and show username
  async function checkAuth() {
    try {
      const res = await fetch(`${API_URL}/check-auth`, { credentials: 'include' });
      if (!res.ok) { window.location.href = 'login.html'; return null; }
      const j = await res.json();
      if (!j.authenticated) { window.location.href = 'login.html'; return null; }
      // show username in header if present
      const userNameSpan = document.querySelector('.user-name');
      if (userNameSpan) userNameSpan.textContent = j.username || 'User';
      return j;
    } catch (err) {
      console.error('checkAuth error', err);
      window.location.href = 'login.html';
      return null;
    }
  }

  // Fetch dashboard summary: counts + avg time
  async function fetchDashboardStats() {
    try {
      const res = await fetch(`${API_URL}/retail-dashboard-stats`, { credentials: 'include' });
      if (!res.ok) { console.warn('fetchDashboardStats returned', res.status); return; }
      const j = await res.json();
      if (!j.success) return;
      pendingCount.textContent = j.data.pending || 0;
      resolvedCount.textContent = j.data.resolved || 0;
      totalCount.textContent = j.data.total || 0;
      avgResolution.textContent = j.data.avg_resolution || '-';
    } catch (err) {
      console.error('fetchDashboardStats', err);
    }
  }

  // Fetch requests list. Query params: ?status=&mine=1 (makes backend filter by current user)
  async function fetchRequests({ status = 'all', mine = true } = {}) {
    try {
      const params = new URLSearchParams();
      if (status && status !== 'all') params.set('status', status);
      if (mine) params.set('mine', '1');
      const url = `${API_URL}/prediction-requests?${params.toString()}`;
      const res = await fetch(url, { credentials: 'include' });
      if (res.status === 401) { toast('Session expired — redirecting', 'error'); setTimeout(()=>window.location.href='login.html',1200); return; }
      const j = await res.json();
      if (!j.success) { toast('Failed to fetch requests', 'error'); return; }
      requestsCache = j.requests || [];
      renderRequests(requestsCache);
      updateCountsFromRequests(requestsCache);
    } catch (err) {
      console.error('fetchRequests', err);
      toast('Error fetching requests', 'error');
    }
  }

  function updateCountsFromRequests(list) {
    const total = list.length;
    const pending = list.filter(r => r.status === 'pending').length;
    const resolved = list.filter(r => r.status === 'resolved').length;
    requestsCount.textContent = `${total} requests shown`;
    if (pendingCount) pendingCount.textContent = pending;
    if (resolvedCount) resolvedCount.textContent = resolved;
    if (totalCount) totalCount.textContent = total;
  }

  function renderRequests(list) {
    requestsTableBody.innerHTML = '';
    if (!list || list.length === 0) {
      requestsTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:28px;color:#666;">No requests found</td></tr>`;
      return;
    }
    list.forEach(r => {
      const tr = document.createElement('tr');
      const desc = (r.description || '').slice(0,180);
      const requestedAt = r.requestedAt ? new Date(r.requestedAt).toLocaleString() : '-';
      const resolvedAt = r.resolvedAt ? new Date(r.resolvedAt).toLocaleString() : '-';
      const jobCell = r.jobID ? `<a href="prediction_results.html?jobId=${r.jobID}" style="text-decoration: none;">${r.jobID}</a>` : '-';
      const statusChip = `<span class="chip ${r.status}">${r.status.toUpperCase()}</span>`;

      // Action column: if current user is IT admin and request is pending show resolve form
      let actionHtml = '-';
      if (r.status === 'pending' && r._canResolve) {
        actionHtml = `<div style="display:flex;gap:8px;align-items:center;">
                        <input type="number" min="1" placeholder="jobID" data-req="${r.requestID}" class="resolveJobInput" style="width:100px;padding:6px;border-radius:8px;border:1px solid #e5e7eb;">
                        <button class="btn btn-primary resolveBtn" data-req="${r.requestID}">Mark Resolved</button>
                      </div>`;
      } else if (r.status === 'resolved' && r.jobID) {
        actionHtml = `<button class="btn btn-secondary viewJobBtn" data-job="${r.jobID}">View Job</button>`;
      }

      tr.innerHTML = `
        <td>${r.requestID}</td>
        <td title="${r.description || ''}">${desc}${ r.description && r.description.length>180 ? '…' : '' }</td>
        <td>${r.requester_username || r.requester || 'N/A'}</td>
        <td>${statusChip}</td>
        <td>${requestedAt}</td>
        <td>${resolvedAt}</td>
        <td>${jobCell}</td>
        <td>${actionHtml}</td>
      `;
      requestsTableBody.appendChild(tr);
    });

    // Attach listeners for resolve buttons
    document.querySelectorAll('.resolveBtn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const requestID = btn.getAttribute('data-req');
        const input = document.querySelector(`.resolveJobInput[data-req="${requestID}"]`);
        const jobID = input ? input.value.trim() : null;
        if (!jobID) { toast('Enter a valid job ID before marking resolved', 'error'); return; }
        // Confirm and call backend
        if (!confirm(`Mark request ${requestID} resolved with Job ID ${jobID}?`)) return;
        try {
          const res = await fetch(`${API_URL}/prediction-requests/${requestID}/resolve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ jobID: Number(jobID) })
          });
          const j = await res.json();
          if (res.ok && j.success) {
            toast('Request marked resolved', 'success');
            await fetchRequests({ status: statusFilter.value, mine: true });
          } else {
            toast(j.message || 'Failed to mark resolved', 'error');
          }
        } catch (err) {
          console.error('resolve error', err);
          toast('Error resolving request', 'error');
        }
      });
    });

    // Attach view job buttons
    document.querySelectorAll('.viewJobBtn').forEach(b => {
      b.addEventListener('click', () => {
        const job = b.getAttribute('data-job');
        if (job) window.location.href = `prediction_results.html?jobId=${job}`;
      });
    });
  }

  // Submit new request
  requestForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const description = document.getElementById('description').value.trim();
    const department = document.getElementById('department').value.trim();
    const notifyEmail = document.getElementById('notifyEmail').value.trim();

    if (!description) { toast('Description is required', 'error'); return; }
    try {
      const res = await fetch(`${API_URL}/prediction-requests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ description, department, notifyEmail })
      });
      const j = await res.json();
      if (res.ok && j.success) {
        toast('Request created successfully', 'success');
        requestForm.reset();
        newRequestPanel.style.display = 'none';
        await fetchRequests({ status: statusFilter.value, mine: true });
      } else {
        toast(j.message || 'Failed to create request', 'error');
      }
    } catch (err) {
      console.error('create request error', err);
      toast('Network error creating request', 'error');
    }
  });

  // Toggle new request panel
  newRequestToggle.addEventListener('click', () => {
    newRequestPanel.style.display = newRequestPanel.style.display === 'none' ? 'block' : 'none';
  });
  cancelRequestBtn.addEventListener('click', () => { newRequestPanel.style.display = 'none'; requestForm.reset(); });

  // Filters
  statusFilter.addEventListener('change', () => fetchRequests({ status: statusFilter.value, mine: true }));
  searchRequest.addEventListener('input', () => {
    const q = searchRequest.value.trim().toLowerCase();
    const filtered = requestsCache.filter(r => {
      if (!q) return true;
      const inDesc = (r.description || '').toLowerCase().includes(q);
      const inRequester = (r.requester_username || '').toLowerCase().includes(q);
      const inJob = r.jobID && String(r.jobID).includes(q);
      return inDesc || inRequester || inJob;
    });
    renderRequests(filtered);
    updateCountsFromRequests(filtered);
  });

  // Manual refresh
  refreshBtn.addEventListener('click', async () => {
    showSpinner(true);
    await Promise.all([fetchDashboardStats(), fetchRequests({ status: statusFilter.value, mine: true })]);
    setTimeout(() => showSpinner(false), 400);
  });

  // Polling for updates (every 30s) and show notification badge when pending changed
  async function startPolling() {
    try {
      // initial state
      await Promise.all([fetchDashboardStats(), fetchRequests({ status: statusFilter.value, mine: true })]);
      pollingIntervalId = setInterval(async () => {
        const prevPending = requestsCache.filter(r => r.status === 'pending').length;
        await fetchDashboardStats();
        await fetchRequests({ status: statusFilter.value, mine: true });
        const newPending = requestsCache.filter(r => r.status === 'pending').length;
        if (newPending < prevPending) {
          // some were resolved -> show toast
          toast('A request was resolved — check your requests', 'success');
        } else if (newPending > prevPending) {
          // new pending
          toast('New request(s) received', 'info');
        }
        if (newPending > 0) {
          notificationBadge.style.display = 'inline-block';
          notificationBadge.textContent = newPending;
        } else {
          notificationBadge.style.display = 'none';
        }
      }, 30000);
    } catch (err) {
      console.error('startPolling error', err);
    }
  }

  // Initialize
  (async () => {
    const user = await checkAuth();
    if (!user) return;
    // show/hide UI depending on role: IT admin sees all requests and resolve actions
    const mineOnly = user.role !== 'IT admin';
    // call the backend endpoints using mine param
    await fetchDashboardStats();
    await fetchRequests({ status: statusFilter.value, mine: mineOnly });
    startPolling();
  })();
});
