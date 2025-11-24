// js/it_requests.js
const API_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {
  const requestsContainer = document.getElementById('requestsContainer');

  async function loadPending() {
    try {
      const res = await fetch(`${API_URL}/pending-requests?status=pending`, { credentials: 'include' });
      const j = await res.json();
      if (!j.success) {
        requestsContainer.innerHTML = `<div style="padding:12px;background:#fee2e2;border-radius:8px">${j.message || 'Error loading'}</div>`;
        return;
      }
      renderPending(j.requests || []);
    } catch (err) {
      console.error('loadPending', err);
      requestsContainer.innerHTML = `<div style="padding:12px;background:#fee2e2;border-radius:8px">Connection error</div>`;
    }
  }

  function renderPending(list) {
    if (!list.length) {
      requestsContainer.innerHTML = `<div style="padding:16px;background:#fff;border-radius:8px;">No pending requests</div>`;
      return;
    }
    const html = list.map(r => {
      return `
        <div class="request-card" style="background:white;padding:16px;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,0.06);margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
              <div style="font-size: 14px; font-weight:600; color: #374151; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">Request #${r.requestID} • ${new Date(r.requestedAt).toLocaleString()}</div>
              <div style="font-size: 14px; margin-top:6px; color: #6b7280;">${r.department} • by ${r.requester_username}</div>
            </div>
            <div style="display:flex;gap:8px;align-items:center">
              <input type="number" placeholder="jobID" class="job-input" data-requestid="${r.requestID}" style="padding:8px;border-radius:6px;border:1px solid #e5e7eb;width:120px" />
              <button class="resolve-btn btn btn-primary" data-requestid="${r.requestID}">Mark Done</button>
            </div>
          </div>
          <div style="margin-top:12px;color:#374151">${escapeHtml(r.description)}</div>
        </div>
      `;
    }).join('');
    requestsContainer.innerHTML = html;

    // attach listeners
    document.querySelectorAll('.resolve-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const reqId = btn.getAttribute('data-requestid');
        const jobInput = document.querySelector(`.job-input[data-requestid="${reqId}"]`);
        const jobID = jobInput.value.trim();
        if (!jobID) {
          alert('Please enter a jobID before marking done.');
          return;
        }
        if (!confirm(`Mark request ${reqId} as resolved with jobID ${jobID}?`)) return;
        try {
          btn.disabled = true;
          const res = await fetch(`${API_URL}/resolve-request`, {
            method: 'POST',
            credentials: 'include',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ requestID: Number(reqId), jobID: Number(jobID) })
          });
          const j = await res.json();
          if (j.success) {
            alert('Request resolved');
            await loadPending();
          } else {
            alert('Failed: ' + (j.message || 'Unknown'));
          }
        } catch (err) {
          console.error('resolve', err);
          alert('Connection error');
        } finally {
          btn.disabled = false;
        }
      });
    });
  }

  function escapeHtml(s){
    if(!s) return '';
    return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
  }

  loadPending();
  // refresh every 30s
  setInterval(loadPending, 30_000);
});
