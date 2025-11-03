// js/retail_request.js
const API_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {
  const requestForm = document.getElementById('requestForm');
  const descriptionEl = document.getElementById('description');
  const departmentEl = document.getElementById('department');
  const successMessage = document.getElementById('successMessage');
  const successText = document.getElementById('successText');
  const errorMessage = document.getElementById('errorMessage');
  const errorText = document.getElementById('errorText');
  const myRequestsList = document.getElementById('myRequestsList');

  function showSuccess(msg) {
    successText.textContent = msg;
    successMessage.style.display = 'flex';
    setTimeout(() => successMessage.style.display = 'none', 5000);
  }
  function showError(msg) {
    errorText.textContent = msg;
    errorMessage.style.display = 'flex';
    setTimeout(() => errorMessage.style.display = 'none', 5000);
  }

  async function loadMyRequests() {
    try {
      const res = await fetch(`${API_URL}/my-requests`, { credentials: 'include' });
      const j = await res.json();
      if (!j.success) {
        showError(j.message || 'Failed to load requests');
        return;
      }
      renderRequests(j.requests || []);
    } catch (err) {
      console.error('loadMyRequests', err);
      showError('Connection error while loading requests');
    }
  }

  function renderRequests(list) {
    if (!list.length) {
      myRequestsList.innerHTML = `<div style="padding:16px; background:#fff; border-radius:8px;">No requests submitted yet</div>`;
      return;
    }
    const html = list.map(r => {
      const statusBadge = r.status === 'pending' ? `<span style="color:#1E88E5;font-weight:700">PENDING</span>` :
        `<span style="color:#16a34a;font-weight:700">RESOLVED</span>`;
      return `
        <div style="background:white;padding:16px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
              <div style="font-weight:700">Request #${r.requestID} • ${new Date(r.requestedAt).toLocaleString()}</div>
              <div style="color:#6b7280;margin-top:6px">${r.department} — ${statusBadge}</div>
            </div>
            <div style="text-align:right">
              ${r.status === 'resolved' ? `<div style="font-size:14px">Job ID: <strong>${r.jobID}</strong></div><div style="color:#6b7280;font-size:12px">Resolved by ${r.resolvedByUsername}</div>` : `<div style="color:#6b7280;font-size:12px">Awaiting IT</div>`}
            </div>
          </div>
          <div style="margin-top:12px;color:#374151">${escapeHtml(r.description)}</div>
        </div>
      `;
    }).join('');
    myRequestsList.innerHTML = html;
  }

  function escapeHtml(s){
    if(!s) return '';
    return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
  }

  requestForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const description = descriptionEl.value.trim();
    const department = departmentEl.value.trim() || 'Retail Banking';
    if (!description) {
      showError('Description is required');
      return;
    }
    try {
      const res = await fetch(`${API_URL}/submit-prediction-request`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description, department })
      });
      const j = await res.json();
      if (j.success) {
        showSuccess('Request submitted. Request ID: ' + (j.requestID || '—'));
        descriptionEl.value = '';
        await loadMyRequests();
      } else {
        showError(j.message || 'Submission failed');
      }
    } catch (err) {
      console.error('submit request', err);
      showError('Connection error - could not submit request');
    }
  });

  // cancel
  document.getElementById('cancelBtn').addEventListener('click', () => {
    if (confirm('Cancel request?')) {
      descriptionEl.value = '';
    }
  });

  // initial load
  loadMyRequests();
});
