const API_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {
  const logsTableBody = document.querySelector('#logsTable tbody');
  const applyBtn = document.getElementById('applyFiltersBtn');
  const clearBtn = document.getElementById('clearFiltersBtn');
  const userInput = document.getElementById('filterUser');
  const endpointInput = document.getElementById('filterEndpoint');
  const statusInput = document.getElementById('filterStatus');
  const startDateInput = document.getElementById('filterStart');
  const endDateInput = document.getElementById('filterEnd');

  async function checkAuth() {
    try {
      const res = await fetch(`${API_URL}/check-auth`, { credentials: 'include' });
      const data = await res.json();

      if (!data.authenticated) {
        alert('Session expired. Please log in again.');
        window.location.href = '../login.html';
        return false;
      }
      document.querySelector('.user-name').textContent = data.username || 'IT Admin';
      return true;
    } catch (err) {
      console.error('Auth check failed:', err);
      window.location.href = '../login.html';
      return false;
    }
  }

  async function loadLogs(filters = {}) {
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, val]) => {
        if (val && val.trim() !== '') params.append(key, val);
      });

      const res = await fetch(`${API_URL}/system-logs?${params.toString()}`, {
        credentials: 'include'
      });
      const data = await res.json();

      if (!data.success) {
        alert('Failed to load logs: ' + data.message);
        return;
      }

      renderLogs(data.logs);
    } catch (err) {
      console.error('Error loading logs:', err);
      alert('Server error while loading logs.');
    }
  }

  function renderLogs(logs) {
    logsTableBody.innerHTML = '';

    if (!logs || logs.length === 0) {
      logsTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:20px;color:#666;">No logs found</td></tr>`;
      return;
    }

    logs.forEach(log => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${new Date(log.timestamp).toLocaleString()}</td>
        <td>${log.username || 'N/A'}</td>
        <td>${log.endpoint}</td>
        <td>${log.method}</td>
        <td>${log.action_description || '-'}</td>
        <td>${log.status_code}</td>
      `;
      logsTableBody.appendChild(tr);
    });
  }

  // --- Filter Handlers ---
  if (applyBtn) {
    applyBtn.addEventListener('click', () => {
      console.log('Apply filters clicked');
      const filters = {
        start_date: startDateInput.value,
        end_date: endDateInput.value,
        user: userInput.value.trim(),
        endpoint: endpointInput.value.trim(),
        status: statusInput.value.trim()
      };
      loadLogs(filters);
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      console.log('Clear filters clicked');
      startDateInput.value = '';
      endDateInput.value = '';
      userInput.value = '';
      endpointInput.value = '';
      statusInput.value = '';
      loadLogs(); // Reload without filters
    });
  }

  // --- Initialize ---
  (async () => {
    const ok = await checkAuth();
    if (ok) loadLogs();
  })();
});
