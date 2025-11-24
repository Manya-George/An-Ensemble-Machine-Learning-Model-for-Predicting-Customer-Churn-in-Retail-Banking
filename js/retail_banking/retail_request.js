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
