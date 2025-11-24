// js/retail_insights_request.js
const API_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {

    const requestForm = document.getElementById('requestForm');
    const descriptionEl = document.getElementById('description');
    const jobIdEl = document.getElementById('jobId');
    const successMessage = document.getElementById('successMessage');
    const successText = document.getElementById('successText');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // -----------------------------
    // UI HELPER FUNCTIONS
    // -----------------------------
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

    function escapeHtml(s) {
        if (!s) return '';
        return s.replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
    }

    // -----------------------------
    // FORM SUBMISSION
    // -----------------------------
    requestForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const description = descriptionEl.value.trim();
        const jobID = parseInt(jobIdEl.value.trim(), 10);

        if (!description) {
            showError('Description is required');
            return;
        }
        if (!jobID || isNaN(jobID)) {
            showError('Valid Job ID is required');
            return;
        }

        try {
            const res = await fetch(`${API_URL}/insights-requests`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    jobID,
                    description
                })
            });

            const j = await res.json();

            if (j.success) {
                showSuccess(`Insights request submitted successfully — ID: ${j.insightID}`);
                descriptionEl.value = '';
                jobIdEl.value = '';
                await loadMyInsightsRequests();
            } else {
                showError(j.message || 'Failed to submit insights request');
            }

        } catch (err) {
            console.error('submit insights request', err);
            showError('Connection error — request could not be submitted');
        }
    });

    // -----------------------------
    // Cancel button
    // -----------------------------
    document.getElementById('cancelBtn').addEventListener('click', () => {
        if (confirm('Cancel this insights request?')) {
            descriptionEl.value = '';
            jobIdEl.value = '';
        }
    });

    // -----------------------------
    // INITIAL LOAD
    // -----------------------------
    loadMyInsightsRequests();

});
