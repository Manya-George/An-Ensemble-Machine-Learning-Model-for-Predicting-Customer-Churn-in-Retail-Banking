const API_URL = 'http://localhost:5000/api';

const searchBtn = document.querySelector('.searchJobBtn');
const clearBtn = document.querySelector('.clearSearchBtn');
const jobIdInput = document.getElementById('jobIdSearch');
const searchMessage = document.getElementById('searchMessage');

// Store the current job data globally
let currentJobData = null;

// ---------------- JOB SEARCH FUNCTIONALITY ---------------- //
searchBtn.addEventListener('click', function (e) {
    e.preventDefault();

    const jobID = jobIdInput.value.trim();

    if (!jobID) {
        showMessage('Please enter a Job ID', 'error');
        return;
    }

    if (isNaN(jobID) || jobID <= 0) {
        showMessage('Please enter a valid Job ID number', 'error');
        return;
    }

    loadResults(jobID);
});

// Clear button handler
clearBtn.addEventListener('click', function (e) {
    e.preventDefault();
    jobIdInput.value = '';
    searchMessage.style.display = 'none';
    currentJobData = null;

    document.getElementById('totalCustomers').textContent = '0';
    document.getElementById('highRisk').textContent = '0';
    document.getElementById('lowRisk').textContent = '0';
    document.getElementById('avgScore').textContent = '0%';
    document.getElementById('timestamp').textContent = 'N/A';
});

// Helper message function
function showMessage(message, type) {
    searchMessage.textContent = message;
    searchMessage.style.display = 'block';

    if (type === 'error') {
        searchMessage.style.backgroundColor = '#fee2e2';
        searchMessage.style.color = '#991b1b';
        searchMessage.style.border = '1px solid #fecaca';
    } else if (type === 'success') {
        searchMessage.style.backgroundColor = '#d1fae5';
        searchMessage.style.color = '#065f46';
        searchMessage.style.border = '1px solid #6ee7b7';
    } else {
        searchMessage.style.backgroundColor = '#F59E0B';
        searchMessage.style.color = '#FFFFFF';
        searchMessage.style.border = '1px solid #F8D8A1';
    }
}

// Load job results
async function loadResults(jobID) {
    try {
        showMessage(`Loading results for Job ID: ${jobID}...`, 'info');

        const response = await fetch(`${API_URL}/prediction-results/${jobID}`, {
            credentials: 'include'
        });

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Job not found. Please check the Job ID and try again.');
            } else if (response.status === 401) {
                throw new Error('Session expired. Please login again.');
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.message || 'Failed to load results');
        }

        currentJobData = { jobId: jobID, ...data };
        updateStatistics(data.statistics);
        document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();

        showMessage('Results loaded successfully!', 'success');
        console.log('Loaded job data:', currentJobData);
    } catch (error) {
        console.error('Error loading results:', error);
        showMessage(error.message || 'Failed to load results. Please try again.', 'error');
        currentJobData = null;

        if (error.message.includes('Session expired')) {
            setTimeout(() => (window.location.href = "login.html"), 1500);
        }
    }
}

// Update statistics UI
function updateStatistics(stats) {
    document.getElementById('totalCustomers').textContent = stats.total || 0;
    document.getElementById('highRisk').textContent = stats.high_risk || 0;
    document.getElementById('lowRisk').textContent = stats.low_risk || 0;
    document.getElementById('avgScore').textContent = ((stats.avg_score || 0) * 100).toFixed(1) + '%';
}

// Export CSV
async function exportQuickResults() {
    if (!currentJobData || !currentJobData.jobId) {
        alert('Please search for a job first');
        return;
    }

    const jobID = currentJobData.jobId;

    try {
        showMessage('Exporting results...', 'info');

        const response = await fetch(`${API_URL}/prediction-results/${jobID}`, {
            credentials: 'include'
        });

        if (!response.ok) throw new Error('Failed to fetch results for export');

        const data = await response.json();

        if (!data.success || !data.results) throw new Error('No results available to export');

        const results = data.results;
        const csv = [
            ['Customer ID', 'Name', 'Churn Probability', 'Risk Level', 'Prediction'].join(','),
            ...results.map(r => [
                r.customer_id,
                r.surname || 'N/A',
                (r.churn_probability * 100).toFixed(2) + '%',
                r.churn_probability >= 0.7 ? 'High Risk' :
                    r.churn_probability >= 0.4 ? 'Medium Risk' : 'Low Risk',
                r.churn_prediction ? 'Yes' : 'No'
            ].join(','))
        ].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `churn_predictions_${jobID}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showMessage('Results exported successfully!', 'success');
    } catch (error) {
        console.error('Error exporting:', error);
        alert('Failed to export results: ' + error.message);
    }
}

// View Table
const viewTableBtn = document.getElementById('viewTableBtn');
const viewTableBtn2 = document.getElementById('viewTableBtn2');

if (viewTableBtn) {
    viewTableBtn.addEventListener('click', () => {
        if (!currentJobData || !currentJobData.jobId) {
            alert('Please search for a job first');
            return;
        }
        window.location.href = `prediction_table.html?jobId=${currentJobData.jobId}`;
    });
}

if (viewTableBtn2) {
    viewTableBtn2.addEventListener('click', () => {
        if (!currentJobData || !currentJobData.jobId) {
            alert('Please search for a job first');
            return;
        }
        window.location.href = `prediction_table.html?jobId=${currentJobData.jobId}`;
    });
}

// Handle URL param jobId
const urlParams = new URLSearchParams(window.location.search);
const urlJobId = urlParams.get('jobId');
if (urlJobId) {
    jobIdInput.value = urlJobId;
    loadResults(urlJobId);
}
