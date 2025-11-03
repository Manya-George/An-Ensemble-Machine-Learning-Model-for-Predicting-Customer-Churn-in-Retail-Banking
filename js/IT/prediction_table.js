document.addEventListener('DOMContentLoaded', function () {
    const API_URL = 'http://localhost:5000/api';
    let allResults = [];
    let filteredResults = [];
    let currentPage = 1;
    const resultsPerPage = 20;

    // Get job ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    const jobId = urlParams.get('jobId');

    async function loadResults() {
        try {
            const response = await fetch(`${API_URL}/prediction-results/${jobId}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch results: ${response.status}`);
            }

            const data = await response.json();
            if (!data.success) throw new Error(data.message || 'Failed to load results');

            allResults = data.results;
            filteredResults = [...allResults];

            updateStatistics(data.statistics);
            document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();
            renderResults();
        } catch (error) {
            console.error('Error loading results:', error);
            document.getElementById('resultsBody').innerHTML = `
                <tr><td colspan="6" style="text-align: center; color: #f44336;">
                    Failed to load results. Please try again.
                </td></tr>
            `;
        }
    }

    // Highlight active sidebar link
    const menuItems = document.querySelectorAll('.menu-item');
    const currPage = window.location.pathname.split('/').pop();
    menuItems.forEach(item => {
        const linkPage = item.getAttribute('href');
        if (linkPage === currPage) item.classList.add('active');
    });

    function updateStatistics(stats) {
        document.getElementById('totalCustomers').textContent = stats.total || 0;
        document.getElementById('highRisk').textContent = stats.high_risk || 0;
        document.getElementById('lowRisk').textContent = stats.low_risk || 0;
        document.getElementById('avgScore').textContent = ((stats.avg_score || 0) * 100).toFixed(1) + '%';
    }

    function getRiskLevel(score) {
        if (score >= 0.7) return { level: 'high', label: 'High Risk' };
        if (score >= 0.4) return { level: 'medium', label: 'Medium Risk' };
        return { level: 'low', label: 'Low Risk' };
    }

    function getScoreColor(score) {
        if (score >= 0.7) return '#f44336';
        if (score >= 0.4) return '#ff9800';
        return '#4caf50';
    }

    function renderResults() {
        const start = (currentPage - 1) * resultsPerPage;
        const end = start + resultsPerPage;
        const pageResults = filteredResults.slice(start, end);

        const tbody = document.getElementById('resultsBody');

        if (!tbody) {
            console.error('❌ Table body element not found');
            return;
        }

        if (pageResults.length === 0) {
            tbody.innerHTML = `
                <tr><td colspan="6" style="text-align: center; padding: 40px;">
                    No results found
                </td></tr>
            `;
            return;
        }

        tbody.innerHTML = pageResults.map(result => {
            const risk = getRiskLevel(result.churn_probability);
            const scoreColor = getScoreColor(result.churn_probability);

            return `
                <tr>
                    <td>${result.customer_id}</td>
                    <td>${result.surname || 'N/A'}</td>
                    <td>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${result.churn_probability * 100}%; background: ${scoreColor}"></div>
                        </div>
                    </td>
                    <td><span class="churn-badge ${risk.level}">${risk.label}</span></td>
                    <td>${(result.churn_probability * 100).toFixed(1)}%</td>
                    <td style="font-size: 12px; color: #666;">${result.top_features ? result.top_features.slice(0, 3).join(', ') : 'N/A'}</td>
                </tr>
            `;
        }).join('');

        updatePagination();
    }

    function updatePagination() {
        const totalPages = Math.ceil(filteredResults.length / resultsPerPage);
        document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
        document.getElementById('prevPage').disabled = currentPage === 1;
        document.getElementById('nextPage').disabled = currentPage === totalPages;
    }

    function changePage(delta) {
        currentPage += delta;
        renderResults();
    }

    // Filters
    document.getElementById('riskFilter').addEventListener('change', applyFilters);
    document.getElementById('searchInput').addEventListener('input', applyFilters);
    document.getElementById('sortBy').addEventListener('change', applyFilters);

    function applyFilters() {
        const riskFilter = document.getElementById('riskFilter').value;
        const searchTerm = document.getElementById('searchInput').value.toLowerCase();
        const sortBy = document.getElementById('sortBy').value;

        filteredResults = allResults.filter(result => {
            const matchesRisk = riskFilter === 'all' || getRiskLevel(result.churn_probability).level === riskFilter;
            const matchesSearch = searchTerm === '' ||
                result.customer_id.toString().includes(searchTerm) ||
                (result.surname && result.surname.toLowerCase().includes(searchTerm));

            return matchesRisk && matchesSearch;
        });

        if (sortBy === 'score_desc') {
            filteredResults.sort((a, b) => b.churn_probability - a.churn_probability);
        } else if (sortBy === 'score_asc') {
            filteredResults.sort((a, b) => a.churn_probability - b.churn_probability);
        } else if (sortBy === 'id') {
            filteredResults.sort((a, b) => a.customer_id - b.customer_id);
        }

        currentPage = 1;
        renderResults();
    }

    function exportResults() {
        const csv = [
            ['Customer ID', 'Name', 'Churn Probability', 'Risk Level', 'Prediction'].join(','),
            ...filteredResults.map(r => [
                r.customer_id,
                r.surname || 'N/A',
                r.churn_probability,
                getRiskLevel(r.churn_probability).label,
                r.churn_prediction
            ].join(','))
        ].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `churn_predictions_${jobId}.csv`;
        a.click();
    }

    // Pagination buttons
    document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
    document.getElementById('nextPage').addEventListener('click', () => changePage(1));
    document.getElementById('exportBtn').addEventListener('click', exportResults);

    // Load results when page is ready
    if (jobId) {
        loadResults();
    } else {
        document.getElementById('resultsBody').innerHTML = `
            <tr><td colspan="6" style="text-align: center; color: #f44336;">
                No job ID provided
            </td></tr>
        `;
    }
});
