const API_URL = 'http://localhost:5000/api';

document.addEventListener("DOMContentLoaded", () => {
    const alertsContainer = document.getElementById('alertsContainer');
    const tableBody = document.querySelector('#interpretabilityTable tbody');
    const loadingState = document.getElementById('loadingState');

    function showAlert(type, message, timeout = 8000) {
        if (!alertsContainer) return;
        const el = document.createElement('div');
        el.className = `alert ${type}`;
        el.innerHTML = `<strong>${type === 'error' ? 'Error' : type === 'warn' ? 'Warning' : 'OK'}</strong>&nbsp;&nbsp;<span>${message}</span>`;
        alertsContainer.prepend(el);
        if (timeout) setTimeout(() => el.remove(), timeout);
    }

    function getShapTag(shapValue) {
        if (shapValue >= 0.05) {
            return `<span class="influence-tag high">High (${shapValue.toFixed(4)})</span>`;
        }
        if (shapValue >= 0.02) {
            return `<span class="influence-tag moderate">Moderate (${shapValue.toFixed(4)})</span>`;
        }
        return `<span class="influence-tag low">Low (${shapValue.toFixed(4)})</span>`;
    }
    
    function getPValueColor(pValue) {
        if (pValue < 0.001) return `<span style="color: #0d6efd; font-weight: 700;">${pValue.toExponential(2)}</span>`;
        if (pValue < 0.05) return `<span style="color: #198754; font-weight: 600;">${pValue.toFixed(4)}</span>`;
        return `<span style="color: #6c757d;">${pValue.toFixed(4)}</span>`;
    }

    async function fetchInterpretabilityData() {
        if (loadingState) loadingState.style.display = 'block';
        tableBody.innerHTML = '';
        
        try {
            const res = await fetch(`${API_URL}/interpretability-data`, { credentials: 'include' });
            
            if (!res.ok) {
                showAlert('error', 'Failed to load model insights. Check backend API status.');
                throw new Error('API request failed');
            }
            
            const data = await res.json();

            data.feature_interpretability.forEach(feature => {
                const row = tableBody.insertRow();
                
                row.innerHTML = `
                    <td>${feature.feature_name}</td>
                    <td><span class="influence-tag low" style="background:#e9ecef; color:#495057;">${feature.feature_type}</span></td>
                    <td>${getShapTag(feature.mean_shap_value)}</td>
                    <td>${getPValueColor(feature.univariate_p_value)}</td>
                    <td>${feature.model_influence_narrative}</td>
                `;
            });

        } catch (error) {
            console.error('Fetch error:', error);
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center;">Failed to load data. Ensure model has been trained.</td></tr>';
            plotsGrid.innerHTML = '<p style="text-align: center; color: #dc3545;">Could not load visualizations.</p>';
        } finally {
            if (loadingState) loadingState.style.display = 'none';
        }
    }

    // Initialize the dashboard data load
    (async () => {
        // Ensure session is authenticated before loading data
        const res = await fetch(`${API_URL}/check-auth`, { credentials: 'include' });
        const authData = await res.json();
        
        if (authData.authenticated) {
             // Optional: Role check (ensure only BA/Admins see this)
             if (authData.role === 'Business analyst' || authData.role === 'IT admin') {
                fetchInterpretabilityData();
             } else {
                showAlert('error', 'Access Denied: Only analysts and IT staff can view this page.', 0);
             }
        }
    })();
});