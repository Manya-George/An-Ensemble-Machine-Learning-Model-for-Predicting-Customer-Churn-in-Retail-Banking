const API_URL = 'http://localhost:5000/api';

document.addEventListener("DOMContentLoaded", () => {
    const alertsContainer = document.getElementById('alertsContainer');
    const plotsGrid = document.getElementById('plotsGrid');
    const loadingState = document.getElementById('loadingState');

    function showAlert(type, message, timeout = 8000) {
        if (!alertsContainer) return;
        const el = document.createElement('div');
        el.className = `alert ${type}`;
        el.innerHTML = `<strong>${type === 'error' ? 'Error' : type === 'warn' ? 'Warning' : 'OK'}</strong>&nbsp;&nbsp;<span>${message}</span>`;
        alertsContainer.prepend(el);
        if (timeout) setTimeout(() => el.remove(), timeout);
    }

    async function fetchInterpretabilityData() {
        if (loadingState) loadingState.style.display = 'block';
        plotsGrid.innerHTML = '';
        
        try {
            const res = await fetch(`${API_URL}/interpretability-data`, { credentials: 'include' });
            
            if (!res.ok) {
                showAlert('error', 'Failed to load model insights. Check backend API status.');
                throw new Error('API request failed');
            }
            
            const data = await res.json();

            // Render Plots
            data.feature_interpretability
                .filter(f => f.plot_filepath && !f.feature_name.startsWith('FE_')) // Filter for original features that have plots
                .slice(0, 14) 
                .forEach(feature => {
                    const plotCard = document.createElement('div');
                    plotCard.className = 'plot-card';
                    
                    // Construct the API endpoint for the plot
                    const plotUrl = `${API_URL}/analytics-plots/${feature.feature_name}_churn_relationship.png`;
                    
                    plotCard.innerHTML = `
                        <img src="${plotUrl}" alt="${feature.feature_name} Relationship Plot" 
                             onerror="this.onerror=null; this.src='https://placehold.co/350x250/cccccc/333333?text=Plot+Unavailable';" 
                             loading="lazy">
                        <p style="margin-top: 10px; font-weight: 600; color: #374151;">${feature.feature_name} Churn Impact</p>
                    `;
                    plotsGrid.appendChild(plotCard);
                });
            
        } catch (error) {
            console.error('Fetch error:', error);
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