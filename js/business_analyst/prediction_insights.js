// ../js/ba/prediction_insights.js - Enhanced with medium risk support
const API_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {
  const inputJobId = document.getElementById('inputJobId');
  const btnLoad = document.getElementById('btnLoad');
  const btnViewGraphs = document.getElementById('btnViewGraphs');
  const riskFilter = document.getElementById('riskFilter');
  const factorSelect = document.getElementById('factorSelect');
  const sortBy = document.getElementById('sortBy');
  const exportBtn = document.getElementById('exportCsv');
  const resultsBody = document.getElementById('resultsBody');
  const summaryTotal = document.getElementById('summaryTotal');
  const summaryHigh = document.getElementById('summaryHigh');
  const summaryMedium = document.getElementById('summaryMedium');
  const summaryLow = document.getElementById('summaryLow');
  const factorsList = document.getElementById('factorsList');
  const prevPage = document.getElementById('prevPage');
  const nextPage = document.getElementById('nextPage');
  const pageInfo = document.getElementById('pageInfo');

  let state = {
    job_id: null,
    page: 1,
    per_page: 50,
    total_pages: 0,
    rows: [],
    available_factors: [],
    overall_counts: {},
    filtered_counts: {},
    statistics: {}
  };

  async function loadJob() {
    const jobId = (inputJobId.value || '').trim();
    if (!jobId) {
      alert('Please enter a job ID');
      return;
    }
    state.job_id = Number(jobId);
    state.page = 1;
    await fetchAndRender();
  }

  async function fetchAndRender() {
    if (!state.job_id) return;
    const params = new URLSearchParams();
    params.set('page', state.page);
    params.set('per_page', state.per_page);
    if (riskFilter.value && riskFilter.value !== 'all') params.set('risk', riskFilter.value);
    if (factorSelect.value) params.set('factor', factorSelect.value);
    if (sortBy.value) params.set('sort', sortBy.value);

    try {
      const res = await fetch(`${API_URL}/ba/prediction-insights/${state.job_id}?${params.toString()}`, { 
        credentials: 'include' 
      });
      const j = await res.json();
      
      if (!j.success) {
        resultsBody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#f44336">${j.message || 'Failed to load'}</td></tr>`;
        btnViewGraphs.disabled = true;
        return;
      }

      state.page = j.page;
      state.per_page = j.per_page;
      state.total_pages = j.total_pages;
      state.rows = j.rows || [];
      state.available_factors = j.available_factors || [];
      state.overall_counts = j.overall_factor_counts || {};
      state.filtered_counts = j.filtered_factor_counts || {};
      state.statistics = j.statistics || {};
      state.total_predictions = j.total_predictions || 0;

      renderSummary();
      renderFactorOptions();
      renderTable();
      updatePagination();
      
      // FIXED: Enable button if we have any data for this job
      console.log('Total predictions:', state.total_predictions);
      btnViewGraphs.disabled = (state.total_predictions === 0);
      
    } catch (err) {
      console.error('fetchAndRender', err);
      resultsBody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#f44336">Connection error</td></tr>`;
      btnViewGraphs.disabled = true;
    }
  }

  function renderSummary() {
    // Use statistics from backend
    const stats = state.statistics;
    const total = stats.high_risk + stats.medium_risk + stats.low_risk;
    
    summaryTotal.textContent = `Total: ${total}`;
    summaryHigh.textContent = `High: ${stats.high_risk || 0}`;
    summaryMedium.textContent = `Medium: ${stats.medium_risk || 0}`;
    summaryLow.textContent = `Low: ${stats.low_risk || 0}`;

    // Top factors list (overall counts) - Make clickable to view graphs
    factorsList.innerHTML = '';
    const items = Object.entries(state.overall_counts || {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12);
    
    if (items.length === 0) {
      factorsList.innerHTML = '<span style="color:#666">No factor data available</span>';
      return;
    }
    
    items.forEach(([name, count]) => {
      const el = document.createElement('div');
      el.className = 'factor-pill';
      el.textContent = `${name} (${count})`;
      el.style.cursor = 'pointer';
      el.title = 'Click to filter by this factor';
      el.addEventListener('click', () => {
        factorSelect.value = name;
        state.page = 1;
        fetchAndRender();
      });
      factorsList.appendChild(el);
    });

    // Add a "View Graphs" pill at the end
    if (state.job_id) {
      const graphsPill = document.createElement('div');
      graphsPill.className = 'factor-pill';
      graphsPill.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
      graphsPill.style.color = 'white';
      graphsPill.style.fontWeight = '600';
      graphsPill.style.cursor = 'pointer';
      graphsPill.innerHTML = '<i class="bi bi-graph-up"></i> View All Graphs';
      graphsPill.title = 'Open visualization dashboard';
      graphsPill.addEventListener('click', () => {
        window.location.href = `prediction_graphs.html?jobId=${state.job_id}`;
      });
      factorsList.appendChild(graphsPill);
    }
  }

  function renderFactorOptions() {
    const prev = factorSelect.value || '';
    factorSelect.innerHTML = '<option value="">All Risk Factors</option>';
    (state.available_factors || []).forEach(f => {
      const opt = document.createElement('option');
      opt.value = f;
      opt.textContent = `${f} (${state.overall_counts[f] || 0})`;
      factorSelect.appendChild(opt);
    });
    if (prev) factorSelect.value = prev;
  }

  function getRiskClass(probability) {
      if (probability >= 0.7) return { level: 'high', label: 'High Risk' };
      if (probability >= 0.4) return { level: 'medium', label: 'Medium Risk' };
      return { level: 'low', label: 'Low Risk' };
  }

  function getScoreBarClass(probability) {
    if (probability >= 0.7) return 'high';
    if (probability >= 0.4) return 'medium';
    return 'low';
  }

  function renderTable() {
    if (!state.rows || state.rows.length === 0) {
      resultsBody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:24px;color:#666">No results for these filters</td></tr>`;
      return;
    }

    resultsBody.innerHTML = state.rows.map(r => {
      const prob = Number(r.churn_probability);
      const scorePct = (prob * 100).toFixed(1);
      const riskClass = getRiskClass(prob);
      const scoreBarClass = getScoreBarClass(prob);
      const topNames = (r.top_factor_names || []).slice(0, 3).join(', ') || 'N/A';
      const created = r.createdAt ? new Date(r.createdAt).toLocaleString() : '';
      const riskLabel = riskClass.label;

      return `
        <tr style="cursor: pointer;" 
            onmouseover="this.style.backgroundColor='#f8f9fa'" 
            onmouseout="this.style.backgroundColor=''"
            title="Customer ${escapeHtml(String(r.customer_id || ''))}">
          <td>${escapeHtml(String(r.customer_id || ''))}</td>
          <td>${escapeHtml(r.surname || 'N/A')}</td>
          <td>
            <div class="score-bar">
              <div class="score-fill ${scoreBarClass}" style="width: ${scorePct}%"></div>
            </div>
          </td>
          <td>
            <span class="badge-risk ${riskClass.level}">${riskClass.label}</span>
          </td>
          <td style="font-weight: 600;">${scorePct}%</td>
          <td style="font-size:13px">${escapeHtml(topNames)}</td>
          <td style="font-size:12px;color:#666">${created}</td>
        </tr>
      `;
    }).join('');
  }

  function updatePagination() {
    pageInfo.textContent = `Page ${state.page} of ${state.total_pages || 0}`;
    prevPage.disabled = state.page <= 1;
    nextPage.disabled = state.page >= (state.total_pages || 1);
  }

  function escapeHtml(s) {
    if (!s) return '';
    return String(s)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  // Event handlers
  btnLoad.addEventListener('click', loadJob);
  
  btnViewGraphs.addEventListener('click', () => {
    if (state.job_id) {
      window.location.href = `prediction_graphs.html?jobId=${state.job_id}`;
    }
  });

  riskFilter.addEventListener('change', async () => { 
    state.page = 1; 
    await fetchAndRender(); 
  });
  
  factorSelect.addEventListener('change', async () => { 
    state.page = 1; 
    await fetchAndRender(); 
  });
  
  sortBy.addEventListener('change', async () => { 
    state.page = 1; 
    await fetchAndRender(); 
  });

  prevPage.addEventListener('click', async () => {
    if (state.page > 1) { 
      state.page -= 1; 
      await fetchAndRender(); 
    }
  });
  
  nextPage.addEventListener('click', async () => {
    if (state.page < state.total_pages) { 
      state.page += 1; 
      await fetchAndRender(); 
    }
  });

  exportBtn.addEventListener('click', () => {
    if (!state.rows || state.rows.length === 0) { 
      alert('No rows to export'); 
      return; 
    }
    
    const csv = [
      ['CustomerID', 'Surname', 'ChurnProbability', 'RiskLevel', 'TopFactors', 'Prediction'].join(',')
    ].concat(state.rows.map(r => [
      `"${(r.customer_id || '')}"`,
      `"${(r.surname || '')}"`,
      (r.churn_probability || 0),
      (r.risk_level || ''),
      `"${(r.top_factor_names || []).join(';')}"`,
      (r.churn_prediction ? '1' : '0')
    ].join(',')));
    
    const blob = new Blob([csv.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prediction_insights_job_${state.job_id || 'unknown'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Auto-load from URL parameter
  (function initFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    const j = urlParams.get('jobId');
    if (j) { 
      inputJobId.value = j; 
      btnLoad.click(); 
    }
  })();
});