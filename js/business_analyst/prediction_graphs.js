// ../js/ba/prediction_graphs.js
const API_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const jobId = urlParams.get('jobId');

  const backLink = document.getElementById('backLink');
  const jobInfo = document.getElementById('jobInfo');
  const loadingState = document.getElementById('loadingState');
  const errorState = document.getElementById('errorState');
  const errorMessage = document.getElementById('errorMessage');
  const chartsContent = document.getElementById('chartsContent');

  let riskChart, factorsChart, probabilityChart;

  if (!jobId) {
    showError('No job ID provided');
    return;
  }

  // Set back link
  backLink.href = `prediction_insights.html?jobId=${jobId}`;

  // Load data and render charts
  loadGraphData();

  async function loadGraphData() {
    try {
      // Fetch all data (no pagination, no filters)
      const res = await fetch(
        `${API_URL}/ba/prediction-insights/${jobId}?page=1&per_page=999999&risk=all`,
        { credentials: 'include' }
      );
      const data = await res.json();

      if (!data.success) {
        throw new Error(data.message || 'Failed to load data');
      }

      if (!data.rows || data.rows.length === 0) {
        throw new Error('No data available for this job');
      }

      // Update UI
      jobInfo.textContent = `Job ID: ${jobId} | ${data.filename || 'Unknown file'} | ${data.total_predictions} customers`;
      
      // Update stats
      document.getElementById('statTotal').textContent = data.statistics.high_risk + data.statistics.medium_risk + data.statistics.low_risk;
      document.getElementById('statHigh').textContent = data.statistics.high_risk;
      document.getElementById('statMedium').textContent = data.statistics.medium_risk;
      document.getElementById('statLow').textContent = data.statistics.low_risk;

      // Show charts container
      loadingState.style.display = 'none';
      chartsContent.style.display = 'block';

      // Render charts
      renderRiskDoughnut(data.statistics);
      renderFactorsBar(data.overall_factor_counts);
      renderProbabilityHistogram(data.rows);

    } catch (error) {
      console.error('Error loading graph data:', error);
      showError(error.message);
    }
  }

  function showError(message) {
    loadingState.style.display = 'none';
    errorState.style.display = 'block';
    errorMessage.textContent = message;
  }

  function renderRiskDoughnut(statistics) {
    const ctx = document.getElementById('riskDoughnutChart');
    
    if (riskChart) riskChart.destroy();

    riskChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['High Risk', 'Medium Risk', 'Low Risk'],
        datasets: [{
          data: [
            statistics.high_risk,
            statistics.medium_risk,
            statistics.low_risk
          ],
          backgroundColor: [
            '#ef5350',
            '#ff9800',
            '#66bb6a'
          ],
          borderWidth: 0,
          hoverOffset: 15
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 20,
              font: {
                size: 14,
                family: "'Nunito Sans', sans-serif"
              },
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.parsed || 0;
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = ((value / total) * 100).toFixed(1);
                return `${label}: ${value} (${percentage}%)`;
              }
            }
          }
        },
        cutout: '65%'
      }
    });
  }

  function renderFactorsBar(factorCounts) {
    const ctx = document.getElementById('factorsBarChart');
    
    if (factorsChart) factorsChart.destroy();

    // Get top 10 factors
    const sorted = Object.entries(factorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    const labels = sorted.map(([name]) => name);
    const values = sorted.map(([, count]) => count);

    // Create gradient
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
    gradient.addColorStop(1, 'rgba(118, 75, 162, 0.8)');

    factorsChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Frequency',
          data: values,
          backgroundColor: gradient,
          borderRadius: 8,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return `Count: ${context.parsed.y}`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              font: {
                family: "'Nunito Sans', sans-serif"
              }
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          },
          x: {
            ticks: {
              font: {
                family: "'Nunito Sans', sans-serif"
              }
            },
            grid: {
              display: false
            }
          }
        }
      }
    });
  }

  function renderProbabilityHistogram(rows) {
    const ctx = document.getElementById('probabilityHistogram');
    
    if (probabilityChart) probabilityChart.destroy();

    // Create histogram bins
    const bins = [
      { label: '0-10%', min: 0, max: 0.1, count: 0, color: '#66bb6a' },
      { label: '10-20%', min: 0.1, max: 0.2, count: 0, color: '#66bb6a' },
      { label: '20-30%', min: 0.2, max: 0.3, count: 0, color: '#66bb6a' },
      { label: '30-40%', min: 0.3, max: 0.4, count: 0, color: '#9ccc65' },
      { label: '40-50%', min: 0.4, max: 0.5, count: 0, color: '#ffb74d' },
      { label: '50-60%', min: 0.5, max: 0.6, count: 0, color: '#ffa726' },
      { label: '60-70%', min: 0.6, max: 0.7, count: 0, color: '#ff9800' },
      { label: '70-80%', min: 0.7, max: 0.8, count: 0, color: '#ef6c00' },
      { label: '80-90%', min: 0.8, max: 0.9, count: 0, color: '#f4511e' },
      { label: '90-100%', min: 0.9, max: 1.0, count: 0, color: '#ef5350' }
    ];

    // Count customers in each bin
    rows.forEach(row => {
      const prob = row.churn_probability;
      const bin = bins.find(b => prob >= b.min && prob < b.max) || bins[bins.length - 1];
      bin.count++;
    });

    probabilityChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: bins.map(b => b.label),
        datasets: [{
          label: 'Number of Customers',
          data: bins.map(b => b.count),
          backgroundColor: bins.map(b => b.color),
          borderRadius: 8,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return `Customers: ${context.parsed.y}`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 1,
              font: {
                family: "'Nunito Sans', sans-serif"
              }
            },
            title: {
              display: true,
              text: 'Number of Customers',
              font: {
                size: 14,
                family: "'Nunito Sans', sans-serif"
              }
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            }
          },
          x: {
            ticks: {
              font: {
                family: "'Nunito Sans', sans-serif"
              }
            },
            title: {
              display: true,
              text: 'Churn Probability Range',
              font: {
                size: 14,
                family: "'Nunito Sans', sans-serif"
              }
            },
            grid: {
              display: false
            }
          }
        }
      }
    });
  }
});