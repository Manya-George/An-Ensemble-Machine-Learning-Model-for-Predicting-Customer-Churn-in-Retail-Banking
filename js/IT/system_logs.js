const API_URL = 'http://localhost:5000/api';

async function checkAuthAndLoadLogs() {
    try {
        const res = await fetch(`${API_URL}/check-auth`, { credentials: 'include' });
        const data = await res.json();

        if (!data.authenticated) {
            alert('Session expired. Please log in again.');
            window.location.href = '../login.html';
            return;
        }

        document.querySelector('.user-name').textContent = data.username || 'IT Admin';
        loadLogs();
    } catch (err) {
        console.error('Auth check failed:', err);
        alert('Unable to verify session. Please log in again.');
        window.location.href = '../login.html';
    }
}

async function loadLogs() {
    try {
        const res = await fetch(`${API_URL}/system-logs`, { credentials: 'include' });
        const data = await res.json();

        if (!data.success) {
            alert('Failed to load logs: ' + data.message);
            return;
        }

        const tbody = document.querySelector('#logsTable tbody');
        tbody.innerHTML = '';
        data.logs.forEach(log => {
            const row = `
                <tr>
                    <td>${new Date(log.timestamp).toLocaleString()}</td>
                    <td>${log.username || 'N/A'}</td>
                    <td>${log.endpoint}</td>
                    <td>${log.method}</td>
                    <td>${log.action_description}</td>
                    <td>${log.status_code}</td>
                </tr>`;
            tbody.innerHTML += row;
        });
    } catch (err) {
        console.error('Error loading logs:', err);
        alert('Server error while loading logs.');
    }
}

checkAuthAndLoadLogs();
