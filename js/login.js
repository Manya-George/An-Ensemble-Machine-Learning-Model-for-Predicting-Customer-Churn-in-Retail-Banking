const API_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();
            const errorMsg = document.getElementById('errorMessage');
            const loginBtn = document.getElementById('loginBtn');
            const btnText = document.getElementById('btnText');
            const btnLoader = document.getElementById('btnLoader');

            // Hide errors
            errorMsg.style.display = 'none';

            // Show loader
            loginBtn.disabled = true;
            btnText.style.display = 'none';
            btnLoader.style.display = 'inline';

            try {
                const response = await fetch(`${API_URL}/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ username, password }),
                });

                const data = await response.json();

                if (data.success) {
                    sessionStorage.setItem('masked_email', data.masked_email);
                    sessionStorage.setItem('otp_sent', 'true');
                    // Redirect to verification
                    window.location.href = 'user_verification.html';
                } else {
                    errorMsg.textContent = data.message;
                    errorMsg.style.display = 'block';
                }
            } catch (err) {
                console.error('Login error:', err);
                errorMsg.textContent = 'Connection error. Please try again.';
                errorMsg.style.display = 'block';
            } finally {
                loginBtn.disabled = false;
                btnText.style.display = 'inline';
                btnLoader.style.display = 'none';
            }
        });
    }
});
