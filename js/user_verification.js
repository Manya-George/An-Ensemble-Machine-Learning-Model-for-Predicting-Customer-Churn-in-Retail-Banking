const API_URL = 'http://localhost:5000/api';
const otpInputs = document.querySelectorAll('.otp-input');
let timeLeft = 60;
let countdownInterval;
let otpExpired = false;

// --- Helper Functions ---

function showToast(message) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span class="toast-icon">✓</span><span>${message}</span>`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

function startTimer() {
    const timerElement = document.getElementById('timer');
    otpExpired = false;
    if (countdownInterval) clearInterval(countdownInterval);

    countdownInterval = setInterval(() => {
        timeLeft--;
        timerElement.textContent = `The OTP will remain valid for ${formatTime(timeLeft)}`;
        if (timeLeft <= 0) {
            clearInterval(countdownInterval);
            timerElement.textContent = 'OTP has expired';
            timerElement.style.color = '#f44336';
            otpExpired = true;
        }
    }, 1000);
}

// --- On Load ---
window.addEventListener('load', () => {
    const otpSent = sessionStorage.getItem('otp_sent');
    const maskedEmail = sessionStorage.getItem('masked_email');
    const emailInfo = document.getElementById('emailInfo');

    if (!otpSent) {
        // No OTP sent — redirect to login
        window.location.href = 'login.html';
        return;
    }

    if (maskedEmail && emailInfo) {
        emailInfo.textContent = `An OTP has been sent to your email address ${maskedEmail}`;
    }

    showToast('OTP successfully sent, kindly check your email.');
    startTimer();
});

// --- OTP Input Navigation ---
otpInputs.forEach((input, index) => {
    input.addEventListener('input', (e) => {
        if (e.target.value.length === 1 && index < otpInputs.length - 1) {
            otpInputs[index + 1].focus();
        }
    });
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace' && !e.target.value && index > 0) {
            otpInputs[index - 1].focus();
        }
    });
    input.addEventListener('keypress', (e) => {
        if (!/[0-9]/.test(e.key)) e.preventDefault();
    });
});

// --- Verify OTP ---
document.getElementById('verifyBtn').addEventListener('click', async () => {
    const otp = Array.from(otpInputs).map(i => i.value).join('');
    const errorMsg = document.getElementById('errorMessage');
    const successMsg = document.getElementById('successMessage');
    const verifyBtn = document.getElementById('verifyBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');

    if (otp.length !== 6) {
        errorMsg.textContent = 'Please enter complete 6-digit OTP';
        errorMsg.style.display = 'block';
        return;
    }
    if (otpExpired) {
        errorMsg.textContent = 'OTP has expired. Please request a new one.';
        errorMsg.style.display = 'block';
        return;
    }

    verifyBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    try {
        const res = await fetch(`${API_URL}/verify-otp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ otp }),
        });

        const data = await res.json();

        if (data.success) {
            successMsg.textContent = 'OTP verified successfully! Redirecting...';
            successMsg.style.display = 'block';
            sessionStorage.removeItem('masked_email');
            sessionStorage.removeItem('otp_sent');

            setTimeout(() => {
                if (data.role === 'IT admin') {
                    window.location.href = 'IT/dashboard.html';
                } else if (data.role === 'Retail admin') {
                    window.location.href = 'retail_banking/dashboard.html';
                } else {
                    window.location.href = 'business_analyst/dashboard.html';
                }
            }, 1500);
        } else {
            errorMsg.textContent = data.message;
            errorMsg.style.display = 'block';
        }
    } catch (err) {
        console.error('Verification error:', err);
        errorMsg.textContent = 'Connection error. Please try again.';
        errorMsg.style.display = 'block';
    } finally {
        verifyBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

// --- Resend OTP ---
document.getElementById('resendLink').addEventListener('click', async (e) => {
    e.preventDefault();
    const resendLink = document.getElementById('resendLink');
    resendLink.textContent = 'Sending...';

    try {
        const res = await fetch(`${API_URL}/resend-otp`, { method: 'POST', credentials: 'include' });
        const data = await res.json();

        if (data.success) {
            showToast('OTP successfully sent, kindly check your email.');
            otpInputs.forEach(input => input.value = '');
            otpInputs[0].focus();
            clearInterval(countdownInterval);
            timeLeft = 60;
            document.getElementById('timer').textContent = 'The OTP will remain valid for 1:00';
            otpExpired = false;
            startTimer();
        } else {
            document.getElementById('errorMessage').textContent = data.message;
            document.getElementById('errorMessage').style.display = 'block';
        }
    } catch (err) {
        console.error('Resend OTP error:', err);
        document.getElementById('errorMessage').textContent = 'Failed to resend OTP. Please try again.';
        document.getElementById('errorMessage').style.display = 'block';
    } finally {
        resendLink.textContent = 'Resend OTP';
    }
});
