const API_URL = 'http://localhost:5000/api';

        // Password Toggle
        const togglePassword = document.getElementById('togglePassword');
        const passwordInput = document.getElementById('password');
        const eyeIcon = document.getElementById('eyeIcon');

        togglePassword.addEventListener('click', function() {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            
            if (type === 'password') {
                eyeIcon.classList.remove('bi-eye-slash');
                eyeIcon.classList.add('bi-eye');
            } else {
                eyeIcon.classList.remove('bi-eye');
                eyeIcon.classList.add('bi-eye-slash');
            }
        });

        // Form Validation and Submission
        const registrationForm = document.getElementById('registrationForm');
        const successMessage = document.getElementById('successMessage');
        const errorMessage = document.getElementById('errorMessage');
        const cancelBtn = document.getElementById('cancelBtn');

        function validateForm() {
            let isValid = true;
            
            // Reset error states
            document.querySelectorAll('.form-input, .form-select').forEach(input => {
                input.classList.remove('input-error');
            });
            document.querySelectorAll('.error-text').forEach(error => {
                error.classList.remove('show');
            });

            // Username validation
            const username = document.getElementById('username');
            if (username.value.trim().length < 3) {
                username.classList.add('input-error');
                document.getElementById('usernameError').classList.add('show');
                isValid = false;
            }

            // Email validation
            const email = document.getElementById('email');
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email.value.trim())) {
                email.classList.add('input-error');
                document.getElementById('emailError').classList.add('show');
                isValid = false;
            }

            // Password validation
            const password = document.getElementById('password');
            if (password.value.length < 8) {
                password.classList.add('input-error');
                document.getElementById('passwordError').classList.add('show');
                isValid = false;
            }

            // Role validation
            const role = document.getElementById('role');
            if (role.value === '') {
                role.classList.add('input-error');
                document.getElementById('roleError').classList.add('show');
                isValid = false;
            }

            return isValid;
        }

        // Form Submission
        registrationForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            successMessage.classList.remove('show');
            errorMessage.classList.remove('show');

            if (validateForm()) {
                const formData = {
                    username: document.getElementById('username').value.trim(),
                    email: document.getElementById('email').value.trim(),
                    password: document.getElementById('password').value,
                    role: document.getElementById('role').value
                };

                try {
                    const response = await fetch(`${API_URL}/register-user`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });

                    const result = await response.json();
                    console.log(result);

                    if (response.ok && result.success) {
                        successMessage.classList.add('show');
                        registrationForm.reset();
                        setTimeout(() => successMessage.classList.remove('show'), 5000);
                    } else {
                        document.getElementById('errorText').textContent = result.message || 'Registration failed.';
                        errorMessage.classList.add('show');
                        setTimeout(() => errorMessage.classList.remove('show'), 5000);
                    }
                } catch (error) {
                    console.error('Error registering user:', error);
                    document.getElementById('errorText').textContent = 'Server error. Please try again.';
                    errorMessage.classList.add('show');
                    setTimeout(() => errorMessage.classList.remove('show'), 5000);
                }
            } else {
                errorMessage.classList.add('show');
                setTimeout(() => errorMessage.classList.remove('show'), 5000);
            }
        });

        // Cancel button
        cancelBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to cancel? All entered data will be lost.')) {
                registrationForm.reset();
                successMessage.classList.remove('show');
                errorMessage.classList.remove('show');
            }
        });

        // Real-time validation on blur
        document.querySelectorAll('.form-input, .form-select').forEach(input => {
            input.addEventListener('blur', function() {
                if (this.value.trim() !== '') {
                    validateForm();
                }
            });
        });