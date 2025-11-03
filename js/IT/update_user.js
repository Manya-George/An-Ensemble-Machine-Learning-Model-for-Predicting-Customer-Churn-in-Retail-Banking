const API_URL = "http://localhost:5000/api";

// --- Elements ---
const lookupForm = document.getElementById("lookupForm");
const lookupSection = document.getElementById("lookupSection");
const updateSection = document.getElementById("updateSection");
const successMessage = document.getElementById("successMessage");
const errorMessage = document.getElementById("errorMessage");
const lookupBtn = document.getElementById("lookupBtn");

// --- LOOKUP USER ---
lookupForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const lookupUsername = document.getElementById("lookupUsername").value.trim().toLowerCase();

    successMessage.classList.remove("show");
    errorMessage.classList.remove("show");

    lookupBtn.innerHTML = '<span class="loading-spinner"></span> Searching...';
    lookupBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/lookup-user`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username: lookupUsername }),
        });

        const data = await response.json();

        if (response.ok && data.success) {
            const userData = data.user;

            // Populate form
            document.getElementById("username").value = userData.username;
            document.getElementById("email").value = userData.email;
            document.getElementById("role").value = userData.role;

            // Update header display
            const initials = userData.username.substring(0, 2).toUpperCase();
            document.getElementById("userAvatar").textContent = initials;
            document.getElementById("displayUsername").textContent = userData.username;
            document.getElementById("displayEmail").textContent = userData.role;

            lookupSection.classList.add("hidden");
            updateSection.classList.add("show");
        } else {
            errorMessage.classList.add("show");
            document.getElementById("errorText").textContent =
                data.message || "User not found.";
        }
    } catch (err) {
        console.error("Error looking up user:", err);
        errorMessage.classList.add("show");
        document.getElementById("errorText").textContent = "Server error. Please try again.";
    } finally {
        lookupBtn.innerHTML = "Lookup User";
        lookupBtn.disabled = false;
    }
});

// --- CHANGE USER ---
document.getElementById("changeUserBtn").addEventListener("click", function () {
    updateSection.classList.remove("show");
    lookupSection.classList.remove("hidden");
    document.getElementById("lookupUsername").value = "";
    document.getElementById("updateForm").reset();
    successMessage.classList.remove("show");
    errorMessage.classList.remove("show");
});

// --- PASSWORD TOGGLE ---
const togglePassword = document.getElementById("togglePassword");
const passwordInput = document.getElementById("password");
const eyeIcon = document.getElementById("eyeIcon");

togglePassword.addEventListener("click", function () {
    const type = passwordInput.getAttribute("type") === "password" ? "text" : "password";
    passwordInput.setAttribute("type", type);
    eyeIcon.classList.toggle("bi-eye");
    eyeIcon.classList.toggle("bi-eye-slash");
});

// --- VALIDATION ---
function validateUpdateForm() {
    let isValid = true;

    document.querySelectorAll(".form-input, .form-select").forEach((input) =>
        input.classList.remove("input-error")
    );
    document.querySelectorAll(".error-text").forEach((error) => error.classList.remove("show"));

    const email = document.getElementById("email");
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.value.trim())) {
        email.classList.add("input-error");
        document.getElementById("emailError").classList.add("show");
        isValid = false;
    }

    const password = document.getElementById("password");
    if (password.value.length > 0 && password.value.length < 8) {
        password.classList.add("input-error");
        document.getElementById("passwordError").classList.add("show");
        isValid = false;
    }

    const role = document.getElementById("role");
    if (role.value === "") {
        role.classList.add("input-error");
        document.getElementById("roleError").classList.add("show");
        isValid = false;
    }

    return isValid;
}

// --- UPDATE USER ---
const updateForm = document.getElementById("updateForm");
updateForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    successMessage.classList.remove("show");
    errorMessage.classList.remove("show");

    if (!validateUpdateForm()) {
        errorMessage.classList.add("show");
        document.getElementById("errorText").textContent =
            "Please fill in all required fields correctly.";
        return;
    }

    const formData = {
        username: document.getElementById("username").value.trim(),
        email: document.getElementById("email").value.trim(),
        role: document.getElementById("role").value,
    };

    const password = document.getElementById("password").value;
    if (password) formData.password = password;

    try {
        const response = await fetch(`${API_URL}/update-user`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData),
        });

        const result = await response.json();

        if (response.ok && result.success) {
            successMessage.classList.add("show");
            document.getElementById("password").value = "";
            setTimeout(() => successMessage.classList.remove("show"), 5000);
        } else {
            errorMessage.classList.add("show");
            document.getElementById("errorText").textContent =
                result.message || "Update failed.";
        }
    } catch (error) {
        console.error("Error updating user:", error);
        errorMessage.classList.add("show");
        document.getElementById("errorText").textContent =
            "Server error. Please try again.";
    }
});

// --- CANCEL BUTTON ---
document.getElementById("cancelBtn").addEventListener("click", function () {
    if (confirm("Are you sure you want to cancel? All changes will be lost.")) {
        updateSection.classList.remove("show");
        lookupSection.classList.remove("hidden");
        document.getElementById("lookupUsername").value = "";
        updateForm.reset();
        successMessage.classList.remove("show");
        errorMessage.classList.remove("show");
    }
});

// --- REVOKE ACCESS ---
document.getElementById("revokeBtn").addEventListener("click", async function () {
    const username = document.getElementById("username").value.trim();
    if (!username) return;

    if (!confirm(`Are you sure you want to revoke access for ${username}?`)) return;

    try {
        const response = await fetch(`${API_URL}/revoke-user`, {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username }),
        });

        const result = await response.json();

        if (response.ok && result.success) {
            alert(`Access revoked for ${username}.`);
            document.getElementById("updateForm").reset();
            document.getElementById("updateSection").classList.remove("show");
            document.getElementById("lookupSection").classList.remove("hidden");
        } else {
            alert(result.message || "Failed to revoke user.");
        }
    } catch (error) {
        console.error("Error revoking user:", error);
        alert("Server error. Please try again.");
    }
});
