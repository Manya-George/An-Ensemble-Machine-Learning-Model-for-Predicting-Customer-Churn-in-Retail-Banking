// Highlight active link based on current page URL
        const menuItems = document.querySelectorAll('.menu-item');
        const currentPage = window.location.pathname.split('/').pop(); // e.g., 'update_user.html'

        menuItems.forEach(item => {
            const linkPage = item.getAttribute('href');

            // Match the link's href with the current page filename
            if (linkPage === currentPage) {
            item.classList.add('active');
            }
        });

        // User Profile Dropdown Functionality
        const userProfileBtn = document.getElementById('userProfileBtn');
        const dropdownMenu = document.getElementById('dropdownMenu');
        
        // Toggle dropdown on click
        userProfileBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            dropdownMenu.classList.toggle('show');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!userProfileBtn.contains(e.target) && !dropdownMenu.contains(e.target)) {
                dropdownMenu.classList.remove('show');
            }
        });
        
        // Logout functionality
        const logoutBtn = document.getElementById('logoutBtn');
        logoutBtn.addEventListener('click', function(e) {
            e.preventDefault();
            // Add your logout logic here
            if (confirm('Are you sure you want to logout?')) {
                console.log('User logged out');
                // Redirect to login page or perform logout action
                // window.location.href = 'login.html';
                alert('Logout functionality - redirect to login page');
            }
        });
        
        // Dropdown menu items
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            item.addEventListener('click', function(e) {
                if (this.id !== 'logoutBtn') {
                    e.preventDefault();
                    const action = this.textContent.trim();
                    console.log('Dropdown action:', action);
                    dropdownMenu.classList.remove('show');
                    // Add your navigation logic here
                    // window.location.href = this.getAttribute('href');
                }
            });
        });