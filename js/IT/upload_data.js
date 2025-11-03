const API_URL = 'http://localhost:5000/api';
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const uploadBtn = document.getElementById('uploadBtn');
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const statusMessage = document.getElementById('statusMessage');
        let selectedFile = null;

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // File selection
        fileInput.addEventListener('change', handleFileSelect);
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.csv')) {
                fileInput.files = files;
                handleFileSelect({ target: { files: files } });
            }
        });
        
        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            if (!file.name.endsWith('.csv')) {
                showStatus('Please select a CSV file', 'error');
                return;
            }
            
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            fileInfo.style.display = 'block';
            uploadBtn.disabled = false;
            statusMessage.style.display = 'none';
        }
        
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
        }
        
        function showStatus(message, type) {
            statusMessage.textContent = message;
            statusMessage.className = 'status-message ' + type;
            statusMessage.style.display = 'block';
        }
        
        function updateProgress(percentage, text) {
            progressBar.style.width = percentage + '%';
            progressBar.textContent = percentage + '%';
            progressText.textContent = text;
        }
        
        uploadBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            uploadBtn.disabled = true;
            progressContainer.style.display = 'block';
            statusMessage.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                updateProgress(10, 'Uploading file...');
                
                const response = await fetch(`${API_URL}/predict-churn`, {
                    method: 'POST',
                    credentials: 'include',
                    body: formData
                });
                
                updateProgress(50, 'Processing data...');
                
                if (!response.ok) {
                    throw new Error('Upload failed');
                }
                
                const result = await response.json();
                
                updateProgress(100, 'Prediction complete!');
                
                setTimeout(() => {
                    showStatus('Prediction completed successfully! Redirecting to results...', 'success');
                    setTimeout(() => {
                        window.location.href = `prediction_results.html?jobId=${result.job_id}`;
                    }, 1500);
                }, 500);
                
            } catch (error) {
                console.error('Upload error:', error);
                showStatus('Failed to process file. Please try again.', 'error');
                progressContainer.style.display = 'none';
                uploadBtn.disabled = false;
            }
        });
