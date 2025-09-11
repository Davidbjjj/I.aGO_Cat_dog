document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.querySelector('.preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resultContainer = document.querySelector('.result-container');
    const resultIcon = document.getElementById('result-icon');
    const resultText = document.getElementById('result-text');
    const confidenceText = document.getElementById('confidence-text');
    const loading = document.querySelector('.loading');
    const errorDiv = document.querySelector('.error');
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });
    
    function handleFile(file) {
        // Reset previous results
        hideError();
        hideResult();
        
        // Check if file is an image
        if (!file.type.match('image.*')) {
            showError('Por favor, selecione uma imagem (JPEG, PNG, etc.)');
            return;
        }
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            
            // Send for prediction
            predictImage(e.target.result);
        };
        reader.readAsDataURL(file);
    }
    
    function predictImage(imageData) {
        showLoading();
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.success) {
                showResult(data);
            } else {
                showError(data.error || 'Erro ao processar a imagem');
            }
        })
        .catch(error => {
            hideLoading();
            showError('Erro de conexão. Tente novamente.');
            console.error('Error:', error);
        });
    }
    
    function showResult(data) {
        resultContainer.className = 'result-container';
        resultContainer.classList.add(data.class + '-result');
        
        resultIcon.textContent = data.class === 'cat' ? '🐱' : '🐶';
        resultText.textContent = data.class_name;
        confidenceText.textContent = `Confiança: ${data.confidence}%`;
        
        resultContainer.style.display = 'block';
    }
    
    function hideResult() {
        resultContainer.style.display = 'none';
    }
    
    function showLoading() {
        loading.style.display = 'block';
    }
    
    function hideLoading() {
        loading.style.display = 'none';
    }
    
    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }
    
    function hideError() {
        errorDiv.style.display = 'none';
    }
});