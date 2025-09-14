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
    
    let model, maxPredictions;
    
    // Inicializa o modelo
    initModel();
    
    async function initModel() {
        try {
            showLoading();
            
            // Carrega o modelo (ajuste os caminhos conforme necessário)
            const modelURL = '/model/model.json';
            const metadataURL = '/model/metadata.json';
            
            model = await tmImage.load(modelURL, metadataURL);
            maxPredictions = model.getTotalClasses();
            
            hideLoading();
            console.log('Modelo carregado com sucesso!');
        } catch (error) {
            hideLoading();
            console.error('Erro ao carregar o modelo:', error);
            showError('Erro ao carregar o modelo de classificação.');
        }
    }
    
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
            predictImage(imagePreview);
        };
        reader.readAsDataURL(file);
    }
    
    async function predictImage(image) {
        if (!model) {
            showError('Modelo ainda não foi carregado. Tente novamente em alguns instantes.');
            return;
        }
        
        showLoading();
        
        try {
            // Faz a predição
            const prediction = await model.predict(image);
            
            // Encontra a classe com maior probabilidade
            let highestProb = 0;
            let predictedClass = '';
            
            for (let i = 0; i < maxPredictions; i++) {
                if (prediction[i].probability > highestProb) {
                    highestProb = prediction[i].probability;
                    predictedClass = prediction[i].className;
                }
            }
            
            // Formata os resultados
            const confidence = (highestProb * 100).toFixed(2);
            const className = predictedClass === 'cat' ? 'Gato' : 'Cachorro';
            
            // Exibe o resultado
            hideLoading();
            showResult({
                class: predictedClass,
                class_name: `É um ${className}!`,
                confidence: confidence
            });

            // Também envia para a API (opcional)
            // await sendToApi(fileInput.files[0], predictedClass, confidence);
            
        } catch (error) {
            hideLoading();
            console.error('Erro na predição:', error);
            showError('Erro ao processar a imagem. Tente novamente.');
        }
    }

    // Função para enviar para a API (opcional)
    async function sendToApi(file, className, confidence) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('className', className);
        formData.append('confidence', confidence);

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            console.log('Resposta da API:', data);
        } catch (error) {
            console.error('Erro ao enviar para API:', error);
        }
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