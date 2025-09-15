document.addEventListener("DOMContentLoaded", function () {
  const uploadArea = document.getElementById("upload-area");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.querySelector(".preview-container");
  const imagePreview = document.getElementById("image-preview");
  const resultContainer = document.querySelector(".result-container");
  const resultIcon = document.getElementById("result-icon");
  const resultText = document.getElementById("result-text");
  const confidenceText = document.getElementById("confidence-text");
  const loading = document.querySelector(".loading");
  const errorDiv = document.querySelector(".error");
  const actionsDiv = document.querySelector(".actions");
  const classifyBtn = document.getElementById("classify-btn");
  const detectBtn = document.getElementById("detect-btn");
  const segmentBtn = document.getElementById("segment-btn");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const segCanvas = document.getElementById("segmentation-canvas");
  const segCtx = segCanvas.getContext("2d");

  let model;
  let detectionModel;
  let segmentationModel = null;
  let labels = ["Gato", "Cachorro"];
  let metadata = {};
  let currentImage = null;
  let currentMode = null;

  // Inicializa os modelos
  initModel();
  loadDetectionModel();
  // Removemos o carregamento automático do modelo de segmentação

  async function initModel() {
    try {
      showLoading();
      model = await tf.loadLayersModel("model/model.json");

      try {
        const metadataResponse = await fetch("model/metadata.json");
        metadata = await metadataResponse.json();
        if (metadata.labels) labels = metadata.labels;
      } catch (e) {
        console.warn("Metadados não encontrados, usando padrão.");
      }

      console.log("Modelo carregado:", metadata);
      hideLoading();
    } catch (error) {
      hideLoading();
      console.error("Erro ao carregar o modelo:", error);
      showError("Erro ao carregar o modelo de classificação.");
    }
  }

  async function loadDetectionModel() {
    try {
      detectionModel = await cocoSsd.load();
      console.log("Modelo de detecção carregado!");
    } catch (error) {
      console.error("Erro ao carregar modelo de detecção:", error);
      showError("Erro ao carregar modelo de detecção.");
    }
  }

  // Função para converter RGB para HSL
  function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
      h = s = 0; 
    } else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch(max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }
      h /= 6;
    }
    return [h * 360, s * 100, l * 100];
  }

  // Segmentação simples baseada em tons de pele (exemplo)
  async function performSimpleSegmentation(imageElement) {
    return new Promise((resolve) => {
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      tempCanvas.width = imageElement.naturalWidth || imageElement.width;
      tempCanvas.height = imageElement.naturalHeight || imageElement.height;

      tempCtx.drawImage(imageElement, 0, 0, tempCanvas.width, tempCanvas.height);
      const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
      const data = imageData.data;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i+1], b = data[i+2];
        const [h, s, l] = rgbToHsl(r, g, b);

        // Exemplo: destaca tons de pele
        if (h >= 0 && h <= 50 && s >= 20 && l >= 20) {
          data[i] = 255; data[i+1] = 200; data[i+2] = 150; data[i+3] = 220;
        } else {
          data[i+3] = 60; // deixa o resto mais transparente
        }
      }

      resolve({
        data: imageData,
        width: tempCanvas.width,
        height: tempCanvas.height
      });
    });
  }

  // Carregamento e uso do modelo BodyPix
  let bodyPixModel;

  async function loadBodyPixModel() {
    if (!bodyPixModel) {
      bodyPixModel = await bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 0.75,
        quantBytes: 2
      });
    }
    return bodyPixModel;
  }

  async function performBodyPixSegmentation(imageElement) {
    const net = await loadBodyPixModel();
    const segmentation = await net.segmentPerson(imageElement);

    const maskBackground = true;
    const backgroundColor = { r: 0, g: 0, b: 0, a: 0 }; // transparente
    const foregroundColor = { r: 0, g: 255, b: 0, a: 255 }; // verde

    const coloredPartImage = bodyPix.toMask(segmentation, foregroundColor, backgroundColor);

    segCtx.putImageData(coloredPartImage, 0, 0);
  }

  // Drag & Drop
  uploadArea.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", function () {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadArea.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  uploadArea.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  function handleFile(file) {
    hideError();
    hideResult();
    canvas.style.display = "none";
    segCanvas.style.display = "none";
    currentMode = null;

    if (!file.type.match("image.*")) {
      showError("Por favor, selecione uma imagem (JPEG, PNG, etc.)");
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.onload = () => {
        previewContainer.style.display = "block";
        actionsDiv.style.display = "flex";
        currentImage = imagePreview;

        imagePreview.style.display = "block";
        canvas.style.display = "none";
        segCanvas.style.display = "none";
      };
      imagePreview.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  classifyBtn.addEventListener("click", function() {
    if (currentImage) {
      canvas.style.display = "none";
      segCanvas.style.display = "none";
      imagePreview.style.display = "block";
      currentMode = 'classification';
      predictImage(currentImage);
    }
  });

  detectBtn.addEventListener("click", async function() {
    if (!currentImage) {
      showError("Carregue uma imagem primeiro!");
      return;
    }
    if (!detectionModel) {
      showError("Modelo de detecção ainda não foi carregado!");
      return;
    }

    imagePreview.style.display = "none";
    segCanvas.style.display = "none";
    currentMode = 'detection';

    hideResult();

    // Redimensiona a imagem para garantir compatibilidade mobile
    const resizedImage = getResizedImageElement(currentImage, 640);

    canvas.width = resizedImage.width;
    canvas.height = resizedImage.height;
    canvas.style.display = "block";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(resizedImage, 0, 0, canvas.width, canvas.height);

    showLoading();
    try {
      const predictions = await detectionModel.detect(resizedImage);
      console.log(predictions); // Veja no console do mobile (inspecione pelo PC)
      hideLoading();

      let detectedObjects = [];

      predictions.forEach((pred) => {
        if (pred.class === "cat" || pred.class === "dog") {
          const [x, y, width, height] = pred.bbox;
          const confidence = (pred.score * 100).toFixed(1);

          ctx.strokeStyle = pred.class === "cat" ? "#00f2fe" : "#f5576c";
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);

          const labelPt = pred.class === "cat" ? "Gato 🐱" : "Cachorro 🐶";

          ctx.fillStyle = pred.class === "cat" ? "#00f2fe" : "#f5576c";
          ctx.font = "18px Arial";
          ctx.fillText(
            `${labelPt} (${confidence}%)`,
            x,
            y > 20 ? y - 5 : y + 20
          );

          detectedObjects.push(`${labelPt} (${confidence}%)`);
        }
      });

      if (detectedObjects.length > 0) {
        showDetectionResult(detectedObjects);
      } else {
        showDetectionResult([]);
      }
    } catch (error) {
      hideLoading();
      console.error("Erro na detecção:", error);
      showError("Erro ao processar a imagem. Tente novamente.");
    }
  });

  segmentBtn.addEventListener("click", async function() {
    if (!currentImage) {
      showError("Carregue uma imagem primeiro!");
      return;
    }

    imagePreview.style.display = "none";
    canvas.style.display = "none";
    currentMode = 'segmentation';

    hideResult();

    segCanvas.width = currentImage.naturalWidth || currentImage.width;
    segCanvas.height = currentImage.naturalHeight || currentImage.height;
    segCanvas.style.display = "block";
    segCtx.clearRect(0, 0, segCanvas.width, segCanvas.height);

    showLoading();
    try {
      const segmentationResult = await performSimpleSegmentation(currentImage);
      segCtx.putImageData(segmentationResult.data, 0, 0);
      showSegmentationResult();
    } catch (error) {
      showError("Erro ao processar a imagem. Tente novamente.");
    } finally {
      hideLoading();
    }
  });

  function toggleView(mode) {
    if (mode === 'classification') {
      imagePreview.style.display = "block";
      canvas.style.display = "none";
      segCanvas.style.display = "none";
    } else if (mode === 'detection') {
      imagePreview.style.display = "none";
      canvas.style.display = "block";
      segCanvas.style.display = "none";
    } else if (mode === 'segmentation') {
      imagePreview.style.display = "none";
      canvas.style.display = "none";
      segCanvas.style.display = "block";
    }
  }

  function showResult(data) {
    toggleView('classification');
    resultContainer.className = "result-container";
    resultContainer.classList.add(data.class + "-result");

    resultIcon.textContent = data.class === "cat" ? "🐱" : "🐶";
    resultText.textContent = data.class_name;
    confidenceText.textContent = `Confiança: ${data.confidence}%`;

    resultContainer.style.display = "block";
  }

  function showDetectionResult(detectedObjects) {
    toggleView('detection');
    resultContainer.className = "result-container";

    if (detectedObjects.length > 0) {
      resultContainer.classList.add("detection-result");
      resultIcon.textContent = "🔎";
      resultText.textContent = "Detectados: " + detectedObjects.join(" | ");
      confidenceText.textContent = "";
    } else {
      resultIcon.textContent = "⚠️";
      resultText.textContent = "Nenhum gato ou cachorro detectado.";
      confidenceText.textContent = "";
    }

    resultContainer.style.display = "block";
  }

  function showSegmentationResult() {
    toggleView('segmentation');
    resultContainer.className = "result-container";
    resultContainer.classList.add("segmentation-result");

    resultIcon.textContent = "🔍";
    resultText.textContent = "Imagem segmentada com sucesso!";
    confidenceText.textContent = "Áreas destacadas mostram regiões segmentadas";

    resultContainer.style.display = "block";
  }

  function hideResult() {
    resultContainer.style.display = "none";
  }

  function showLoading() {
    loading.style.display = "block";
  }

  function hideLoading() {
    loading.style.display = "none";
  }

  function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = "block";
  }

  function hideError() {
    errorDiv.style.display = "none";
  }

  function getResizedImageElement(imageElement, maxSize = 640) {
    const canvas = document.createElement('canvas');
    let width = imageElement.naturalWidth || imageElement.width;
    let height = imageElement.naturalHeight || imageElement.height;
    if (width > maxSize || height > maxSize) {
      if (width > height) {
        height = Math.round(height * (maxSize / width));
        width = maxSize;
      } else {
        width = Math.round(width * (maxSize / height));
        height = maxSize;
      }
    }
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').drawImage(imageElement, 0, 0, width, height);
    return canvas;
  }

  async function predictImage(imageElement) {
    if (!model) {
      showError('Modelo ainda não foi carregado. Tente novamente em alguns instantes.');
      return;
    }

    showLoading();

    try {
      const tensor = tf.tidy(() => {
        let img = tf.browser.fromPixels(imageElement)
          .resizeNearestNeighbor([metadata.imageSize || 224, metadata.imageSize || 224])
          .toFloat()
          .div(255.0)
          .expandDims(0);
        return img;
      });

      const predictions = await model.predict(tensor).data();
      tensor.dispose();

      const maxIndex = predictions.indexOf(Math.max(...predictions));
      const label = (metadata.labels && metadata.labels[maxIndex]) || labels[maxIndex] || `Classe ${maxIndex}`;
      const confidence = (predictions[maxIndex] * 100).toFixed(2);

      const labelMap = { cat: "Gato", dog: "Cachorro" };
      const className = labelMap[label] || label;

      hideLoading();
      showResult({
        class: label.toLowerCase(),
        class_name: `É um ${className}!`,
        confidence: confidence
      });

    } catch (error) {
      hideLoading();
      console.error('Erro na predição:', error);
      showError('Erro ao processar a imagem. Tente novamente.');
    }
  }
});