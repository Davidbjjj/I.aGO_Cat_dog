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
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  let model;
  let detectionModel;
  let labels = ["Gato", "Cachorro"];
  let metadata = {};
  let currentImage = null;
  let currentMode = null; // 'classification' ou 'detection'

  // Inicializa os modelos
  initModel();
  loadDetectionModel();

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

  // Modificado: reset de estado e visualização ao carregar novo arquivo
  function handleFile(file) {
    hideError();
    hideResult();
    canvas.style.display = "none";
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

        // Mostrar a imagem original ao carregar novo arquivo
        imagePreview.style.display = "block";
        canvas.style.display = "none";
      };
      imagePreview.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // Modificado: alterna visualização e modo ao classificar
  classifyBtn.addEventListener("click", function() {
    if (currentImage) {
      // Esconder resultados de detecção e mostrar imagem original
      canvas.style.display = "none";
      imagePreview.style.display = "block";
      currentMode = 'classification';
      predictImage(currentImage);
    }
  });

  // Modificado: alterna visualização e modo ao detectar
  detectBtn.addEventListener("click", async function() {
    if (!currentImage) {
      showError("Carregue uma imagem primeiro!");
      return;
    }
    if (!detectionModel) {
      showError("Modelo de detecção ainda não foi carregado!");
      return;
    }

    // Esconder imagem original e mostrar canvas para detecção
    imagePreview.style.display = "none";
    currentMode = 'detection';

    // Limpa resultados anteriores
    hideResult();

    // Ajusta o canvas para o tamanho da imagem
    canvas.width = currentImage.width;
    canvas.height = currentImage.height;
    canvas.style.display = "block";

    // Limpa o canvas antes de desenhar
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Desenha a imagem no canvas
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

    // Faz a detecção
    showLoading();
    try {
      const predictions = await detectionModel.detect(currentImage);
      hideLoading();

      let detectedObjects = [];

      predictions.forEach((pred) => {
        if (pred.class === "cat" || pred.class === "dog") {
          const [x, y, width, height] = pred.bbox;
          const confidence = (pred.score * 100).toFixed(1);

          // Caixa colorida
          ctx.strokeStyle = pred.class === "cat" ? "#00f2fe" : "#f5576c";
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);

          // Tradução + emoji
          const labelPt = pred.class === "cat" ? "Gato 🐱" : "Cachorro 🐶";

          // Texto sobre a caixa
          ctx.fillStyle = pred.class === "cat" ? "#00f2fe" : "#f5576c";
          ctx.font = "18px Arial";
          ctx.fillText(
            `${labelPt} (${confidence}%)`,
            x,
            y > 20 ? y - 5 : y + 20
          );

          // Guarda para exibir embaixo
          detectedObjects.push(`${labelPt} (${confidence}%)`);
        }
      });

      // Exibe os resultados
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

  // Alterna visualização entre modos
  function toggleView(mode) {
    if (mode === 'classification') {
      imagePreview.style.display = "block";
      canvas.style.display = "none";
    } else if (mode === 'detection') {
      imagePreview.style.display = "none";
      canvas.style.display = "block";
    }
  }

  // Modificado: usa toggleView
  function showResult(data) {
    toggleView('classification');
    resultContainer.className = "result-container";
    resultContainer.classList.add(data.class + "-result");

    resultIcon.textContent = data.class === "cat" ? "🐱" : "🐶";
    resultText.textContent = data.class_name;
    confidenceText.textContent = `Confiança: ${data.confidence}%`;

    resultContainer.style.display = "block";
  }

  // Modificado: usa toggleView
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

  // Adicione esta função antes do final do script
  async function predictImage(imageElement) {
    if (!model) {
      showError('Modelo ainda não foi carregado. Tente novamente em alguns instantes.');
      return;
    }

    showLoading();

    try {
      // Pré-processa a imagem
      const tensor = tf.tidy(() => {
        let img = tf.browser.fromPixels(imageElement)
          .resizeNearestNeighbor([metadata.imageSize || 224, metadata.imageSize || 224])
          .toFloat()
          .div(255.0)
          .expandDims(0);
        return img;
      });

      // Faz a predição
      const predictions = await model.predict(tensor).data();
      tensor.dispose();

      // Encontra o índice com maior probabilidade
      const maxIndex = predictions.indexOf(Math.max(...predictions));
      const label = (metadata.labels && metadata.labels[maxIndex]) || labels[maxIndex] || `Classe ${maxIndex}`;
      const confidence = (predictions[maxIndex] * 100).toFixed(2);

      // Mapeia para exibição em português
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