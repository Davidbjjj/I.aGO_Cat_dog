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

  let model;
  let labels = ["Gato", "Cachorro"];
  let metadata = {};

  // Inicializa o modelo
  initModel();

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

    if (!file.type.match("image.*")) {
      showError("Por favor, selecione uma imagem (JPEG, PNG, etc.)");
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.onload = () => {
        previewContainer.style.display = "block";
        predictImage(imagePreview); // só chama quando a imagem carregou
      };
      imagePreview.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function predictImage(image) {
    if (!model) {
      showError("Modelo ainda não foi carregado.");
      return;
    }

    showLoading();

    try {
      const tensor = tf.tidy(() => {
        let tensor = tf.browser
          .fromPixels(image)
          .resizeNearestNeighbor([
            metadata.imageSize || 224,
            metadata.imageSize || 224,
          ])
          .toFloat()
          .div(127.5)
          .sub(1) // normalização [-1, 1]
          .expandDims(0);
        return tensor;
      });

      const predictions = await model.predict(tensor).data();
      tensor.dispose();

      console.log("Predictions:", predictions);

      const maxIndex = predictions.indexOf(Math.max(...predictions));
      const className = labels[maxIndex] || `Classe ${maxIndex}`;
      const confidence = (predictions[maxIndex] * 100).toFixed(2);

      hideLoading();
      showResult({
        class: className.toLowerCase(),
        class_name: `É um ${className}!`,
        confidence: confidence,
      });
    } catch (error) {
      hideLoading();
      console.error("Erro na predição:", error);
      showError("Erro ao processar a imagem. Tente novamente.");
    }
  }

  function showResult(data) {
    resultContainer.className = "result-container";
    resultContainer.classList.add(data.class + "-result");

    resultIcon.textContent = data.class.includes("gato") ? "🐱" : "🐶";
    resultText.textContent = data.class_name;
    confidenceText.textContent = `Confiança: ${data.confidence}%`;

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
});
