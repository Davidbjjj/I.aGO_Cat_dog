import express from "express";
import bodyParser from "body-parser";
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";

// Caminhos dos arquivos do modelo
const MODEL_DIR = path.resolve("./model");
const MODEL_PATH = `file://${MODEL_DIR}/model.json`;
const METADATA_PATH = path.join(MODEL_DIR, "metadata.json");

let model;
let inputShape;
let classNames = ["🐱 Gato", "🐶 Cachorro"];

// Carregar modelo e metadados
async function loadModel() {
  try {
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log("✅ Modelo carregado com sucesso!");
    inputShape = model.inputs[0].shape; // ex: [null, 224, 224, 3]
    console.log("Modelo espera entrada:", inputShape);

    if (fs.existsSync(METADATA_PATH)) {
      const metadata = JSON.parse(fs.readFileSync(METADATA_PATH, "utf-8"));
      if (metadata.labels) classNames = metadata.labels;
      console.log("Classes carregadas do metadata.json:", classNames);
    }
  } catch (err) {
    console.error("Erro ao carregar modelo:", err);
  }
}

// Pré-processamento da imagem
function preprocessImage(imageBuffer) {
  try {
    let tensor = tf.node.decodeImage(imageBuffer, 3); // RGB
    const [, H, W, C] = inputShape;

    if (H && W) {
      tensor = tf.image.resizeBilinear(tensor, [H, W]);
    }
    tensor = tensor.expandDims(0).div(255.0); // Normalização
    return tensor;
  } catch (err) {
    console.error("Erro no pré-processamento:", err);
    return null;
  }
}

const app = express();
app.use(bodyParser.json({ limit: "10mb" }));

// Rota principal
app.get("/", (req, res) => {
  res.send("Servidor rodando! Use POST /predict com uma imagem base64.");
});

// Rota de predição
app.post("/predict", async (req, res) => {
  if (!model) return res.json({ error: "Modelo não carregado" });

  try {
    const imageData = req.body.image.split(",")[1]; // tira o prefixo base64
    const imageBuffer = Buffer.from(imageData, "base64");

    const tensor = preprocessImage(imageBuffer);
    if (!tensor) return res.json({ error: "Erro no pré-processamento" });

    const prediction = model.predict(tensor);
    const prob = (await prediction.data())[0];

    let predictedClass, confidence, className;

    if (prob >= 0.5) {
      predictedClass = "dog";
      confidence = prob;
      className = classNames[1];
    } else {
      predictedClass = "cat";
      confidence = 1 - prob;
      className = classNames[0];
    }

    res.json({
      class: predictedClass,
      class_name: className,
      confidence: +(confidence * 100).toFixed(2),
      success: true,
    });
  } catch (err) {
    console.error("Erro na predição:", err);
    res.json({ error: "Erro ao processar a imagem" });
  }
});

// Iniciar servidor
const PORT = 8000;
app.listen(PORT, async () => {
  await loadModel();
  console.log(`🚀 Servidor rodando em http://127.0.0.1:${PORT}`);
});
