from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import io
from tensorflow import keras
from PIL import Image

app = Flask(__name__)

# Carregar modelo
try:
    model = keras.models.load_model('cat_dog_classifier.h5')
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    model = None

IMG_SIZE = (224, 224)
CLASS_NAMES = ['🐱 Gato', '🐶 Cachorro']

def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Modelo não carregado"})

    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        processed = preprocess_image(image)
        prediction = model.predict(processed, verbose=0)[0][0]

        if prediction >= 0.5:
            predicted_class, confidence, class_name = "dog", float(prediction), CLASS_NAMES[1]
        else:
            predicted_class, confidence, class_name = "cat", float(1 - prediction), CLASS_NAMES[0]

        return jsonify({
            "class": predicted_class,
            "class_name": class_name,
            "confidence": round(confidence * 100, 2),
            "success": True
        })
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({"error": "Erro ao processar a imagem"})

# ⚠️ Não colocar app.run(), a Vercel gerencia isso
