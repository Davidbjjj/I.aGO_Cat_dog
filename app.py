from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
from PIL import Image
import io
import base64
import sys
import traceback

app = Flask(__name__)

# Carregar modelo
try:
    model = keras.models.load_model('cat_dog_classifier.h5')
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}", file=sys.stderr)
    model = None

IMG_SIZE = (224, 224)
CLASS_NAMES = ['🐱 Gato', '🐶 Cachorro']

def preprocess_image(image):
    try:
        # Garantir RGB e redimensionar
        image = image.convert("RGB").resize(IMG_SIZE)
        # Converter para numpy e normalizar
        image = np.array(image).astype("float32") / 255.0
        # Adicionar dimensão do batch
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Erro no pré-processamento: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

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
        if processed is None:
            return jsonify({"error": "Erro no pré-processamento da imagem"})

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
        print(f"Erro na predição: {e}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": "Erro interno ao processar a imagem"})

# ⚠️ Não colocar app.run(), o Vercel gerencia isso
