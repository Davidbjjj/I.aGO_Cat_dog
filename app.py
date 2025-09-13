from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import io
from tensorflow import keras
from PIL import Image
import sys
import traceback

app = Flask(__name__)

MODEL_PATH = 'dog_cat_model.h5'

# Carregar modelo
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso!")
    input_shape = model.input_shape  # ex: (None, None, None, 3) ou (None, 224, 224, 3)
    print("Modelo espera entrada:", input_shape)
except Exception as e:
    print(f"Erro ao carregar modelo: {e}", file=sys.stderr)
    model = None
    input_shape = (None, 224, 224, 3)

# Detectar dinamicamente altura, largura e canais
_, H, W, C = input_shape
if H is None or W is None:
    H, W = None, None

CLASS_NAMES = ['🐱 Gato', '🐶 Cachorro']

def preprocess_image(image):
    try:
        image = image.convert("RGB")
        # Redimensionar apenas se altura/largura do modelo forem fixas
        if H is not None and W is not None:
            image = image.resize((W, H))
        # Converter para numpy e normalizar
        img_array = np.array(image).astype("float32") / 255.0
        # Adicionar dimensão do batch
        return np.expand_dims(img_array, axis=0)
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
        return jsonify({"error": "Erro ao processar a imagem"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
