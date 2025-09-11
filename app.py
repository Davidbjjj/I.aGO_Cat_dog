import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import os

# Configurações
IMG_SIZE = (224, 224)
CLASS_NAMES = ['cat', 'dog']

def load_and_preprocess_image(image_path):
    """Carrega e pré-processa uma imagem para predição"""
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Converter BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img = cv2.resize(img, IMG_SIZE)
        
        # Melhorar contraste com CLAHE (igual foi feito no treinamento)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Normalizar
        img = img / 255.0
        
        # Adicionar dimensão do batch
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        return None

def predict_image(model, image_path, threshold=0.5):
    """Faz predição para uma única imagem"""
    # Pré-processar imagem
    processed_img = load_and_preprocess_image(image_path)
    if processed_img is None:
        return None, None
    
    # Fazer predição
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Determinar classe
    if prediction >= threshold:
        predicted_class = 'dog'
        confidence = prediction
    else:
        predicted_class = 'cat'
        confidence = 1 - prediction
    
    return predicted_class, confidence

def predict_multiple_images(model, image_folder):
    """Faz predições para todas as imagens em uma pasta"""
    results = []
    
    # Verificar se a pasta existe
    if not os.path.exists(image_folder):
        print(f"Pasta não encontrada: {image_folder}")
        return results
    
    # Listar arquivos de imagem
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(image_extensions)]
    
    print(f"Encontradas {len(image_files)} imagens para classificação...")
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        predicted_class, confidence = predict_image(model, image_path)
        
        if predicted_class is not None:
            results.append({
                'filename': image_file,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            print(f"{image_file}: {predicted_class} ({confidence:.2%})")
    
    return results

def show_prediction_results(model, image_path):
    """Mostra visualização da predição para uma imagem"""
    processed_img = load_and_preprocess_image(image_path)
    if processed_img is None:
        return
    
    # Fazer predição
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Determinar resultado
    if prediction >= 0.5:
        predicted_class = 'dog'
        confidence = prediction
        color = 'red'
    else:
        predicted_class = 'cat'
        confidence = 1 - prediction
        color = 'blue'
    
    # Mostrar imagem com resultado
    plt.figure(figsize=(8, 8))
    plt.imshow(processed_img[0])
    plt.title(f'Predição: {predicted_class}\nConfiança: {confidence:.2%}', 
              fontsize=16, color=color, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence

# Carregar os modelos treinados
print("Carregando modelos...")
try:
    # Tente carregar o modelo avançado primeiro
    model = keras.models.load_model('advanced_cat_dog_classifier.h5')
    print("Modelo avançado carregado com sucesso!")
except:
    try:
        # Se não encontrar, carrega o modelo básico
        model = keras.models.load_model('cat_dog_classifier.h5')
        print("Modelo básico carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        exit()

# Menu interativo
while True:
    print("\n" + "="*50)
    print("CLASSIFICADOR GATOS vs CACHORROS")
    print("="*50)
    print("1. Classificar uma imagem específica")
    print("2. Classificar todas as imagens de uma pasta")
    print("3. Testar com imagem da webcam")
    print("4. Sair")
    
    choice = input("\nEscolha uma opção (1-4): ").strip()
    
    if choice == '1':
        # Classificar uma imagem específica
        image_path = input("Digite o caminho completo da imagem: ").strip()
        if os.path.exists(image_path):
            predicted_class, confidence = show_prediction_results(model, image_path)
            if predicted_class:
                print(f"\nResultado: {predicted_class} (Confiança: {confidence:.2%})")
        else:
            print("Arquivo não encontrado!")
    
    elif choice == '2':
        # Classificar todas as imagens de uma pasta
        folder_path = input("Digite o caminho da pasta com imagens: ").strip()
        results = predict_multiple_images(model, folder_path)
        
        if results:
            # Estatísticas
            cats = sum(1 for r in results if r['predicted_class'] == 'cat')
            dogs = sum(1 for r in results if r['predicted_class'] == 'dog')
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"\nEstatísticas:")
            print(f"Gatos: {cats}")
            print(f"Cachorros: {dogs}")
            print(f"Confiança média: {avg_confidence:.2%}")
        else:
            print("Nenhuma imagem foi processada.")
    
    elif choice == '3':
        # Testar com webcam (opcional)
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Webcam não disponível!")
                continue
            
            print("Pressione 'q' para sair da webcam, 'c' para capturar e classificar")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mostrar frame
                cv2.imshow('Webcam - Pressione "c" para capturar', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Salvar frame temporariamente
                    temp_path = 'temp_capture.jpg'
                    cv2.imwrite(temp_path, frame)
                    
                    # Classificar
                    predicted_class, confidence = predict_image(model, temp_path)
                    if predicted_class:
                        print(f"Classificação: {predicted_class} (Confiança: {confidence:.2%})")
                    
                    # Limpar arquivo temporário
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Erro com webcam: {e}")
    
    elif choice == '4':
        print("Saindo...")
        break
    
    else:
        print("Opção inválida! Tente novamente.")