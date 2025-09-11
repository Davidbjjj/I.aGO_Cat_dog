import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2
from tqdm import tqdm
import seaborn as sns

# TensorFlow e Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0

# Configurações
IMG_SIZE = (224, 224)  # Aumentei o tamanho para melhor capturar características
BATCH_SIZE = 32
EPOCHS = 50
DATA_PATH = r"C:\Users\d.a.vieira.da.silva\Desktop\dataset\imagens"

def load_data(data_path):
    """Carrega as imagens e labels do dataset"""
    classes = ['cat', 'dog']
    images = []
    labels = []
    
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        class_idx = classes.index(class_name)
        
        if not os.path.exists(class_path):
            print(f"Diretório não encontrado: {class_path}")
            continue
            
        for img_name in tqdm(os.listdir(class_path), desc=f"Carregando {class_name}"):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE)
                    
                    # Melhorar contraste com CLAHE
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    l = clahe.apply(l)
                    lab = cv2.merge((l, a, b))
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    continue
    
    return np.array(images), np.array(labels)

# Carregar dados
print("Carregando imagens...")
X, y = load_data(DATA_PATH)
print(f"Dataset carregado: {X.shape[0]} imagens, tamanho {X.shape[1:]}")

if len(X) == 0:
    print("Nenhuma imagem foi carregada. Verifique o caminho do dataset.")
    exit()

# Balanceamento de classes
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights))
print(f"Pesos das classes: {class_weights}")

# Dividir em treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

print(f"Treino: {X_train.shape[0]}, Validação: {X_val.shape[0]}, Teste: {X_test.shape[0]}")

# Data Augmentation avançado
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    channel_shift_range=20.0
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Criar modelo mais avançado com Transfer Learning
def create_advanced_model():
    # Usar EfficientNetB0 pré-treinado
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Congelar camadas base inicialmente
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model, base_model

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=8,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Criar e compilar modelo
print("Criando modelo avançado...")
model, base_model = create_advanced_model()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

model.summary()

# Primeira fase de treinamento (apenas camadas densas)
print("Primeira fase de treinamento...")
history1 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=20,
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Segunda fase: Fine-tuning (descongelar algumas camadas)
print("Segunda fase: Fine-tuning...")
base_model.trainable = True

# Congelar as primeiras 100 camadas
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

history2 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=30,
    validation_data=val_datagen.flow(X_val, y_val),
    validation_steps=len(X_val) // BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Avaliar o modelo
print("Avaliando modelo...")
X_test_processed = X_test / 255.0
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    test_datagen.flow(X_test, y_test, shuffle=False),
    verbose=0
)

print(f"\nResultados no conjunto de teste:")
print(f"Acurácia: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# Fazer previsões detalhadas
y_pred_proba = model.predict(X_test_processed)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.show()

# Gráfico de aprendizado
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='Treino')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'] + history2.history['loss'], label='Treino')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='Validação')
plt.title('Loss durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Visualizar previsões
class_names = ['cat', 'dog']
plt.figure(figsize=(15, 10))
for i in range(min(12, len(X_test))):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[i] / 255.0)
    confidence = max(y_pred_proba[i][0], 1 - y_pred_proba[i][0])
    plt.title(f"Real: {class_names[y_test[i]]}\nPred: {class_names[y_pred[i][0]]}\nConf: {confidence:.2f}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Salvar o modelo
model.save('advanced_cat_dog_classifier.h5')
print("Modelo avançado salvo como 'advanced_cat_dog_classifier.h5'")