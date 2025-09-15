import os
import cv2
import numpy as np
import random
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import warnings

# Ignorar avisos que não são críticos
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO ---
# Caminhos de entrada e saída (versão com caminhos relativos)
# --- VERSÃO CORRIGIDA ---
BASE_PATH = './data/raw/RAF-DB/DATASET'
OUTPUT_PATH = './data/processed/raf_db_balanced'
temp_output_path = './data/processed/raf_db_temp_gray_aligned'
TARGET_SIZE = (224, 224)
TARGET_SAMPLES_PER_CLASS = 1000

# Mapeamento de emoções (como no notebook EDA)
emotion_mapping = {
    1: 'Surpresa', 2: 'Medo', 3: 'Nojo', 4: 'Felicidade',
    5: 'Tristeza', 6: 'Raiva', 7: 'Neutro'
}

# Inicializar o detetor de rosto MTCNN
detector = MTCNN()

# --- FUNÇÕES AUXILIARES ---

def align_face(image, left_eye, right_eye):
    """
    Gira a imagem para alinhar os olhos na horizontal.
    """
    # Calcular o ângulo entre os olhos
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(eye_dy, eye_dx))

    # Obter o centro da imagem e a matriz de rotação
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Aplicar a rotação
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return aligned_image

def augment_image(image):
    """
    Aplica transformações de data augmentation leves.
    """
    # Rotação aleatória
    angle = random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    # Espelhamento horizontal aleatório
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    return image

# --- SCRIPT PRINCIPAL ---

print("Iniciando pré-processamento avançado da RAF-DB...")

# 1. Processar e salvar todas as imagens (treino e teste) em um diretório temporário
print(f"Passo 1/3: Alinhando, recortando e salvando imagens em escala de cinza em '{temp_output_path}'...")

for split in ['train', 'test']:
    input_split_path = os.path.join(BASE_PATH, split)
    output_split_path = os.path.join(temp_output_path, split)

    for emotion_id_str in tqdm(os.listdir(input_split_path), desc=f"Processando {split}"):
        emotion_name = emotion_mapping.get(int(emotion_id_str))
        input_emotion_path = os.path.join(input_split_path, emotion_id_str)
        output_emotion_path = os.path.join(output_split_path, emotion_name)
        os.makedirs(output_emotion_path, exist_ok=True)

        for image_name in os.listdir(input_emotion_path):
            input_image_path = os.path.join(input_emotion_path, image_name)
            output_image_path = os.path.join(output_emotion_path, image_name)

            img = cv2.imread(input_image_path)
            if img is None:
                continue

            results = detector.detect_faces(img)
            if results:
                keypoints = results[0]['keypoints']
                left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']

                # Passo 1: Alinhamento
                aligned_img = align_face(img, left_eye, right_eye)
                
                # Re-detetar para obter a bounding box na imagem alinhada
                results_aligned = detector.detect_faces(aligned_img)
                if not results_aligned:
                    continue
                
                # Passo 2: Recorte
                x, y, w, h = results_aligned[0]['box']
                cropped_face = aligned_img[y:y+h, x:x+w]
                
                # Passo 3: Redimensionamento
                resized_face = cv2.resize(cropped_face, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                # Passo 4: Escala de Cinza
                gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(output_image_path, gray_face)

# 2. Balancear o conjunto de treino
train_temp_path = os.path.join(temp_output_path, 'train')
train_balanced_path = os.path.join(OUTPUT_PATH, 'train')
print("\nPasso 2/3: Balanceando o conjunto de treino para 1000 amostras por classe...")

for emotion_name in tqdm(os.listdir(train_temp_path), desc="Balanceando classes"):
    input_class_path = os.path.join(train_temp_path, emotion_name)
    output_class_path = os.path.join(train_balanced_path, emotion_name)
    os.makedirs(output_class_path, exist_ok=True)

    images = [os.path.join(input_class_path, f) for f in os.listdir(input_class_path)]
    current_samples = len(images)

    if current_samples > TARGET_SAMPLES_PER_CLASS:
        # Undersampling
        selected_images = random.sample(images, TARGET_SAMPLES_PER_CLASS)
        for img_path in selected_images:
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(output_class_path, os.path.basename(img_path)), img)
    else:
        # Oversampling com Data Augmentation
        # Primeiro, copiar todas as imagens originais
        for img_path in images:
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(output_class_path, os.path.basename(img_path)), img)
        
        # Gerar as imagens novas necessárias
        samples_to_generate = TARGET_SAMPLES_PER_CLASS - current_samples
        for i in range(samples_to_generate):
            random_img_path = random.choice(images)
            img_to_augment = cv2.imread(random_img_path)
            augmented_img = augment_image(img_to_augment)
            
            # Salvar com um nome único
            base_name = os.path.splitext(os.path.basename(random_img_path))[0]
            new_name = f"{base_name}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(output_class_path, new_name), augmented_img)

# 3. Copiar o conjunto de teste (ele não deve ser balanceado)
test_temp_path = os.path.join(temp_output_path, 'test')
test_final_path = os.path.join(OUTPUT_PATH, 'test')
print("\nPasso 3/3: Copiando conjunto de teste processado...")
import shutil
if os.path.exists(test_final_path):
    shutil.rmtree(test_final_path)
shutil.copytree(test_temp_path, test_final_path)


print("\nPré-processamento e balanceamento concluídos com sucesso!")
print(f"Os dados finais estão em: '{OUTPUT_PATH}'")