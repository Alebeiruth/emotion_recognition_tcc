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
BASE_PATH = r'C:\Users\luhal\TCC\data\raw\RAF-DB\DATASET'
# O novo output é um dataset limpo, mas ainda desbalanceado
OUTPUT_PATH = r'C:\Users\luhal\TCC\data\processed\raf_db_aligned_gray' 
TARGET_SIZE = (224, 224)

# Mapeamento de emoções (como no notebook EDA)
emotion_mapping = {
    1: 'Surpresa', 2: 'Medo', 3: 'Nojo', 4: 'Felicidade',
    5: 'Tristeza', 6: 'Raiva', 7: 'Neutro'
}

# Inicializar o detetor de rosto MTCNN
detector = MTCNN()

# --- FUNÇÃO AUXILIAR DE ALINHAMENTO (mantida) ---
def align_face(image, left_eye, right_eye):
    """
    Gira a imagem para alinhar os olhos na horizontal.
    """
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(eye_dy, eye_dx))
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return aligned_image

# --- SCRIPT PRINCIPAL ---

print(f"Iniciando pré-processamento da RAF-DB...")
print(f"Os dados processados serão salvos em: '{OUTPUT_PATH}'")

# Processar e salvar todas as imagens (treino e teste)
for split in ['train', 'test']:
    input_split_path = os.path.join(BASE_PATH, split)
    output_split_path = os.path.join(OUTPUT_PATH, split)

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
                
                if cropped_face.size == 0:
                    continue

                # Passo 3: Redimensionamento
                resized_face = cv2.resize(cropped_face, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                # Passo 4: Escala de Cinza
                gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(output_image_path, gray_face)

print("\nPré-processamento (limpeza e padronização) concluído com sucesso!")