import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("Iniciando pré-processamento do dataset de vídeo DFEW...")

# --- CONFIGURAÇÃO ---
# Caminhos de ENTRADA (ajuste os nomes das pastas/ficheiros se forem diferentes)
DFEW_BASE_PATH = './data/raw/DFEW/'
DFEW_VIDEOS_PATH = os.path.join(DFEW_BASE_PATH, 'Clip/clip_224x224')
DFEW_TRAIN_LABELS_PATH = os.path.join(DFEW_BASE_PATH, 'EmoLabel_DataSplit/train_single_label.csv')
DFEW_TEST_LABELS_PATH = os.path.join(DFEW_BASE_PATH, 'EmoLabel_DataSplit/test_single_label.csv')

# Caminho de SAÍDA para os frames processados
OUTPUT_PATH = './data/processed/dfew_aligned_gray_frames'
TARGET_SIZE = (224, 224)

# Inicializar o detetor de rosto MTCNN
detector = MTCNN()

# --- FUNÇÃO AUXILIAR DE ALINHAMENTO (A mesma dos outros scripts) ---
def align_face(image, left_eye, right_eye):
    """Gira a imagem para alinhar os olhos na horizontal."""
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(eye_dy, eye_dx))
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return aligned_image

# --- SCRIPT PRINCIPAL ---

# 1. Carregar e combinar os ficheiros de rótulos
print("Carregando ficheiros de rótulos...")
try:
    df_train = pd.read_csv(DFEW_TRAIN_LABELS_PATH)
    df_test = pd.read_csv(DFEW_TEST_LABELS_PATH)

    df_train['split'] = 'train'
    df_test['split'] = 'test'
    
    # Supondo que as colunas são 'Clip_Name' e 'Emo_Type' (ajuste se necessário)
    df_train.rename(columns={'Clip_Name': 'video_filename', 'Emo_Type': 'emotion_id'}, inplace=True)
    df_test.rename(columns={'Clip_Name': 'video_filename', 'Emo_Type': 'emotion_id'}, inplace=True)

    df_dfew = pd.concat([df_train, df_test], ignore_index=True)
    
    emotion_mapping = {
        1: 'Raiva', 2: 'Nojo', 3: 'Medo', 4: 'Felicidade',
        5: 'Tristeza', 6: 'Surpresa', 7: 'Neutro'
    }
    df_dfew['emotion_name'] = df_dfew['emotion_id'].map(emotion_mapping)
    print(f"Total de {len(df_dfew)} vídeos para processar.")

except FileNotFoundError:
    print("ERRO: Ficheiros de rótulos não encontrados. Verifique os caminhos e nomes dos ficheiros.")
    exit()

# 2. Loop principal para processar cada vídeo
print("\nIniciando extração e processamento de frames...")
for index, row in tqdm(df_dfew.iterrows(), total=len(df_dfew), desc="Processando Vídeos"):
    video_filename = f"{row['video_filename']}.mp4" # Adicionar extensão
    split = row['split']
    emotion_name = row['emotion_name']

    # Montar caminho de entrada do vídeo
    video_path = os.path.join(DFEW_VIDEOS_PATH, video_filename)

    if not os.path.exists(video_path):
        # print(f"AVISO: Vídeo não encontrado, pulando: {video_path}")
        continue
    
    # Montar caminho de saída para os frames deste vídeo
    output_video_folder = os.path.join(OUTPUT_PATH, split, emotion_name)
    os.makedirs(output_video_folder, exist_ok=True)

    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Fim do vídeo

        # Processar o frame (mesma lógica dos outros scripts)
        results = detector.detect_faces(frame)
        if results:
            keypoints = results[0]['keypoints']
            left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']
            
            aligned_img = align_face(frame, left_eye, right_eye)
            
            results_aligned = detector.detect_faces(aligned_img)
            if not results_aligned:
                continue
            
            x, y, w, h = results_aligned[0]['box']
            cropped_face = aligned_img[y:y+h, x:x+w]
            
            if cropped_face.size == 0:
                continue
            
            resized_face = cv2.resize(cropped_face, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            
            # Salvar o frame processado com um nome único
            frame_filename = f"{os.path.splitext(video_filename)[0]}_frame_{frame_count:04d}.jpg"
            output_frame_path = os.path.join(output_video_folder, frame_filename)
            cv2.imwrite(output_frame_path, gray_face)
            
            frame_count += 1
    
    cap.release()

print("\nPré-processamento do DFEW concluído com sucesso!")
print(f"Os frames processados estão em: '{OUTPUT_PATH}'")