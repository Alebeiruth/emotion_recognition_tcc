import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("Iniciando a reorganização do dataset ExpW...")

# --- CONFIGURAÇÃO ---
# Caminhos de ENTRADA (ajuste para os seus caminhos confirmados)
IMAGE_SRC_PATH = r'C:\Users\luhal\ExpCriativa\emotion_recognition_tcc\data\raw\ExpW\image\origin'
LABEL_SRC_PATH = r'C:\Users\luhal\ExpCriativa\emotion_recognition_tcc\data\raw\ExpW\label\label.lst'

# Caminho de SAÍDA para a nova estrutura de pastas
OUTPUT_PATH = r'C:\Users\luhal\ExpCriativa\emotion_recognition_tcc\data\processed\ExpW_organized'

# Mapeamento de emoções
emotion_mapping = {
    0: 'Raiva', 1: 'Nojo', 2: 'Medo', 3: 'Felicidade',
    4: 'Tristeza', 5: 'Surpresa', 6: 'Neutro'
}

# --- 1. Carregar os Rótulos ---
print("Carregando o ficheiro de rótulos...")
column_names = ['image_name', 'face_id', 'top', 'left', 'right', 'bottom', 'confidence', 'emotion_id']

# --- LINHA CORRIGIDA AQUI ---
df_expw = pd.read_csv(LABEL_SRC_PATH, sep='\s+', names=column_names)
# -----------------------------

df_expw['emotion_name'] = df_expw['emotion_id'].map(emotion_mapping)

# Remover linhas onde a emoção não foi mapeada (se houver)
df_expw.dropna(subset=['emotion_name'], inplace=True)
print(f"Total de {len(df_expw)} rótulos carregados.")

# --- 2. Criar a Divisão Treino/Teste ---
print("Criando a divisão de treino/teste (80/20)...")
# Usamos stratify para garantir que a proporção de cada emoção seja a mesma nos dois conjuntos
train_df, test_df = train_test_split(
    df_expw,
    test_size=0.2,          # 20% dos dados para teste
    random_state=42,        # Para garantir que a divisão seja sempre a mesma
    stratify=df_expw['emotion_id']
)
print(f"Conjunto de treino: {len(train_df)} imagens.")
print(f"Conjunto de teste: {len(test_df)} imagens.")


# --- 3. Função para Copiar os Ficheiros ---
def organize_files(df, split_name):
    """
    Copia os ficheiros de um dataframe para a nova estrutura de pastas.
    """
    print(f"\nOrganizando os ficheiros do conjunto de '{split_name}'...")
    # Usar tqdm para criar uma barra de progresso
    for index, row in tqdm(df.iterrows(), total=len(df)):
        emotion_folder = row['emotion_name']
        image_filename = row['image_name']
        
        # Criar a pasta de destino final (ex: .../train/Felicidade/)
        destination_folder = os.path.join(OUTPUT_PATH, split_name, emotion_folder)
        os.makedirs(destination_folder, exist_ok=True)
        
        # Definir o caminho de origem e destino para o ficheiro
        source_path = os.path.join(IMAGE_SRC_PATH, image_filename)
        destination_path = os.path.join(destination_folder, image_filename)
        
        # Copiar o ficheiro, apenas se ele existir na origem
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path) # copy2 preserva metadados
        else:
            print(f"AVISO: Imagem não encontrada na origem e foi pulada: {source_path}")

# --- 4. Executar a Organização ---
organize_files(train_df, 'train')
organize_files(test_df, 'test')

print("\nReorganização do dataset ExpW concluída com sucesso!")
print(f"Os dados organizados estão em: '{OUTPUT_PATH}'")