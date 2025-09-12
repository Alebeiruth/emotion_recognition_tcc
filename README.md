# Emotion Recognition TCC - Deep Learning Project

Um projeto abrangente de TCC focado no reconhecimento de emoções faciais usando técnicas modernas de Deep Learning. Este projeto implementa e compara diferentes arquiteturas de redes neurais para classificação de emoções, incluindo análises de erro detalhadas, estudos de ablação e estratégias avançadas de data augmentation.

## 📋 Índice

- [Características](#características)
- [Arquiteturas Implementadas](#arquiteturas-implementadas)
- [Datasets](#datasets)
- [Requisitos do Sistema](#requisitos-do-sistema)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuração](#configuração)
- [Experimentos](#experimentos)
- [Docker](#docker)
- [Resultados](#resultados)
- [API de Inferência](#api-de-inferência)
- [Contribuição](#contribuição)

## ✨ Características

### 🧠 Modelos de Deep Learning
- **EfficientNet-B0** com fine-tuning em duas fases
- **ResNet-50** com transfer learning
- **EfficientViT** para eficiência computacional
- Fusão de classificadores para melhor performance

### 📊 Análises Avançadas
- Análise de erro detalhada com identificação de vieses
- Estudos de ablação para componentes dos modelos
- Análise de drift entre datasets
- Métricas de fairness e equidade
- Visualização de espaços de features

### 🔧 Ferramentas e Utilitários
- Monitoramento completo de recursos (CPU, GPU, memória)
- Tracking de experimentos com Weights & Biases e MLflow
- Data augmentation robusta e adaptativa
- Containerização com Docker
- Notebooks Jupyter organizados para cada experimento

### 📈 Métricas e Visualizações
- Relatórios detalhados de performance
- Matrizes de confusão interativas
- Análise de complexidade computacional
- Exportação automática de resultados em CSV

## 🏗️ Arquiteturas Implementadas

| Modelo | Parâmetros | Complexidade | Características |
|--------|------------|--------------|----------------|
| **EfficientNet-B0** | ~5.3M | Baixa-Média | Compound scaling, Mobile-friendly |
| **ResNet-50** | ~25.6M | Média | Skip connections, Proven architecture |
| **EfficientViT** | ~3.2M | Baixa | Vision Transformer, Efficient attention |

## 📂 Datasets

### Principais Datasets
1. **FER2013** - 35,887 imagens em escala de cinza (48x48)
2. **RAF-DB** - 29,672 imagens de alta qualidade com anotações
3. **DFEW** - Vídeos dinâmicos para análise temporal

### Emoções Reconhecidas
- 😠 Raiva (Anger)
- 🤢 Desgosto (Disgust) 
- 😨 Medo (Fear)
- 😊 Felicidade (Happy)
- 😐 Neutro (Neutral)
- 😢 Tristeza (Sadness)
- 😲 Surpresa (Surprise)

## 💻 Requisitos do Sistema

### Hardware Recomendado
- **GPU**: NVIDIA GPU com CUDA 12.1+ (RTX 3060 ou superior)
- **RAM**: 16GB mínimo, 32GB recomendado
- **Storage**: 50GB de espaço livre (datasets + modelos)
- **CPU**: Intel Core i7 ou AMD Ryzen 7

### Software
- **OS**: Ubuntu 24.04 LTS (recomendado)
- **Python**: 3.8 ou superior
- **Docker**: 20.10+
- **NVIDIA Drivers**: 525.60+ para suporte CUDA

## 🚀 Instalação

### Opção 1: Instalação Automatizada (Recomendada)

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/emotion-recognition-tcc.git
cd emotion-recognition-tcc

# Execute o script de setup automatizado
python scripts/setup_environment.py

# Ative o ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### Opção 2: Instalação Manual

```bash
# 1. Criar ambiente virtual
python3.12 -m venv venv
source venv/bin/activate

# 2. Atualizar pip
pip install --upgrade pip

# 3. Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar TensorFlow com GPU
pip install tensorflow[and-cuda]

# 5. Instalar outras dependências
pip install -r requirements.txt

# 6. Instalar projeto em modo desenvolvimento
pip install -e .
```

### Opção 3: Docker (Mais Simples)

```bash
# Build e execute o container
docker-compose up --build

# Acesse Jupyter Lab em: http://localhost:8888
```

## ⚡ Uso Rápido

### 1. Preparar Dados
```bash
# Baixe os datasets e organize na estrutura:
data/raw/FER2013/
data/raw/RAF-DB/DATASET/
data/raw/DFEW/

# Execute o script de download (se disponível)
python scripts/download_datasets.py
```

### 2. Executar Experimentos

```bash
# Todos os modelos
python scripts/run_experiments.py

# Modelo específico
python scripts/run_experiments.py --models efficientnet

# Com configuração customizada
python scripts/run_experiments.py --config custom_config.yaml
```

### 3. Usar Notebooks

```bash
# Iniciar Jupyter Lab
jupyter lab

# Notebooks disponíveis:
# - 01_exploratory_data_analysis.ipynb
# - 05_model_training_efficientnet.ipynb
# - 08_ablation_studies.ipynb
```

## 📁 Estrutura do Projeto

```
emotion_recognition_tcc/
├── 📋 README.md
├── 🐳 Dockerfile & docker-compose.yml
├── 📦 requirements.txt & setup.py
│
├── 🔧 config/
│   ├── config.yaml              # Configuração principal
│   └── model_config.py          # Configurações de modelo
│
├── 📊 data/
│   ├── raw/                     # Datasets originais
│   ├── processed/               # Dados processados
│   └── splits/                  # Divisões train/val/test
│
├── 📓 notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 05_model_training_efficientnet.ipynb
│   ├── 08_ablation_studies.ipynb
│   └── 11_real_time_inference.ipynb
│
├── 🧠 src/
│   ├── models/                  # Implementações dos modelos
│   ├── data/                    # Carregamento e preprocessing
│   ├── training/                # Pipeline de treinamento
│   ├── evaluation/              # Avaliação e métricas
│   ├── inference/               # Inferência em tempo real
│   └── utils/                   # Utilitários
│
├── 🧪 tests/                    # Testes automatizados
│
├── 📜 scripts/
│   ├── setup_environment.py    # Setup automatizado
│   ├── run_experiments.py      # Executor de experimentos
│   └── generate_reports.py     # Geração de relatórios
│
└── 📈 results/
    ├── models/                  # Modelos treinados
    ├── logs/                    # Logs de treinamento
    ├── plots/                   # Visualizações
    └── csv_outputs/             # Resultados em CSV
```

## ⚙️ Configuração

O arquivo `config/config.yaml` centraliza todas as configurações:

```yaml
# Exemplo de configurações principais
data:
  datasets:
    fer2013:
      path: "data/raw/FER2013"
    raf_db:
      path: "data/raw/RAF-DB/DATASET"
  
  preprocessing:
    image_size: 224
    normalize: true

training:
  batch_size: 32
  epochs: 0
  learning_rate: 0.001

models:
  efficientnet:
    pretrained: true
    dropout_rate: 0.5
```

### Configurações Importantes

| Parâmetro | Descrição | Valor Padrão |
|-----------|-----------|--------------|
| `image_size` | Tamanho das imagens | 224 |
| `batch_size` | Tamanho do batch | 32 |
| `epochs` | Épocas de treinamento | -*- |
| `validation_split` | Divisão para validação | 0.3 |
| `mixed_precision` | Precisão mista (GPU) | true |

## 🧪 Experimentos

### Execução Completa
```bash
# Todos os experimentos com monitoramento
python scripts/run_experiments.py \
    --config config/config.yaml \
    --output-dir results/experiment_$(date +%Y%m%d)
```

### Experimentos Específicos
```bash
# Apenas EfficientNet
python scripts/run_experiments.py --models efficientnet

# Comparação de arquiteturas
python scripts/run_experiments.py --models efficientnet resnet50

# Estudos de ablação
python scripts/run_experiments.py --config config/ablation_config.yaml
```

### Notebooks Interativos

1. **EDA** - `01_exploratory_data_analysis.ipynb`
   - Análise exploratória dos dados
   - Distribuição de classes
   - Qualidade das imagens

2. **Treinamento** - `05_model_training_efficientnet.ipynb`
   - Treinamento step-by-step
   - Visualização em tempo real
   - Fine-tuning detalhado

3. **Avaliação** - `09_bias_analysis.ipynb`
   - Análise de vieses
   - Métricas de fairness
   - Casos de falha

## 🐳 Docker

### Desenvolvimento
```bash
# Ambiente completo com Jupyter
docker-compose up

# Acesso:
# - Jupyter Lab: http://localhost:8888
# - MLflow: http://localhost:5000
```

### Produção
```bash
# Apenas API de inferência
docker-compose --profile production up

# Acesso: http://localhost:8000
```

### Treinamento
```bash
# Container dedicado para treinamento
docker-compose --profile training up
```

### Arquivos Gerados

Cada experimento gera automaticamente:

- `model_comparison_TIMESTAMP.csv` - Comparação entre modelos
- `efficientnet_overall_TIMESTAMP.csv` - Resultados gerais
- `efficientnet_per_class_TIMESTAMP.csv` - Métricas por classe
- `efficientnet_training_history_TIMESTAMP.csv` - Histórico de treinamento
- `experiment_summary_TIMESTAMP.txt` - Relatório completo

## 🔌 API de Inferência

### Iniciar API
```bash
# Modo desenvolvimento
python -m src.inference.model_serving

# Com Docker
docker-compose --profile production up
```

### Contribuições
Contribuições são bem-vindas! Veja [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

## 📚 Referências Acadêmicas

1. **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
2. **ResNet**: He, K., et al. (2016). Deep Residual Learning for Image Recognition.
3. **FER2013**: Goodfellow, I., et al. (2013). Challenges in Representation Learning.
4. **RAF-DB**: Li, S., et al. (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning.


## 🙏 Agradecimentos

- Orientador: Prof. Rayson Laroca
- Pontifícia Universidade Católica do Paraná
- Comunidade open-source do TensorFlow e PyTorch
- Datasets disponibilizados publicamente

---

**📧 Contato**: alexandre.beiruth@pucpr.edu.br 
**🔗 LinkedIn**: https://www.linkedin.com/in/alexandre-beiruth-bcc/

> Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) em Ciência da Computação, focando na aplicação prática de técnicas modernas de Deep Learning para reconhecimento de emoções faciais.