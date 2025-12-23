import torch

class Config:
    # 모델 하이퍼파라미터 (개선)
    SEQ_LEN = 30  # 20 → 30 (더 긴 컨텍스트)
    EMBED_DIM = 256  # 128 → 256 (더 큰 표현력)
    NUM_LAYERS = 6  # 4 → 6 (더 깊은 네트워크)
    NHEAD = 8  # 명시적 추가
    
    # 학습 설정
    EPOCHS = 50  # 30 → 50
    BATCH_SIZE = 32  # 64 → 32 (큰 모델이므로)
    LEARNING_RATE = 5e-4  # 1e-3 → 5e-4 (더 안정적)
    WEIGHT_DECAY = 1e-4  # 정규화 추가
    SEED = 42
    
    # 경로 설정
    TRAIN_PATH = "./data/raw/train.csv"
    MODEL_SAVE_PATH = "./saved_models/best_model.pth"
    
    # 장치 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 손실 함수 가중치
    LOSS_ALPHA = 0.7  # Euclidean vs MSE 비율