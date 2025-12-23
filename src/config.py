import torch

class Config:
    SEQ_LEN = 20
    EMBED_DIM = 128
    NUM_LAYERS = 4
    EPOCHS = 30
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    SEED = 42
    # 경로 설정
    TRAIN_PATH = "./data/raw/train.csv"
    # [수정 후] 파일 이름(.pth)까지 정확히 적어주세요
    MODEL_SAVE_PATH = "./saved_models/best_model.pth"
    
    # 장치 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')