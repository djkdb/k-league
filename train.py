import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import SoccerEventDataset
from src.model import SpatialTemporalTransformer
from src.trainer import train_model
from src.feature_engineering import feature_engineering
from src.utils import seed_everything
import pandas as pd
from torch.utils.data import DataLoader, random_split

def main():
    # 0. 시드 고정
    torch.manual_seed(Config.SEED)
    print(f"Project Running on Device: {Config.DEVICE}")
    
    # 1. 데이터 로드
    try:
        print(f"데이터 로드 중: {Config.TRAIN_PATH}")
        df = pd.read_csv(Config.TRAIN_PATH)
    except FileNotFoundError:
        print(f"에러: 파일을 찾을 수 없습니다. {Config.TRAIN_PATH} 경로를 확인하세요.")
        return

    # 2. 피처 엔지니어링 수행 (데이터프레임 변환 + 인코더 획득)
    processed_df, encoders = feature_engineering(df)
    
    # 3. 데이터셋 생성
    full_dataset = SoccerEventDataset(processed_df, seq_len=Config.SEQ_LEN, is_train=True)
    
    # Train / Validation Split (8:2)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    print(f"데이터셋 준비 완료 - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoader 생성
    # drop_last=True: 배치 크기가 안 맞아 남는 데이터를 버림 (오류 방지)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # =========================================================================
    # [수정된 부분] 모델에 들어갈 입력 크기 자동 계산
    # =========================================================================
    
    # 1) 연속형 변수 개수 확인 (첫 번째 데이터 샘플을 꺼내서 확인)
    # dataset[0] returns -> (x_cont, x_cat, target)
    sample_x_cont, sample_x_cat, _ = full_dataset[0]
    num_cont_features = sample_x_cont.shape[1]  # 예: 6개 (속도, 거리, 각도 등)
    
    # 2) 범주형 변수 클래스 개수 확인 (encoders 딕셔너리 활용)
    # dataset.py의 self.cat_cols 순서와 일치해야 함 ['type_name', 'team_id']
    cat_dims = []
    # dataset.py에 정의된 순서대로 차원을 가져옵니다.
    target_cat_cols = ['type_name', 'team_id'] 
    
    for col in target_cat_cols:
        if col in encoders:
            # 해당 컬럼의 고유값 개수 (예: 팀이 25개면 25)
            cat_dims.append(len(encoders[col].classes_))
        else:
            # 만약 인코더가 없다면 기본값 100 등으로 설정 (에러 방지용)
            print(f"경고: {col}에 대한 인코더를 찾을 수 없습니다.")
            cat_dims.append(100)
            
    print(f"모델 입력 설정 - Cont Features: {num_cont_features}, Cat Dims: {cat_dims}")

    # [수정] Config 객체를 통째로 넘기는 대신, 필요한 값만 쏙쏙 뽑아서 넣어줍니다.
    model = SpatialTemporalTransformer(
        num_cont_features=num_cont_features,
        cat_dims=cat_dims,
        embed_dim=Config.EMBED_DIM,   # 128
        num_layers=Config.NUM_LAYERS, # 4
        seq_len=Config.SEQ_LEN,       # 20
        nhead=4  # Config에 없으므로 직접 지정 (128 / 4 = 32로 나누어 떨어짐)
                 # 또는 src/config.py에 N_HEAD = 4 를 추가하고 Config.N_HEAD로 써도 됩니다.
    ).to(Config.DEVICE)
    
    # 5. 학습 시작
    train_model(model, train_loader, val_loader, Config)

if __name__ == "__main__":
    main()