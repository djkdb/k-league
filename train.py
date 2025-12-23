import torch
from torch.utils.data import DataLoader, random_split
from src.config import Config
from src.dataset import SoccerEventDataset
from src.model import ImprovedSpatialTemporalTransformer
from src.trainer import train_model
from src.feature_engineering import feature_engineering
from src.utils import seed_everything
import pandas as pd
import os

def main():
    # 0. 시드 고정
    seed_everything(Config.SEED)
    print(f"🚀 프로젝트 시작 | Device: {Config.DEVICE}")
    
    # 모델 저장 디렉토리 생성
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    
    # 1. 데이터 로드
    try:
        print(f"📂 데이터 로드 중: {Config.TRAIN_PATH}")
        df = pd.read_csv(Config.TRAIN_PATH)
        print(f"   원본 데이터 크기: {df.shape}")
    except FileNotFoundError:
        print(f"❌ 에러: 파일을 찾을 수 없습니다. {Config.TRAIN_PATH}")
        return
    
    # 2. 피처 엔지니어링 수행
    print("🔧 피처 엔지니어링 수행 중...")
    processed_df, encoders = feature_engineering(df)
    print(f"   처리 후 데이터 크기: {processed_df.shape}")
    
    # 3. 데이터셋 생성
    print(f"📊 데이터셋 생성 중 (SEQ_LEN={Config.SEQ_LEN})...")
    full_dataset = SoccerEventDataset(
        processed_df, 
        seq_len=Config.SEQ_LEN, 
        is_train=True
    )
    
    # Train / Validation Split (8:2)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    print(f"   ✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # 4. DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=0  # Windows 호환성
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # 5. 모델 입력 크기 계산
    print("🔍 모델 입력 크기 계산 중...")
    sample_x_cont, sample_x_cat, _ = full_dataset[0]
    num_cont_features = sample_x_cont.shape[1]
    
    # 범주형 변수 차원
    cat_dims = []
    target_cat_cols = ['type_name', 'team_id']
    
    for col in target_cat_cols:
        if col in encoders:
            cat_dims.append(len(encoders[col].classes_))
        else:
            print(f"⚠️ 경고: {col}에 대한 인코더 없음. 기본값 100 사용")
            cat_dims.append(100)
    
    print(f"   연속형 피처: {num_cont_features}개")
    print(f"   범주형 차원: {cat_dims}")
    
    # 6. 모델 생성
    print("🏗️ 모델 생성 중...")
    model = ImprovedSpatialTemporalTransformer(
        num_cont_features=num_cont_features,
        cat_dims=cat_dims,
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.NUM_LAYERS,
        seq_len=Config.SEQ_LEN,
        nhead=Config.NHEAD
    ).to(Config.DEVICE)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   총 파라미터: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")
    
    # 7. 학습 시작
    print("\n" + "=" * 60)
    print("🎓 모델 학습 시작")
    print("=" * 60)
    
    train_model(model, train_loader, val_loader, Config)
    
    print("\n✅ 모든 과정이 완료되었습니다!")

if __name__ == "__main__":
    main()