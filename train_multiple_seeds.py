"""
여러 시드로 모델을 학습하여 앙상블용 모델 생성
"""
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

def train_with_seed(seed, model_save_path):
    """특정 시드로 모델 학습"""
    print(f"\n{'='*60}")
    print(f"🌱 SEED {seed}로 학습 시작")
    print(f"{'='*60}\n")
    
    # 시드 설정
    seed_everything(seed)
    
    # 데이터 로드
    df = pd.read_csv(Config.TRAIN_PATH)
    processed_df, encoders = feature_engineering(df)
    
    # 데이터셋 생성
    full_dataset = SoccerEventDataset(
        processed_df, 
        seq_len=Config.SEQ_LEN, 
        is_train=True
    )
    
    # Train/Val Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # 모델 생성
    sample_x_cont, sample_x_cat, _ = full_dataset[0]
    num_cont_features = sample_x_cont.shape[1]
    
    cat_dims = []
    target_cat_cols = ['type_name', 'team_id']
    for col in target_cat_cols:
        if col in encoders:
            cat_dims.append(len(encoders[col].classes_))
        else:
            cat_dims.append(100)
    
    model = ImprovedSpatialTemporalTransformer(
        num_cont_features=num_cont_features,
        cat_dims=cat_dims,
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.NUM_LAYERS,
        seq_len=Config.SEQ_LEN,
        nhead=Config.NHEAD
    ).to(Config.DEVICE)
    
    # 학습 (모델 저장 경로 임시 변경)
    original_path = Config.MODEL_SAVE_PATH
    Config.MODEL_SAVE_PATH = model_save_path
    
    train_model(model, train_loader, val_loader, Config)
    
    Config.MODEL_SAVE_PATH = original_path
    
    print(f"\n✅ SEED {seed} 모델 저장 완료: {model_save_path}\n")

def main():
    """여러 시드로 모델 학습"""
    
    # 앙상블할 시드 목록 (5개 정도 추천)
    seeds = [42, 123, 456, 789, 2024]
    
    print("🎯 다중 시드 학습 시작")
    print(f"총 {len(seeds)}개 모델 학습 예정")
    print(f"시드 목록: {seeds}")
    
    # 저장 디렉토리 생성
    os.makedirs("./saved_models", exist_ok=True)
    
    # 각 시드로 학습
    for i, seed in enumerate(seeds):
        model_path = f"./saved_models/best_model_seed{seed}.pth"
        
        try:
            train_with_seed(seed, model_path)
        except Exception as e:
            print(f"❌ SEED {seed} 학습 실패: {e}")
            continue
        
        print(f"\n진행률: {i+1}/{len(seeds)} 완료\n")
    
    print("\n" + "="*60)
    print("🎉 모든 시드 학습 완료!")
    print("="*60)
    print("\n앙상블 추론을 위해 inference_ensemble.py를 실행하세요:")
    print("python inference_ensemble.py")

if __name__ == "__main__":
    main()