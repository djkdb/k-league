"""
Test Time Augmentation (TTA)
추론 시 입력에 약간의 노이즈를 추가하여 여러 번 예측 후 평균
"""
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import Config
from src.model import ImprovedSpatialTemporalTransformer
from src.utils import seed_everything
from src.feature_engineering import feature_engineering

class TTADataset(Dataset):
    """TTA를 위한 데이터셋 (노이즈 추가 가능)"""
    def __init__(self, base_dataset, noise_std=0.3):
        self.base_dataset = base_dataset
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x_cont, x_cat = self.base_dataset[idx]
        
        # 연속형 변수에 작은 노이즈 추가
        if self.noise_std > 0:
            noise = torch.randn_like(x_cont) * self.noise_std
            x_cont = x_cont + noise
        
        return x_cont, x_cat

def apply_train_encoding(train_df, test_df, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_values = train_df[col].astype(str).unique()
        le.fit(train_values)
        encoders[col] = le
        
        test_values = test_df[col].astype(str).values
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        test_df[col] = [mapping.get(val, 0) for val in test_values]
        
    return test_df, encoders

def find_actual_data_path(meta_df_path_sample, start_dir='.'):
    target_filename = os.path.basename(meta_df_path_sample)
    for root, dirs, files in os.walk(start_dir):
        if target_filename in files:
            full_path = os.path.join(root, target_filename)
            dir_containing_file = os.path.dirname(full_path) 
            test_root = os.path.dirname(dir_containing_file)
            return test_root
    return None

def load_test_data(meta_path, seq_len):
    try:
        meta_df = pd.read_csv(meta_path)
    except:
        return None, None

    first_path = meta_df.iloc[0]['path']
    real_test_root = find_actual_data_path(first_path)
    if real_test_root is None: 
        return None, None
    
    all_sequences = []
    episode_ids = [] 
    
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="테스트 데이터 로드"):
        parts = row['path'].replace('\\', '/').split('/')
        relative_part = os.path.join(parts[-2], parts[-1])
        file_path = os.path.join(real_test_root, relative_part)
        
        try:
            df = pd.read_csv(file_path)
            if len(df) < seq_len:
                pad_len = seq_len - len(df)
                pad = pd.DataFrame([df.iloc[0]] * pad_len, columns=df.columns)
                df = pd.concat([pad, df], ignore_index=True)
            else:
                df = df.iloc[-seq_len:]
            
            df['game_id'] = row['game_episode']
            all_sequences.append(df)
            episode_ids.append(row['game_episode'])
        except:
            pass

    full_test_df = pd.concat(all_sequences, ignore_index=True)
    return full_test_df, episode_ids

def inference_tta(n_tta=5, noise_std=0.3):
    """
    TTA 추론
    
    Args:
        n_tta: TTA 반복 횟수 (5~10 추천)
        noise_std: 노이즈 표준편차 (0.2~0.5 추천)
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    
    print(f"🚀 TTA 추론 시작 (반복: {n_tta}회, 노이즈: {noise_std})")
    print("=" * 60)

    # 1. Train 데이터 로드
    print("📚 학습 데이터 로드 중...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    train_df, _ = feature_engineering(train_df) 
    train_df = train_df.fillna(0)

    # 2. Test 데이터 로드
    print("\n📂 테스트 데이터 로드 중...")
    test_df, episode_ids = load_test_data("./data/raw/test.csv", Config.SEQ_LEN)
    if test_df is None: 
        return
    
    test_df, _ = feature_engineering(test_df)
    test_df = test_df.fillna(0)

    # 3. 전처리
    print("\n⚖️ 전처리 적용 중...")
    cat_cols = ['type_name', 'team_id']
    test_df, _ = apply_train_encoding(train_df, test_df, cat_cols)
    
    cont_cols = [
        'start_x', 'start_y', 'time_diff', 'velocity', 
        'dist_to_goal', 'angle_to_goal',
        'pass_direction_x', 'pass_direction_y', 'pass_length', 'pass_angle',
        'dist_diff', 'minute', 'is_final_third', 'is_penalty_area',
        'zone_x', 'zone_y', 'team_change', 'acceleration',
        'avg_x_3', 'avg_y_3', 'pass_tempo_3',
        'avg_x_5', 'avg_y_5', 'pass_tempo_5',
        'avg_x_10', 'avg_y_10', 'pass_tempo_10'
    ]
    
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols].values)
    
    # 4. 기본 Dataset 생성
    from src.dataset import SoccerEventDataset
    base_dataset = SoccerEventDataset(
        test_df, 
        seq_len=Config.SEQ_LEN, 
        is_train=False, 
        scaler=scaler
    )
    
    # 5. 모델 로드
    print("\n🏗️ 모델 로드 중...")
    num_cont_features = len(cont_cols)
    cat_dims = [train_df[col].nunique() for col in cat_cols]
    
    model = ImprovedSpatialTemporalTransformer(
        num_cont_features=num_cont_features, 
        cat_dims=cat_dims, 
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.NUM_LAYERS,
        seq_len=Config.SEQ_LEN,
        nhead=Config.NHEAD
    ).to(device)
    
    model_path = Config.MODEL_SAVE_PATH
    try:
        if device.type == 'cpu':
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
        print(f"   ✅ 모델 로드 성공")
    except Exception as e:
        print(f"   ❌ 모델 로드 실패: {e}")
        return

    model.eval()
    
    # 6. 🔥 TTA 추론
    print(f"\n🔮 TTA 추론 시작 ({n_tta}회 반복)...")
    all_tta_predictions = []
    
    for tta_idx in range(n_tta):
        print(f"   TTA {tta_idx+1}/{n_tta}...")
        
        # TTA Dataset 생성 (첫 번째는 원본, 나머지는 노이즈 추가)
        if tta_idx == 0:
            tta_dataset = TTADataset(base_dataset, noise_std=0)  # 원본
        else:
            tta_dataset = TTADataset(base_dataset, noise_std=noise_std)
        
        tta_loader = DataLoader(
            tta_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        # 추론
        predictions = []
        with torch.no_grad():
            for x_cont, x_cat in tta_loader:
                x_cont = x_cont.to(device)
                x_cat = x_cat.to(device)
                outputs = model(x_cont, x_cat)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        all_tta_predictions.append(predictions)
    
    # 7. TTA 결과 평균
    final_predictions = np.mean(all_tta_predictions, axis=0)
    
    print(f"\n📊 TTA 완료")
    print(f"   예측 개수: {len(final_predictions)}")
    
    # 8. 좌표 범위 클리핑
    final_predictions[:, 0] = np.clip(final_predictions[:, 0], 0, 105)
    final_predictions[:, 1] = np.clip(final_predictions[:, 1], 0, 68)
    
    # 9. 제출 파일 생성
    save_path = './submission_tta.csv'
    
    if len(episode_ids) == len(final_predictions):
        submission = pd.DataFrame({
            'game_episode': episode_ids,
            'end_x': final_predictions[:, 0],
            'end_y': final_predictions[:, 1]
        })
        submission.to_csv(save_path, index=False, encoding='utf-8')
        print(f"\n✅ 제출 파일 저장: {save_path}")
    
    print("\n" + "=" * 60)
    print("🎉 TTA 추론 완료!")
    print("=" * 60)

if __name__ == '__main__':
    # TTA 파라미터 조정 가능
    inference_tta(n_tta=5, noise_std=0.3)