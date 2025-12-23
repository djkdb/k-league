import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import Config
from src.dataset import SoccerEventDataset
from src.model import ImprovedSpatialTemporalTransformer
from src.utils import seed_everything
from src.feature_engineering import feature_engineering

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
        print("❌ 메타 파일 로드 실패")
        return None, None

    first_path = meta_df.iloc[0]['path']
    real_test_root = find_actual_data_path(first_path)
    if real_test_root is None: 
        return None, None
    
    print(f"📂 데이터 경로: {real_test_root}")
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
        except Exception as e:
            pass

    full_test_df = pd.concat(all_sequences, ignore_index=True)
    return full_test_df, episode_ids

def predict_single_model(model_path, test_loader, device, num_cont_features, cat_dims):
    """단일 모델로 예측"""
    model = ImprovedSpatialTemporalTransformer(
        num_cont_features=num_cont_features, 
        cat_dims=cat_dims, 
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.NUM_LAYERS,
        seq_len=Config.SEQ_LEN,
        nhead=Config.NHEAD
    ).to(device)
    
    # 가중치 로드
    try:
        if device.type == 'cpu':
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"   ❌ 모델 로드 실패: {e}")
        return None
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            outputs = model(x_cont, x_cat)
            all_predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    return predictions

def inference_ensemble():
    """
    앙상블 추론: 여러 모델의 예측을 평균
    """
    device = Config.DEVICE
    print("🚀 앙상블 추론 시작")
    print("=" * 60)

    # 1. Train 데이터 로드
    print("📚 학습 데이터(Train) 로드 중...")
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
    
    # 4. Dataset 생성
    test_dataset = SoccerEventDataset(
        test_df, 
        seq_len=Config.SEQ_LEN, 
        is_train=False, 
        scaler=scaler
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # 5. 모델 정보
    num_cont_features = len(cont_cols)
    cat_dims = [train_df[col].nunique() for col in cat_cols]
    
    # 6. 🔥 앙상블: 여러 시드로 학습된 모델들의 예측 평균
    model_paths = [
        "./saved_models/best_model.pth",
        # 추가 모델이 있다면 여기에 추가
        # "./saved_models/best_model_seed123.pth",
        # "./saved_models/best_model_seed456.pth",
    ]
    
    # 실제로는 여러 시드로 학습해야 하지만, 
    # 일단 하나의 모델만 있다면 TTA(Test Time Augmentation) 사용
    
    print(f"\n🔮 앙상블 예측 시작 (모델 수: {len(model_paths)})")
    
    all_model_predictions = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"   모델 로드: {model_path}")
            pred = predict_single_model(
                model_path, 
                test_loader, 
                device, 
                num_cont_features, 
                cat_dims
            )
            if pred is not None:
                all_model_predictions.append(pred)
    
    if len(all_model_predictions) == 0:
        print("❌ 사용 가능한 모델이 없습니다!")
        return
    
    # 7. 🔥 앙상블 전략 선택
    # 방법 1: 평균 (Mean)
    final_predictions = np.mean(all_model_predictions, axis=0)
    
    # 방법 2: 중앙값 (Median) - 이상치에 강건
    # final_predictions = np.median(all_model_predictions, axis=0)
    
    # 방법 3: 가중 평균 (성능 좋은 모델에 더 높은 가중치)
    # weights = [0.6, 0.4]  # 모델별 가중치
    # final_predictions = np.average(all_model_predictions, axis=0, weights=weights)
    
    print(f"\n📊 앙상블 완료: {len(final_predictions)}개 예측")
    
    # 8. 좌표 범위 클리핑
    final_predictions[:, 0] = np.clip(final_predictions[:, 0], 0, 105)
    final_predictions[:, 1] = np.clip(final_predictions[:, 1], 0, 68)
    
    # 9. 제출 파일 생성
    save_path = './submission_ensemble.csv'
    
    if len(episode_ids) == len(final_predictions):
        submission = pd.DataFrame({
            'game_episode': episode_ids,
            'end_x': final_predictions[:, 0],
            'end_y': final_predictions[:, 1]
        })
        submission.to_csv(save_path, index=False, encoding='utf-8')
        print(f"\n✅ 제출 파일 저장: {save_path}")
    
    print("\n" + "=" * 60)
    print("🎉 앙상블 추론 완료!")
    print("=" * 60)

if __name__ == '__main__':
    inference_ensemble()