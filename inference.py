import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import Config
from src.dataset import SoccerEventDataset
from src.model import SpatialTemporalTransformer
from src.utils import seed_everything
from src.feature_engineering import feature_engineering

# -----------------------------------------------------------
# [도우미 함수] Test 데이터에 Train의 기준(Encoder) 적용하기
# -----------------------------------------------------------
def apply_train_encoding(train_df, test_df, cat_cols):
    """
    Train 데이터로 LabelEncoder를 학습(fit)시키고,
    Test 데이터에 그 규칙을 적용(transform)합니다.
    새로운 카테고리(Unknown)는 -1 또는 0으로 처리합니다.
    """
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Train의 모든 값을 문자열로 변환하여 학습 (에러 방지)
        train_values = train_df[col].astype(str).unique()
        le.fit(train_values)
        encoders[col] = le
        
        # Test 변환 (Unknown 처리 포함)
        test_values = test_df[col].astype(str).values
        # le.transform은 모르는 값이 오면 에러가 나므로 map 방식을 사용
        # 딕셔너리로 변환표 생성 {Class: Index}
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        
        # 매핑 적용 (없으면 0번으로 대체 - 보통 0번이 가장 흔한 클래스거나 임의 지정)
        # 더 정교하게 하려면 'Unknown' 클래스를 추가해야 하지만 여기선 0으로 처리
        test_df[col] = [mapping.get(val, 0) for val in test_values]
        
    return test_df, encoders

# -----------------------------------------------------------
# Main Inference Logic
# -----------------------------------------------------------
def find_actual_data_path(meta_df_path_sample, start_dir='.'):
    target_filename = os.path.basename(meta_df_path_sample)
    print(f"🔍 데이터 위치 찾는 중... ({target_filename})")
    for root, dirs, files in os.walk(start_dir):
        if target_filename in files:
            full_path = os.path.join(root, target_filename)
            dir_containing_file = os.path.dirname(full_path) 
            test_root = os.path.dirname(dir_containing_file)
            return test_root
    return None

def load_test_data(meta_path, seq_len):
    # Test 데이터 로드 로직 (기존과 동일)
    try:
        meta_df = pd.read_csv(meta_path)
    except:
        print("메타 파일 로드 실패")
        return None, None

    first_path = meta_df.iloc[0]['path']
    real_test_root = find_actual_data_path(first_path)
    if real_test_root is None: return None, None
    
    print(f"📂 데이터 경로: {real_test_root}")
    all_sequences = []
    episode_ids = [] 
    
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
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
        except: pass

    full_test_df = pd.concat(all_sequences, ignore_index=True)
    return full_test_df, episode_ids

def inference():
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Inference Device: {device}")

    # 1. [중요] Train 데이터 로드 (기준 잡기용)
    print("🎓 학습 데이터(Train) 로드 중... (기준점 설정을 위해 필요)")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    
    # Train 피처 엔지니어링 (velocity 등 생성)
    # feature_engineering 함수가 (df, encoders)를 반환한다고 가정
    # 여기서 반환되는 encoders는 무시하고, 아래에서 안전하게 다시 만듭니다.
    train_df, _ = feature_engineering(train_df) 
    train_df = train_df.fillna(0)

    # 2. Test 데이터 로드
    test_df, episode_ids = load_test_data("./data/raw/test.csv", Config.SEQ_LEN)
    if test_df is None: return
    
    # Test 피처 엔지니어링
    test_df, _ = feature_engineering(test_df)
    test_df = test_df.fillna(0)

    print(f"데이터 준비 완료 - Train: {train_df.shape}, Test: {test_df.shape}")

    # 3. [핵심] Train 기준으로 인코딩 & 스케일링 적용
    print("⚖️ 학습 데이터 기준으로 스케일링 및 인코딩 적용 중...")
    
    # (1) 범주형 변수 (Label Encoding)
    cat_cols = ['type_name', 'team_id']
    test_df, _ = apply_train_encoding(train_df, test_df, cat_cols)
    
    # (2) 연속형 변수 (StandardScaler)
    # Train 데이터로 Scaler 학습
    cont_cols = ['start_x', 'start_y', 'time_diff', 'velocity', 'dist_to_goal', 'angle_to_goal']
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols].values) # Train으로 Fit!
    
    # 4. Dataset 생성 (만들어진 scaler 전달)
    test_dataset = SoccerEventDataset(test_df, seq_len=Config.SEQ_LEN, is_train=False, scaler=scaler)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 5. 모델 로드 및 추론
    num_cont_features = len(cont_cols)
    cat_dims = [26, 12] # 학습 때와 동일하게 고정
    
    model = SpatialTemporalTransformer(
        num_cont_features=num_cont_features, 
        cat_dims=cat_dims, 
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.NUM_LAYERS,
        seq_len=Config.SEQ_LEN,
        nhead=4
    ).to(device)
    
    model_path = Config.MODEL_SAVE_PATH
    try:
        if device.type == 'cpu':
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
        print("✅ 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    model.eval()
    all_predictions = []
    
    print("🚀 추론 시작...")
    with torch.no_grad():
        for x_cont, x_cat in tqdm(test_loader):
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            outputs = model(x_cont, x_cat)
            all_predictions.append(outputs.cpu().numpy())
            
    predictions = np.concatenate(all_predictions, axis=0)
    predictions[:, 0] = np.clip(predictions[:, 0], 0, 105)
    predictions[:, 1] = np.clip(predictions[:, 1], 0, 68)
    
    # 6. 제출 파일 생성 (ID + 예측값)
    save_path = './submission.csv'
    if len(episode_ids) == len(predictions):
        submission = pd.DataFrame({
            'game_episode': episode_ids,
            'end_x': predictions[:, 0],
            'end_y': predictions[:, 1]
        })
        submission.to_csv(save_path, index=False)
        print(f"✅ 제출 파일 저장 완료: {save_path}")
    else:
        print(f"⚠️ 개수 불일치 (ID: {len(episode_ids)} vs Pred: {len(predictions)})")
        df_result = pd.DataFrame(predictions, columns=['end_x', 'end_y'])
        df_result.to_csv(save_path, index=False)
        print("비상 저장 완료")

if __name__ == '__main__':
    inference()