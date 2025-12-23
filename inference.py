# inference.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.config import Config
from src.feature_engineering import feature_engineering
from src.dataset import SoccerEventDataset
from src.model import SpatialTemporalTransformer
from src.utils import seed_everything

def inference():
    # 1. 설정 및 시드 고정
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Inference Device: {device}")

    # 2. 데이터 로드 (Train은 인코더/스케일러 기준 잡기용, Test는 예측용)
    print("데이터 로딩 중...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df = pd.read_csv('./data/test.csv') # 경로 확인 필요
    
    # 3. 전처리 (Feature Engineering)
    # 주의: Train과 Test를 합쳐서 인코딩하거나, Train의 인코더를 저장했다가 써야 함.
    # 여기서는 간편하게 두 데이터를 합쳐서 기준을 잡고 다시 나눕니다.
    train_len = len(train_df)
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # 통합 데이터로 피처 엔지니어링 (인코딩 일관성 유지)
    processed_combined, encoders = feature_engineering(combined_df)
    
    # 다시 분리
    processed_test = processed_combined.iloc[train_len:].reset_index(drop=True)
    
    # 4. Test Dataset & DataLoader 생성
    test_dataset = SoccerEventDataset(processed_test, seq_len=Config.SEQ_LEN, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 5. 모델 구조 초기화 (학습 때와 똑같은 설정이어야 함)
    # 데이터셋에서 feature 개수 가져오기
    sample_x_cont, sample_x_cat = test_dataset[0]
    num_cont_features = sample_x_cont.shape[1]
    cat_dims = [len(enc.classes_) for enc in encoders.values()]
    
    model = SpatialTemporalTransformer(
        num_cont_features=num_cont_features, 
        cat_dims=cat_dims, 
        config=Config
    ).to(device)
    
    # 6. 학습된 가중치 로드 (.pth 파일)
    model_path = Config.MODEL_SAVE_PATH  # "./best_model.pth"
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
        
    print(f"모델 로드 완료: {model_path}")
    
    # 7. 추론 수행
    model.eval()
    all_predictions = []
    
    print("추론 시작...")
    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            
            # 예측 (x, y)
            outputs = model(x_cont, x_cat)
            all_predictions.append(outputs.cpu().numpy())
            
    # 결과를 하나의 배열로 합치기
    predictions = np.concatenate(all_predictions, axis=0)
    
    # 8. 후처리 (좌표값 클리핑: 경기장을 벗어나지 않게)
    # K리그 경기장 크기: 0~105 (x), 0~68 (y)
    predictions[:, 0] = np.clip(predictions[:, 0], 0, 105)
    predictions[:, 1] = np.clip(predictions[:, 1], 0, 68)
    
    # 9. 제출 파일 생성
    # submission 형식을 확인해야 합니다. 보통 game_id 등의 식별자가 필요할 수 있습니다.
    # 예시: sample_submission.csv가 있다면 그걸 불러와서 채워넣는 방식 권장
    
    try:
        submission = pd.read_csv('./data/sample_submission.csv')
        # 모델의 출력 순서와 submission의 순서가 일치한다고 가정 (Dataset 순서 유지됨)
        # 하지만 Sequence Data 특성상 Test 데이터의 행 개수와 예측 개수가 맞는지 확인 필수!
        
        # 참고: SoccerEventDataset은 시퀀스 단위로 데이터를 만듭니다.
        # Test 데이터의 예측 대상 이벤트 수와 predictions의 길이가 같은지 확인
        print(f"Submission 행 수: {len(submission)}, 예측 결과 수: {len(predictions)}")
        
        if len(submission) == len(predictions):
            submission['x'] = predictions[:, 0]
            submission['y'] = predictions[:, 1]
            submission.to_csv('./submission.csv', index=False)
            print("제출 파일 저장 완료: ./submission.csv")
        else:
            print("경고: 예측 개수와 제출 파일 행 수가 다릅니다. 데이터셋 생성 로직을 확인하세요.")
            # Dataset 생성 시 seq_len 때문에 앞부분 데이터가 잘렸을 수 있습니다.
            # Test 시에는 padding을 하거나, 앞부분을 복제해서 길이를 맞춰주는 테크닉이 필요합니다.
            
    except FileNotFoundError:
        print("sample_submission.csv를 찾을 수 없어, 예측 결과만 csv로 저장합니다.")
        df_result = pd.DataFrame(predictions, columns=['x', 'y'])
        df_result.to_csv('./submission_raw.csv', index=False)

if __name__ == '__main__':
    inference()