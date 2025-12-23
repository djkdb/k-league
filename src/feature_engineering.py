import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    """
    강화된 피처 엔지니어링 (NaN 안전 처리)
    """
    # 0. NaN 사전 처리
    df = df.fillna(0)
    
    # 1. 기본 시간 차이
    df['time_diff'] = df.groupby('game_id')['time_seconds'].diff().fillna(0)
    
    # 2. 패스 방향 및 길이
    if 'end_x' in df.columns and 'end_y' in df.columns:
        df['pass_direction_x'] = df['end_x'] - df['start_x']
        df['pass_direction_y'] = df['end_y'] - df['start_y']
        df['pass_length'] = np.sqrt(df['pass_direction_x']**2 + df['pass_direction_y']**2)
        df['pass_angle'] = np.arctan2(df['pass_direction_y'], df['pass_direction_x'])
    else:
        # Test 데이터의 경우
        df['pass_direction_x'] = 0
        df['pass_direction_y'] = 0
        df['pass_length'] = 0
        df['pass_angle'] = 0
    
    # 3. 이동 거리 및 속도
    df['prev_x'] = df.groupby('game_id')['start_x'].shift(1)
    df['prev_y'] = df.groupby('game_id')['start_y'].shift(1)
    df['prev_x'] = df['prev_x'].fillna(df['start_x'])
    df['prev_y'] = df['prev_y'].fillna(df['start_y'])
    
    df['dist_diff'] = np.sqrt((df['start_x'] - df['prev_x'])**2 + (df['start_y'] - df['prev_y'])**2)
    df['velocity'] = df['dist_diff'] / (df['time_diff'] + 1e-6)
    
    # 4. 골대 관련 피처
    goal_x, goal_y = 105, 34
    df['dist_to_goal'] = np.sqrt((goal_x - df['start_x'])**2 + (goal_y - df['start_y'])**2)
    df['angle_to_goal'] = np.arctan2(goal_y - df['start_y'], goal_x - df['start_x'])
    
    # 5. 경기 상황 컨텍스트
    df['minute'] = (df['time_seconds'] // 60).fillna(0).astype(int)
    df['is_final_third'] = (df['start_x'] > 70).astype(int)
    df['is_penalty_area'] = ((df['start_x'] > 88.5) & 
                              (df['start_y'] > 13.84) & 
                              (df['start_y'] < 54.16)).astype(int)
    
    # 6. 필드 위치 구역 (3x3 그리드) - 안전 처리
    df['zone_x'] = pd.cut(df['start_x'], bins=[-0.1, 35, 70, 105.1], labels=[0, 1, 2], include_lowest=True)
    df['zone_y'] = pd.cut(df['start_y'], bins=[-0.1, 22.67, 45.33, 68.1], labels=[0, 1, 2], include_lowest=True)
    
    # Categorical을 안전하게 숫자로 변환
    df['zone_x'] = pd.to_numeric(df['zone_x'], errors='coerce').fillna(1).astype(int)
    df['zone_y'] = pd.to_numeric(df['zone_y'], errors='coerce').fillna(1).astype(int)
    
    # 7. 시퀀스 통계 (Rolling Features) - 안전 처리
    for window in [3, 5, 10]:
        # Rolling 계산
        rolling_x = df.groupby('game_id')['start_x'].rolling(window, min_periods=1).mean()
        rolling_y = df.groupby('game_id')['start_y'].rolling(window, min_periods=1).mean()
        rolling_tempo = df.groupby('game_id')['time_diff'].rolling(window, min_periods=1).mean()
        
        # Index 정렬 후 할당
        df[f'avg_x_{window}'] = rolling_x.reset_index(level=0, drop=True).values
        df[f'avg_y_{window}'] = rolling_y.reset_index(level=0, drop=True).values
        df[f'pass_tempo_{window}'] = rolling_tempo.reset_index(level=0, drop=True).values
        
        # NaN 대체
        df[f'avg_x_{window}'] = df[f'avg_x_{window}'].fillna(df['start_x'])
        df[f'avg_y_{window}'] = df[f'avg_y_{window}'].fillna(df['start_y'])
        df[f'pass_tempo_{window}'] = df[f'pass_tempo_{window}'].fillna(0)
    
    # 8. 팀 소유권 변화
    df['team_change'] = (df.groupby('game_id')['team_id'].shift(1) != df['team_id']).astype(int)
    df['team_change'] = df['team_change'].fillna(0)
    
    # 9. 가속도
    df['prev_velocity'] = df.groupby('game_id')['velocity'].shift(1).fillna(0)
    df['acceleration'] = df['velocity'] - df['prev_velocity']
    
    # 10. 범주형 변수 인코딩
    encoders = {}
    cat_features = ['type_name', 'team_id']
    
    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # 최종 NaN 처리
    df = df.fillna(0)
    
    # 무한대 값 처리
    df = df.replace([np.inf, -np.inf], 0)
    
    return df, encoders