import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    # 1. 시간 차이 (Delta Time)
    # game_id 별로 그룹화하여 이전 이벤트와의 시간 차 계산
    df['time_diff'] = df.groupby('game_id')['time_seconds'].diff().fillna(0)
    
    # 2. 이동 거리 및 속도 (Velocity)
    # 이전 이벤트의 end_x, end_y가 현재 이벤트의 start_x, start_y라고 가정 (연속성)
    # 실제로는 데이터에 따라 shift가 필요할 수 있습니다.
    df['prev_x'] = df.groupby('game_id')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_y'] = df.groupby('game_id')['start_y'].shift(1).fillna(df['start_y'])
    
    df['dist_diff'] = np.sqrt((df['start_x'] - df['prev_x'])**2 + (df['start_y'] - df['prev_y'])**2)
    
    # 0으로 나누기 방지
    df['velocity'] = df['dist_diff'] / (df['time_diff'] + 1e-6)
    
    # 3. 골대와의 거리 및 각도 (Polar Coordinates)
    # 공격 방향이 왼쪽->오른쪽(105, 34)이라고 가정 (Spatial Alignment 전처리 되었다고 가정)
    goal_x, goal_y = 105, 34
    
    df['dist_to_goal'] = np.sqrt((goal_x - df['start_x'])**2 + (goal_y - df['start_y'])**2)
    df['angle_to_goal'] = np.arctan2(goal_y - df['start_y'], goal_x - df['start_x'])
    
    # 4. 범주형 변수 인코딩 (Label Encoding)
    cat_features = ['type_name', 'player_id', 'team_id'] # 예시 컬럼
    encoders = {}
    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    return df, encoders