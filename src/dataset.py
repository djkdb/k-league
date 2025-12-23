import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class SoccerEventDataset(Dataset):
    def __init__(self, df, seq_len=30, is_train=True, scaler=None):
        self.df = df.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.is_train = is_train
        
        # 연속형 피처 (확장)
        self.cont_cols = [
            'start_x', 'start_y', 'time_diff', 'velocity', 
            'dist_to_goal', 'angle_to_goal',
            'pass_direction_x', 'pass_direction_y', 'pass_length', 'pass_angle',
            'dist_diff', 'minute', 'is_final_third', 'is_penalty_area',
            'zone_x', 'zone_y', 'team_change', 'acceleration',
            'avg_x_3', 'avg_y_3', 'pass_tempo_3',
            'avg_x_5', 'avg_y_5', 'pass_tempo_5',
            'avg_x_10', 'avg_y_10', 'pass_tempo_10'
        ]
        
        # 범주형 피처
        self.cat_cols = ['type_name', 'team_id']
        
        # 타겟 변수 처리 (Train만)
        if self.is_train:
            self.df['end_x'] = pd.to_numeric(self.df['end_x'], errors='coerce').fillna(0.0)
            self.df['end_y'] = pd.to_numeric(self.df['end_y'], errors='coerce').fillna(0.0)
        
        # 연속형 변수 에러 방지
        for col in self.cont_cols:
            if col not in self.df.columns:
                print(f"Warning: {col} not found. Filling with 0.")
                self.df[col] = 0.0
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0)
        
        # 범주형 변수 타입 변환
        for col in self.cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)
        
        # 스케일링
        if scaler is not None:
            self.scaler = scaler
            self.df[self.cont_cols] = self.scaler.transform(self.df[self.cont_cols])
        else:
            self.scaler = StandardScaler()
            self.df[self.cont_cols] = self.scaler.fit_transform(self.df[self.cont_cols])
        
        # 시퀀스 인덱스 생성
        if 'game_id' in self.df.columns:
            # game_id가 연속된 구간만 선택
            valid_indices = []
            for end_idx in range(self.seq_len - 1, len(self.df)):
                start_idx = end_idx - self.seq_len + 1
                game_ids = self.df.iloc[start_idx:end_idx + 1]['game_id'].unique()
                if len(game_ids) == 1:  # 하나의 game_id만 포함
                    valid_indices.append(end_idx)
            self.indices = valid_indices
        else:
            self.indices = list(range(self.seq_len - 1, len(self.df)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        end_idx = self.indices[idx]
        start_idx = end_idx - self.seq_len + 1
        
        sequence_data = self.df.iloc[start_idx:end_idx + 1]
        
        # 길이 검증
        if len(sequence_data) != self.seq_len:
            diff = self.seq_len - len(sequence_data)
            if diff > 0:
                pad = pd.concat([sequence_data.iloc[[0]]] * diff)
                sequence_data = pd.concat([pad, sequence_data], ignore_index=True)
        
        # 텐서 변환
        x_cont = torch.FloatTensor(sequence_data[self.cont_cols].values.astype(float))
        x_cat = torch.LongTensor(sequence_data[self.cat_cols].values.astype(int))
        
        if self.is_train:
            target = self.df.iloc[end_idx][['end_x', 'end_y']].values.astype(float)
            return x_cont, x_cat, torch.FloatTensor(target)
        else:
            return x_cont, x_cat