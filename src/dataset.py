import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class SoccerEventDataset(Dataset):
    def __init__(self, df, seq_len=20, is_train=True, scaler=None):
        # [수정] scaler를 외부에서 받을 수 있게 변경
        self.df = df.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.is_train = is_train
        
        self.cont_cols = ['start_x', 'start_y', 'time_diff', 'velocity', 'dist_to_goal', 'angle_to_goal']
        self.cat_cols = ['type_name', 'team_id'] 
        
        if self.is_train:
            self.df['end_x'] = pd.to_numeric(self.df['end_x'], errors='coerce').fillna(0.0)
            self.df['end_y'] = pd.to_numeric(self.df['end_y'], errors='coerce').fillna(0.0)

        # 연속형 변수 에러 방지
        for col in self.cont_cols:
            if col not in self.df.columns:
                print(f"Warning: {col} not found in dataframe. Filling with 0.")
                self.df[col] = 0.0
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0)

        # [핵심 수정] 외부에서 Scaler를 받으면 그것을 사용 (Test용), 없으면 새로 생성 (Train용)
        if scaler is not None:
            self.scaler = scaler
            self.df[self.cont_cols] = self.scaler.transform(self.df[self.cont_cols])
        else:
            self.scaler = StandardScaler()
            self.df[self.cont_cols] = self.scaler.fit_transform(self.df[self.cont_cols])
        
        # 시퀀스 인덱스 생성
        # 테스트 데이터의 경우 game_id가 끊기는 지점을 고려
        if 'game_id' in self.df.columns:
            valid_indices_mask = self.df['game_id'] == self.df['game_id'].shift(self.seq_len - 1)
            self.indices = self.df.index[valid_indices_mask].tolist()
        else:
            # game_id가 없으면 그냥 길이만큼 (위험하지만 fallback)
            self.indices = list(range(self.seq_len - 1, len(self.df)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end_idx = self.indices[idx]
        start_idx = end_idx - self.seq_len + 1
        
        sequence_data = self.df.iloc[start_idx : end_idx + 1]
        
        if len(sequence_data) != self.seq_len:
            # 혹시 모를 길이 불일치 예외처리
            diff = self.seq_len - len(sequence_data)
            if diff > 0: # 짧으면 앞을 복사해서 채움
                pad = sequence_data.iloc[[0] * diff]
                sequence_data = pd.concat([pad, sequence_data])
        
        x_cont = torch.FloatTensor(sequence_data[self.cont_cols].values.astype(float))
        x_cat = torch.LongTensor(sequence_data[self.cat_cols].values.astype(int))
        
        if self.is_train:
            target = self.df.iloc[end_idx][['end_x', 'end_y']].values.astype(float)
            return x_cont, x_cat, torch.FloatTensor(target)
        else:
            return x_cont, x_cat