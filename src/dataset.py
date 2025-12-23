import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SoccerEventDataset(Dataset):
    def __init__(self, df, seq_len=20, is_train=True):
        # [수정 1] 인덱스가 꼬이지 않도록 초기화 (매우 중요)
        self.df = df.copy().reset_index(drop=True)
        self.seq_len = seq_len
        self.is_train = is_train
        
        # 사용할 피처 정의
        self.cont_cols = ['start_x', 'start_y', 'time_diff', 'velocity', 'dist_to_goal', 'angle_to_goal']
        self.cat_cols = ['type_name', 'team_id'] 
        
        # 타겟 변수 전처리 (에러 방지)
        if self.is_train:
            self.df['end_x'] = pd.to_numeric(self.df['end_x'], errors='coerce').fillna(0.0)
            self.df['end_y'] = pd.to_numeric(self.df['end_y'], errors='coerce').fillna(0.0)

        # 데이터 정규화
        self.scaler = StandardScaler()
        for col in self.cont_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0)
            
        self.df[self.cont_cols] = self.scaler.fit_transform(self.df[self.cont_cols])
        
        # [수정 2] 시퀀스 인덱스 로직 변경 (핵심 수정)
        # 이전 코드: shift(-seq_len) -> 미래를 봄
        # 수정 코드: shift(seq_len - 1) -> 과거를 봄 (우리는 과거 데이터를 학습하므로 이게 맞음)
        # 의미: "나(row)와 내 19개 전 조상의 game_id가 같다면, 나는 유효한 시퀀스의 끝이다."
        valid_indices_mask = self.df['game_id'] == self.df['game_id'].shift(self.seq_len - 1)
        self.indices = self.df.index[valid_indices_mask].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 현재 시점(t)을 기준으로 과거 seq_len개의 데이터를 가져옴
        end_idx = self.indices[idx]
        start_idx = end_idx - self.seq_len + 1
        
        # 입력 시퀀스 (History)
        # iloc은 끝 인덱스를 포함하지 않으므로 +1을 해줘야 end_idx까지 포함됨
        sequence_data = self.df.iloc[start_idx : end_idx + 1]
        
        # [디버깅용 안전장치] 만약 계산 착오로 길이가 안 맞으면 강제 조정
        if len(sequence_data) != self.seq_len:
            # 에러가 나면 그냥 0으로 채운 더미 데이터를 반환 (학습 중단 방지)
            print(f"Warning: Index {idx} length mismatch. Expected {self.seq_len}, got {len(sequence_data)}")
            # 임시 처리: 부족한 만큼 패딩하거나 해당 배치 스킵 필요 (여기서는 일단 에러 발생 유도하되 메시지 출력)
        
        # 연속형 변수
        x_cont = torch.FloatTensor(sequence_data[self.cont_cols].values.astype(float))
        
        # 범주형 변수
        x_cat = torch.LongTensor(sequence_data[self.cat_cols].values.astype(int))
        
        if self.is_train:
            # 타겟: 마지막 이벤트(t)의 결과
            target = self.df.iloc[end_idx][['end_x', 'end_y']].values.astype(float)
            return x_cont, x_cat, torch.FloatTensor(target)
        else:
            return x_cont, x_cat