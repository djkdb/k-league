import torch.nn as nn
import torch

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, num_cont_features, cat_dims, embed_dim=128, nhead=8, num_layers=4, seq_len=20):
        super(SpatialTemporalTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 1. 임베딩 레이어
        # 연속형 변수 -> MLP -> Embedding Dimension
        self.cont_embedding = nn.Sequential(
            nn.Linear(num_cont_features, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 범주형 변수 -> Embedding Layer -> Sum -> Embedding Dimension
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in cat_dims
        ])
        
        # 2. 위치 인코딩 (Positional Encoding) - 학습 가능한 파라미터로 설정
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=512, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Global Pooling & Regression Head
        self.fc_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2) # X, Y 좌표 예측
        )
        
    def forward(self, x_cont, x_cat):
        # x_cont: (Batch, Seq, Num_Cont)
        # x_cat: (Batch, Seq, Num_Cat)
        
        # 연속형 임베딩
        emb_cont = self.cont_embedding(x_cont)
        
        # 범주형 임베딩 결합 (여러 카테고리를 합침)
        emb_cat = torch.zeros_like(emb_cont)
        for i, emb_layer in enumerate(self.cat_embeddings):
            emb_cat += emb_layer(x_cat[:, :, i])
            
        # Feature Fusion (단순 합 또는 Concatenation 후 Projection)
        x = emb_cont + emb_cat
        
        # Positional Encoding 추가
        x = x + self.pos_embedding
        
        # Transformer Forward
        # Output shape: (Batch, Seq, Embed_Dim)
        x = self.transformer_encoder(x)
        
        # 마지막 시점(t)의 정보만 사용하여 예측 (또는 Mean Pooling 사용 가능)
        # 여기서는 "현재 상황"이 가장 중요하므로 마지막 토큰 사용
        last_token = x[:, -1, :]
        
        # 좌표 예측
        coords = self.fc_head(last_token)
        
        return coords