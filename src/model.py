import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedSpatialTemporalTransformer(nn.Module):
    def __init__(self, num_cont_features, cat_dims, embed_dim=256, nhead=8, num_layers=6, seq_len=30):
        super(ImprovedSpatialTemporalTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 1. 연속형 변수 임베딩 (강화)
        self.cont_embedding = nn.Sequential(
            nn.Linear(num_cont_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 2. 범주형 변수 임베딩
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in cat_dims
        ])
        
        # 3. 위치 인코딩 (학습 가능)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # 4. 시간 정보 명시적 인코딩
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 5. Transformer Encoder (더 깊고 강력하게)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,  # 1024
            batch_first=True, 
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 6. Attention Pooling (모든 시퀀스 활용)
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # 7. 좌표 예측 헤드 (더 깊게)
        self.fc_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # X, Y 좌표
        )
        
    def forward(self, x_cont, x_cat):
        # x_cont: (Batch, Seq, Num_Cont)
        # x_cat: (Batch, Seq, Num_Cat)
        
        batch_size, seq_len, _ = x_cont.shape
        
        # 1. 연속형 임베딩
        emb_cont = self.cont_embedding(x_cont)
        
        # 2. 범주형 임베딩 결합
        emb_cat = torch.zeros_like(emb_cont)
        for i, emb_layer in enumerate(self.cat_embeddings):
            emb_cat += emb_layer(x_cat[:, :, i])
        
        # 3. 시간 정보 인코딩 (time_seconds가 cont에 포함되어 있다고 가정)
        # 실제로는 별도로 추출하거나 index로 접근
        # 여기서는 간단히 시퀀스 인덱스를 시간으로 사용
        time_indices = torch.arange(seq_len, device=x_cont.device).float().unsqueeze(0).unsqueeze(-1)
        time_indices = time_indices.expand(batch_size, -1, -1) / seq_len  # 정규화
        time_emb = self.time_mlp(time_indices)
        
        # 4. Feature Fusion
        x = emb_cont + emb_cat + time_emb
        
        # 5. Positional Encoding 추가
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # 6. Transformer Forward
        x = self.transformer_encoder(x)
        
        # 7. Attention Pooling (가중 평균)
        attn_weights = self.attention_pool(x)  # (Batch, Seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        x_pooled = (x * attn_weights).sum(dim=1)  # (Batch, Embed_Dim)
        
        # 8. 좌표 예측
        coords = self.fc_head(x_pooled)
        
        # 9. 좌표 범위 제한 (Sigmoid + Scale)
        coords_x = torch.sigmoid(coords[:, 0:1]) * 105
        coords_y = torch.sigmoid(coords[:, 1:2]) * 68
        coords = torch.cat([coords_x, coords_y], dim=1)
        
        return coords