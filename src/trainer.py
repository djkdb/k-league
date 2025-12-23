import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 복합 손실 함수
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # 1. Euclidean Distance (주 손실)
        euclidean = torch.sqrt(torch.sum((pred - target)**2, dim=1) + 1e-6).mean()
        
        # 2. MSE (좌표별 정확도)
        mse = self.mse(pred, target)
        
        # 3. 방향 손실 (각도 유사도)
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        direction_loss = 1 - (pred_norm * target_norm).sum(dim=1).mean()
        
        # 복합 손실
        total_loss = self.alpha * euclidean + (1 - self.alpha) * (mse + 0.1 * direction_loss)
        
        return total_loss, euclidean  # 유클리드는 로깅용

def train_model(model, train_loader, val_loader, config):
    print(f"학습 시작... Device: {config.DEVICE}")
    
    criterion = CombinedLoss(alpha=config.LOSS_ALPHA)
    
    # AdamW 옵티마이저 (Weight Decay 포함)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 코사인 어닐링 스케줄러 (Warm Restart)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 10 epoch마다 재시작
        T_mult=2,
        eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        # ========================
        # Training Loop
        # ========================
        model.train()
        train_loss_sum = 0
        train_euclidean_sum = 0
        
        for batch_idx, (x_cont, x_cat, y) in enumerate(train_loader):
            x_cont = x_cont.to(config.DEVICE)
            x_cat = x_cat.to(config.DEVICE)
            y = y.to(config.DEVICE)
            
            optimizer.zero_grad()
            output = model(x_cont, x_cat)
            
            loss, euclidean = criterion(output, y)
            loss.backward()
            
            # Gradient Clipping (폭발 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_euclidean_sum += euclidean.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_euclidean = train_euclidean_sum / len(train_loader)
        
        # ========================
        # Validation Loop
        # ========================
        model.eval()
        val_loss_sum = 0
        val_euclidean_sum = 0
        
        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont = x_cont.to(config.DEVICE)
                x_cat = x_cat.to(config.DEVICE)
                y = y.to(config.DEVICE)
                
                output = model(x_cont, x_cat)
                loss, euclidean = criterion(output, y)
                
                val_loss_sum += loss.item()
                val_euclidean_sum += euclidean.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_euclidean = val_euclidean_sum / len(val_loader)
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 로그 출력
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} (Euc: {avg_train_euclidean:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (Euc: {avg_val_euclidean:.4f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Best Model Save (Euclidean 기준)
        if avg_val_euclidean < best_val_loss:
            best_val_loss = avg_val_euclidean
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  --> ✅ Best Model Saved (Val Euclidean: {avg_val_euclidean:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= patience:
            print(f"Early Stopping at Epoch {epoch+1}")
            break
    
    print("=" * 60)
    print(f"학습 완료! Best Validation Euclidean: {best_val_loss:.4f}")
    print("=" * 60)