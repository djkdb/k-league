import torch
import torch.nn as nn
import torch.optim as optim
import math

# 유클리드 거리 손실 함수
class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
        
    def forward(self, pred, target):
        # pred: (Batch, 2), target: (Batch, 2)
        # sqrt((x1-x2)^2 + (y1-y2)^2)
        return torch.sqrt(torch.sum((pred - target)**2, dim=1)).mean()

def train_model(model, train_loader, val_loader, config):
    print(f"학습 시작... Device: {config.DEVICE}")
    
    criterion = EuclideanLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=config.EPOCHS
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss_sum = 0
        
        # Training Loop
        for batch_idx, (x_cont, x_cat, y) in enumerate(train_loader):
            x_cont = x_cont.to(config.DEVICE)
            x_cat = x_cat.to(config.DEVICE)
            y = y.to(config.DEVICE)
            
            optimizer.zero_grad()
            output = model(x_cont, x_cat)
            
            loss = criterion(output, y)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss_sum = 0
        
        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont = x_cont.to(config.DEVICE)
                x_cat = x_cat.to(config.DEVICE)
                y = y.to(config.DEVICE)
                
                output = model(x_cont, x_cat)
                loss = criterion(output, y)
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Best Model Save
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"--> Best Model Saved (Val Loss: {avg_val_loss:.4f})")
            
    print("모든 학습이 완료되었습니다.")