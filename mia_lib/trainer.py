# generic training loops for target/shadow models

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from mia_lib.utils import EarlyStopPatience

def trainer(model, train_loader, val_loader, CFG, save_path):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.target_model.learning_rate,
        weight_decay=CFG.target_model.weight_decay
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    step_scheduler = StepLR(optimizer, step_size=CFG.target_model.learning_rate_step_size, gamma=CFG.target_model.learning_rate_gamma)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    epochs = CFG.target_model.epochs
    early_stop = EarlyStopPatience(CFG.target_model.early_stop_patience)

    best_valid_acc = 0.0
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        train_acc = correct / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0

        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
        
        val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = correct / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0

        plateau_scheduler.step(val_acc)
        step_scheduler.step()

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Checkpoint
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            best_valid_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"   -> Saved new best model with val acc: {val_acc:.4f}")

        if early_stop(val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return best_valid_acc, best_valid_loss