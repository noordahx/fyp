# generic training loops for target/shadow models

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from mia_lib.utils import EarlyStopPatience

def train_model(model, train_loader, val_loader, config, device, save_path):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"] 
        )

    epochs = config["training"]["epochs"]

    early_stop = EarlyStopPatience(config["training"]["early_stop_patience"])

    best_valid_acc = 0.0
    beat_valis_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        # Validate
        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    correct += (outputs.argmax(1) == labels).sum().item()
        
        val_loss = total_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Checkpoint
        if val_acc > best_valid_acc or (val_acc == best_valid_acc and val_loss < best_valid_loss):
            best_valid_acc = val_acc
            best_valid_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"   -> Saved new bset model with val acc: {val_acc:.4f}")
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
        
        if early_stop(val_acc):
            print("Early stopping triggered.")
            break
    
    return best_valid_acc, best_valid_loss