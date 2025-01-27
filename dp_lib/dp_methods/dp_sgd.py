# DP-SGD with Opacus for now
# TODO: implemnt DP-SGD by myself

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine

def train_with_dp_sgd(model, train_loader, val_loader, config, device):
    """
    Train the model using DP-SGD via Opacus.
    config: subfield with dp_sgd
    """
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]

    noise_multiplier = config["dp_sgd"]["noise_multiplier"]
    max_grad_norm = config["dp_sgd"]["max_grad_norm"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Attach Opacus PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=config["dp"]["target_epsilon"],
        target_delta=config["dp"]["target_delta"],
        max_grad_norm=max_grad_norm,
        epochs=epochs
    )

    model = model.to(device)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
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

        # Eval
        val_loss, val_acc = evaluate(model, val_loader, device)

        epsilon_spent, best_alpha = optimizer.privacy_engine.get_privacy_spent(config["dp"]["target_delta"])
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"(ε = {epsilon_spent:.2f}, α={best_alpha:.2f})")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
    return model, best_val_acc

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)
