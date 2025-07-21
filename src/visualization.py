import matplotlib.pyplot as plt

def plot_training_metrics(history, save_path, method="standard", epsilon=None):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    elif 'test_acc' in history:
        plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title(f'{method.upper()} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    elif 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test Loss')
    plt.title(f'{method.upper()} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if 'lr' in history:
        plt.plot(history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
    else:
        # If no learning rate history, show training progress instead
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, [0.01] * len(epochs), 'g--', label='Fixed LR')
        plt.title('Learning Rate (Fixed)')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    if 'epsilon' in history and method != 'standard':
        plt.subplot(2, 2, 4)
        plt.plot(history['epsilon'])
        plt.title('Privacy Budget (ε)')
        plt.xlabel('Epochs')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, 'No Privacy Budget', ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.suptitle(f"Training Metrics - {method.upper()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()
