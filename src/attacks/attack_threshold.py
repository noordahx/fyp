import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, Any

class ThresholdAttack:
    """
    Threshold-based Membership Inference Attack.
    
    This attack assumes that models have higher confidence on training data
    (members) compared to test data (non-members).
    """
    
    def __init__(self, config=None, threshold=0.5, device='cpu'):
        self.threshold = config.attack.threshold if config and hasattr(config.attack, 'threshold') else threshold
        self.device = device
        self.optimal_threshold = None
        
    def calibrate_threshold(self, model, member_loader, non_member_loader) -> float:
        """
        Calibrate the optimal threshold using validation data.
        
        Args:
            model: Target model
            member_loader: DataLoader with member samples
            non_member_loader: DataLoader with non-member samples
            
        Returns:
            Optimal threshold value
        """
        model.eval()
        confidences = []
        labels = []
        
        with torch.no_grad():
            # Get confidences for members
            for images, _ in tqdm(member_loader, desc="Processing members"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                confidences.extend(max_probs.cpu().numpy())
                labels.extend([1] * len(images))
                
            # Get confidences for non-members
            for images, _ in tqdm(non_member_loader, desc="Processing non-members"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                confidences.extend(max_probs.cpu().numpy())
                labels.extend([0] * len(images))
        
        # Find optimal threshold
        confidences = np.array(confidences)
        labels = np.array(labels)
        
        # Print statistics for debugging
        member_confs = confidences[labels == 1]
        non_member_confs = confidences[labels == 0]
        print(f"Member confidence - Mean: {member_confs.mean():.4f}, Std: {member_confs.std():.4f}")
        print(f"Non-member confidence - Mean: {non_member_confs.mean():.4f}, Std: {non_member_confs.std():.4f}")
        
        best_threshold = 0.5
        best_accuracy = 0.0
        
        # Try different thresholds based on percentiles
        thresholds = np.percentile(confidences, np.linspace(10, 90, 100))
        for threshold in thresholds:
            predictions = (confidences > threshold).astype(int)
            accuracy = accuracy_score(labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                
        self.optimal_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.4f}, Accuracy: {best_accuracy:.4f}")
        return best_threshold
        
    def infer_membership(self, model, data_loader, 
                        use_optimal_threshold: bool = True) -> Dict[str, Any]:
        """
        Perform membership inference using threshold attack.
        
        Args:
            model: Target model
            data_loader: DataLoader with samples to classify
            use_optimal_threshold: Whether to use calibrated threshold
            
        Returns:
            Dictionary with attack results
        """
        threshold = self.optimal_threshold if (use_optimal_threshold and self.optimal_threshold) else self.threshold
        
        model.eval()
        confidences = []
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Threshold attack inference"):
                if len(batch) == 2:
                    images, labels = batch
                else:
                    # Handle case where batch might have different structure
                    images = batch[0]
                    labels = batch[1] if len(batch) > 1 else torch.zeros(len(images))
                
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                
                confidences.extend(max_probs.cpu().numpy())
                batch_predictions = (max_probs.cpu().numpy() > threshold).astype(int)
                predictions.extend(batch_predictions)
                true_labels.extend(labels.numpy())
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions) if len(set(true_labels)) > 1 else 0.0
        
        results = {
            'predictions': predictions,
            'confidences': confidences,
            'probabilities': confidences,  # For visualization compatibility
            'true_labels': true_labels,    # For visualization compatibility
            'threshold': threshold,
            'accuracy': accuracy
        }
        
        # Calculate additional metrics if we have both classes
        if len(set(true_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', zero_division=0
            )
            try:
                auc_score = roc_auc_score(true_labels, confidences)
            except ValueError as e:
                print(f"AUC calculation failed: {e}")
                auc_score = 0.5
            
            results.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            })
        
        return results
        
    def analyze_confidence_distribution(self, model, member_loader, 
                                      non_member_loader) -> Dict[str, np.ndarray]:
        """
        Analyze confidence distributions for members vs non-members.
        
        Args:
            model: Target model
            member_loader: DataLoader with member samples
            non_member_loader: DataLoader with non-member samples
            
        Returns:
            Dictionary with confidence distributions
        """
        model.eval()
        member_confidences = []
        non_member_confidences = []
        
        with torch.no_grad():
            # Get member confidences
            for images, _ in tqdm(member_loader, desc="Analyzing member confidences"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                member_confidences.extend(max_probs.cpu().numpy())
                
            # Get non-member confidences
            for images, _ in tqdm(non_member_loader, desc="Analyzing non-member confidences"):
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                non_member_confidences.extend(max_probs.cpu().numpy())
        
        return {
            'member_confidences': np.array(member_confidences),
            'non_member_confidences': np.array(non_member_confidences),
            'member_mean': np.mean(member_confidences),
            'non_member_mean': np.mean(non_member_confidences),
            'member_std': np.std(member_confidences),
            'non_member_std': np.std(non_member_confidences)
        } 
        
    def get_features(self, model, data_loader):
        features = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
                # Get multiple features
                max_probs = torch.max(probs, dim=1)[0]
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                features.append(torch.stack([max_probs, entropy], dim=1).cpu())
        
        return torch.cat(features, dim=0).numpy()
        
    def save_state(self, path):
        """Save attack state including calibrated threshold."""
        state = {
            'threshold': self.threshold,
            'optimal_threshold': self.optimal_threshold
        }
        torch.save(state, path)
    
    def load_state(self, path):
        """Load attack state."""
        state = torch.load(path)
        self.threshold = state['threshold']
        self.optimal_threshold = state['optimal_threshold']
    
    def run(self, target_model, train_dataset, test_dataset, device):
        """
        Run the threshold-based attack.
        
        Args:
            target_model: Target model to attack
            train_dataset: Training dataset (members)
            test_dataset: Test dataset (non-members)
            device: Device to run on
            
        Returns:
            Dictionary with attack results
        """
        self.device = device
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=64,
            shuffle=False, 
            num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=64,
            shuffle=False, 
            num_workers=2
        )
        
        # Calibrate threshold using a subset of data
        print("Calibrating threshold...")
        try:
            # Use smaller subset for calibration to save time
            train_subset = torch.utils.data.Subset(train_dataset, range(min(1000, len(train_dataset))))
            test_subset = torch.utils.data.Subset(test_dataset, range(min(1000, len(test_dataset))))
            
            train_subset_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=False)
            test_subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
            
            self.calibrate_threshold(target_model, train_subset_loader, test_subset_loader)
        except Exception as e:
            print(f"Threshold calibration failed: {e}. Using default threshold.")
            self.optimal_threshold = 0.5  # Default threshold
        
        # Run attack on full datasets
        print("Running threshold attack on training data...")
        train_results = self.infer_membership(target_model, train_loader, use_optimal_threshold=True)
        
        print("Running threshold attack on test data...")
        test_results = self.infer_membership(target_model, test_loader, use_optimal_threshold=True)
        
        # Combine results
        all_predictions = np.concatenate([train_results['predictions'], test_results['predictions']])
        all_confidences = np.concatenate([train_results['confidences'], test_results['confidences']])
        all_true_labels = np.concatenate([
            np.ones(len(train_results['predictions'])),  # Training data = members
            np.zeros(len(test_results['predictions']))   # Test data = non-members
        ])
        
        # Calculate metrics
        accuracy = accuracy_score(all_true_labels, all_predictions)
        try:
            auc = roc_auc_score(all_true_labels, all_confidences)
        except:
            auc = 0.5
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Calculate attack advantage
        train_accuracy = accuracy_score(np.ones(len(train_results['predictions'])), train_results['predictions'])
        test_accuracy = accuracy_score(np.zeros(len(test_results['predictions'])), test_results['predictions'])
        attack_advantage = train_accuracy + test_accuracy - 1
        
        results = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'attack_advantage': attack_advantage,
            'threshold': self.optimal_threshold,
            'all_predictions': all_predictions,
            'all_true_labels': all_true_labels,
            'all_scores': all_confidences
        }
        
        print(f"Threshold Attack Results: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}, Advantage = {attack_advantage:.4f}")
        
        return results