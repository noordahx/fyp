from abc import ABC, abstractmethod
from src.attacks.attack_loss import LossBasedAttack
from src.attacks.attack_shadow import ShadowAttack
from src.attacks.attack_threshold import ThresholdAttack
from src.attacks.attack_population import PopulationAttack
from src.new_utils.config import Config
from typing import Dict, Any
import torch

class BaseAttack(ABC):
    """Base class for all MIA types"""
    
    def __init__(self, config: Config):
        self.config = config
        self.name = "base"
        self.results = {}

    @abstractmethod
    def run(self, target_model: torch.nn.Module, train_dataset, test_dataset, device: str) -> Dict[str, Any]:
        """Run the attack and return results."""
        pass

def get_attack_by_name(attack_name: str, config: Config):
    """Factory function to get attack by name with proper device handling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    attacks = {
        "shadow": lambda cfg: ShadowAttack(cfg, device),
        "threshold": lambda cfg: ThresholdAttack(cfg, device=device),
        "loss": lambda cfg: LossBasedAttack(device=device),
        "population": lambda cfg: PopulationAttack(cfg, device=device),
    }
    
    if attack_name not in attacks:
        available_attacks = list(attacks.keys())
        raise ValueError(f"Unknown attack type: {attack_name}. Available attacks: {available_attacks}")

    return attacks[attack_name](config)

def run_all_attacks(config: Config, target_model: torch.nn.Module, train_dataset, test_dataset, device: str) -> Dict[str, Dict[str, Any]]:
    """Run all available attacks and return combined results."""
    all_results = {}
    attack_names = ["threshold", "loss", "population", "shadow"]  # Order from simplest to most complex
    
    for attack_name in attack_names:
        try:
            print(f"\n=== Running {attack_name.upper()} Attack ===")
            attack = get_attack_by_name(attack_name, config)
            results = attack.run(target_model, train_dataset, test_dataset, device)
            all_results[attack_name] = results
            print(f"{attack_name.upper()} Attack completed successfully")
        except Exception as e:
            print(f"Error running {attack_name} attack: {str(e)}")
            all_results[attack_name] = {
                'error': str(e),
                'auc': 0.5,
                'accuracy': 0.0,
                'attack_advantage': 0.0
            }
    
    return all_results