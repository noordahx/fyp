from dataclasses import dataclass, field
from typing import List, Optional

import yaml

@dataclass
class ShadowAttackConfig:
    num_shadows: int = 2
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.001
    shadow_train_size: int = 1000
    shadow_eval_size: int = 500


@dataclass
class LossAttackConfig:
    threshold: float = 0.5


@dataclass
class PopulationAttackConfig:
    num_reference: int = 2
    reference_size: int = 1000
    
@dataclass
class ModelConfig:
    architecture: str = 'resnet18' # get it from torchvision.models
    num_classes: int = 10
    pretrained: bool = False
    dropout_rate: float = 0.5
    
    def __post_init__(self):
        assert (self.dropout_rate >= 0 and self.dropout_rate <= 1)
        
@dataclass
class DataConfig:
    dataset: str = 'cifar10'
    train_batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 2
    train_size: Optional[int] = None  # For subset training
    test_size: Optional[int] = None   # For subset testing

@dataclass
class AttackConfig:
    attack_type: str = 'shadow'
    features: List[str] = field(default_factory=lambda: ["max_prob", "true_class_prob", "entropy"])
    shadow: ShadowAttackConfig = field(default_factory=ShadowAttackConfig)
    loss: LossAttackConfig = field(default_factory=LossAttackConfig)
    population: PopulationAttackConfig = field(default_factory=PopulationAttackConfig)

@dataclass
class DefenseConfig:
    method: str = 'standard'
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    max_physical_batch_size: int = 128  # New parameter for Opacus memory management

@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001

@dataclass
class OutputConfig:
    save_dir: str = "./results"
    visualizations: bool = True

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        if 'attack' in config_dict:
            attack_dict = config_dict['attack'].copy()
            
            shadow_config = ShadowAttackConfig()
            if 'shadow' in attack_dict:
                shadow_config = ShadowAttackConfig(**attack_dict.pop('shadow'))
            
            loss_config = LossAttackConfig()
            if 'loss' in attack_dict:
                loss_config = LossAttackConfig(**attack_dict.pop('loss'))
            
            population_config = PopulationAttackConfig()
            if 'population' in attack_dict:
                population_config = PopulationAttackConfig(**attack_dict.pop('population'))
            
            config.attack = AttackConfig(
                shadow=shadow_config,
                loss=loss_config,
                population=population_config,
                **attack_dict
            )
        
        if 'defense' in config_dict:
            config.defense = DefenseConfig(**config_dict['defense'])
            
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
            
        if 'output' in config_dict:
            config.output = OutputConfig(**config_dict['output'])
        
        for k, v in config_dict.items():
            if k not in ['model', 'data', 'attack', 'defense', 'training', 'output'] and hasattr(config, k):
                setattr(config, k, v)
        
        return config


def load_config_from_yaml(path: str) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(path)