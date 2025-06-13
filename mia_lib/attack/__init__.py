"""
Membership Inference Attack implementations.

This module provides various membership inference attack methods including:
- Shadow Model Attack
- Reference Attack  
- Threshold Attack
- Loss-based Attack
- Population Attack
- Leave-one-out Attack
- Distillation Attack
"""

from .attack_shadow import ShadowAttack, train_shadow_models, create_shadow_attack_dataset, train_attack_model
from .attack_reference import ReferenceAttack
from .attack_threshold import ThresholdAttack
from .attack_loss import LossBasedAttack
from .attack_population import PopulationAttack
from .attack_leave_one_out import LeaveOneOutAttack
from .attack_distillation import DistillationAttack

__all__ = [
    'ShadowAttack',
    'ReferenceAttack', 
    'ThresholdAttack',
    'LossBasedAttack',
    'PopulationAttack',
    'LeaveOneOutAttack',
    'DistillationAttack',
    'train_shadow_models',
    'create_shadow_attack_dataset',
    'train_attack_model'
]
