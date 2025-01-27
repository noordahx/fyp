# very simplified version of PATE method
# TODO: Add real PATE https://arxiv.org/abs/1610.05755

import torch
import torch.nn as nn
import numpy as np

def train_with_pate(model, train_loader, val_loader, config, device):
    """
    Illustrative example of PATE method.
    Typically, have multiple teachers and students that vote on labels to privately lable a 'student' dataset.
    """
    teachers_count = config["pate"]["teachers_count"]
    laplace_scale = config["pate"]["laplace_scale"]

    # 1. Train teacher models on partitions of the data (not shown here).
    #    Suppose you already have teacher models in a list: teacher_models = [...]
    teacher_models = train_multiple_teachers(train_loader, teachers_count, device)

    # 2. Aggregate teacher predictions with noise (Laplace or Gaussian).
    #    Then create a labeled dataset for the student model:
    student_dataset = generate_student_labels(teacher_models, train_loader, laplace_scale, device)

    # 3. Train your main "student" model on the aggregated labels
    #    (similar to normal training)
    student_model = train_student(model, student_dataset, val_loader, dp_config, device)
    return student_model


def train_multiple_teachers(train_loader, teachers_count, device):
    # Pseudocode. In practice, you'd partition train_loader's data for each teacher.
    teacher_models = []
    for i in range(teachers_count):
        # create & train each teacher model...
        # teacher_models.append(...)
        pass
    return teacher_models


def generate_student_labels(teacher_models, unlabeled_loader, laplace_scale, device):
    # Each teacher predicts => we do a noisy aggregation of teacher votes
    # Return new dataset with (image, aggregated_label)
    pass

def train_student(student_model, student_dataset, val_loader, dp_config, device):
    # train the student model with the aggregated labels
    pass