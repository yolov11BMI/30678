import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader
from my_yolov11_architecture import YOLOv11Model  # 假設你的模型定義在這個文件中

# Load YOLOv11 model based on version
def load_yolo_model(version):
    # version can be 'teacher' or 'student'
    if version == 'teacher':
        # 加載教師模型，可以使用較大的架構版本，例如 `yolo11l`
        return YOLOv11Model(version='l')  # 修改為教師模型的版本
    elif version == 'student':
        # 加載學生模型，可以使用較小的架構版本，例如 `yolo11n`
        return YOLOv11Model(version='n')  # 修改為學生模型的版本

# Pruning Function
def prune_model(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Apply pruning to Conv2d layers
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

# Knowledge Distillation Loss Function
def distillation_loss(student_outputs, teacher_outputs, targets, alpha=0.5, temperature=2.0):
    # Soft targets from teacher model
    soft_teacher_outputs = F.softmax(teacher_outputs / temperature, dim=1)
    # Soft predictions from student model
    soft_student_outputs = F.log_softmax(student_outputs / temperature, dim=1)
    # Distillation loss (KL Divergence)
    distill_loss = F.kl_div(soft_student_outputs, soft_teacher_outputs, reduction='batchmean') * (temperature ** 2)
    # Hard label loss (Cross-Entropy)
    hard_loss = F.cross_entropy(student_outputs, targets)
    # Combined loss
    return alpha * hard_loss + (1 - alpha) * distill_loss

# Training Loop
def train_student_model(teacher_model, student_model, dataloader, num_epochs=10, learning_rate=0.001):
    # Set teacher model to evaluation mode
    teacher_model.eval()
    # Define optimizer for student model
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            # Forward pass through student model
            student_outputs = student_model(inputs)
            # Forward pass through teacher model (no gradients needed)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            # Compute distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs, targets)
            # Backpropagation
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Main Execution
if __name__ == "__main__":
    # Load teacher and student models
    teacher_model = load_yolo_model('teacher')
    student_model = load_yolo_model('student')

    # Prune the student model to reduce complexity
    prune_amount = 0.2  # Adjust pruning amount as needed
    pruned_student_model = prune_model(student_model, amount=prune_amount)

    # Load your dataset
    # Assume we have a dataset and DataLoader ready
    train_dataset = ...  # Replace with actual dataset
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Train the pruned student model using knowledge distillation
    num_epochs = 20
    learning_rate = 0.001
    train_student_model(teacher_model, pruned_student_model, train_loader, num_epochs, learning_rate)

    # Save the trained pruned and distilled student model
    torch.save(pruned_student_model.state_dict(), "pruned_distilled_student_model.pth")
