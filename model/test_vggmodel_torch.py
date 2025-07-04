from model.RepVGG_model_torch import RepVGG_Model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def test_repvgg_cifar10():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    print("Creating RepVGG model for CIFAR-10...")
    model = RepVGG_Model(
        channel_scale_A=0.5,    
        channel_scale_B=1.0,
        group_conv=1,          
        classify_classes=10,     
        model_type='A'
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n=== Training Phase ===")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    print("\n=== Testing Original Model ===")
    model.eval()
    test_accuracy_orig = test_model(model, test_loader, device)
    
    print("\n=== Verifying Conversion ===")
    verify_conversion(model, device)
    
    print("\n=== Converting to Inference Mode ===")
    model.convert_to_infer()
    
    print("=== Testing Converted Model ===")
    model.eval()
    test_accuracy_infer = test_model(model, test_loader, device)
    
    print(f"\n=== Results ===")
    print(f"Original accuracy: {test_accuracy_orig:.2f}%")
    print(f"Inference accuracy: {test_accuracy_infer:.2f}%")
    print(f"Difference: {abs(test_accuracy_orig - test_accuracy_infer):.4f}%")

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def verify_conversion(model, device):
    """验证转换前后一致性"""
    model.eval()
    test_input = torch.randn(4, 3, 32, 32).to(device)
    
    with torch.no_grad():
        output_before = model(test_input)
    
    import copy
    model_copy = copy.deepcopy(model)
    model_copy.convert_to_infer()
    
    with torch.no_grad():
        output_after = model_copy(test_input)
    
    diff = torch.abs(output_before - output_after).max().item()
    print(f"diff : {diff:.6f}")

if __name__ == "__main__":
    test_repvgg_cifar10()