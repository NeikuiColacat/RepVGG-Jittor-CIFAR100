import sys
import os

from RepVGG_model_jittor import RepVGG_Model
import jittor as jt
import jittor.nn as nn
from jittor import optim
from jittor.dataset import CIFAR10

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.train_logger import Logger

# 启用 CUDA
jt.flags.use_cuda = 1

def test_repvgg_cifar10():
    logger = Logger('./test_with_cifar_jittor/','jittor','jittor_repvgg')
    
    normalize = jt.transform.ImageNormalize(
        mean=[0.4914, 0.4822, 0.4465], 
        std=[0.2023, 0.1994, 0.2010]
    )
    
    train_transform = jt.transform.Compose([
        jt.transform.Resize((36, 36)),  # 先放大
        jt.transform.RandomCrop(32),    # 然后随机裁剪到32x32
        jt.transform.RandomHorizontalFlip(0.5),
        jt.transform.ToTensor(),
        normalize
    ])
    
    test_transform = jt.transform.Compose([
        jt.transform.ToTensor(),
        normalize
    ])
    
    # 加载数据集
    train_dataset = CIFAR10(train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(train=False, transform=test_transform)
    
    train_loader = train_dataset.set_attrs(batch_size=128, shuffle=True)
    test_loader = test_dataset.set_attrs(batch_size=256, shuffle=False)
    
    print("Creating RepVGG model for CIFAR-10...")
    model = RepVGG_Model(
        channel_scale_A=0.5,    
        channel_scale_B=1.0,
        group_conv=1,          
        classify_classes=10,     
        model_type='A'
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if hasattr(p, 'numel'))
    print(f"Model parameters: {total_params:,}")
    
    print("\n=== Training Phase ===")
    num_epochs = 20
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        logger.start_epoch_monitoring()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 前向传播
            output = model(data)
            loss = nn.cross_entropy_loss(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            pred = jt.argmax(output, dim=1)[0]
            correct += (pred == target).sum().item()
            total += target.shape[0]
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

        logger.log_epoch(epoch,total_loss,accuracy,123,123,123,123,123)

    
    print("\n=== Testing Original Model ===")
    model.eval()
    test_accuracy_orig = test_model(model, test_loader)
    
    print("\n=== Converting to Inference Mode ===")
    model.convert_to_infer()
    
    print("=== Testing Converted Model ===")
    model.eval()
    test_accuracy_infer = test_model(model, test_loader)
    
    print(f"\n=== Results ===")
    print(f"Original accuracy: {test_accuracy_orig:.2f}%")
    print(f"Inference accuracy: {test_accuracy_infer:.2f}%")
    print(f"diff: {abs(test_accuracy_orig - test_accuracy_infer):.4f}%")

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with jt.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = jt.argmax(output, dim=1)[0]
            correct += (pred == target).sum().item()
            total += target.shape[0]
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy



if __name__ == "__main__":
    print("test repvgg model jittor")
    print("="*60)
    test_repvgg_cifar10()






    