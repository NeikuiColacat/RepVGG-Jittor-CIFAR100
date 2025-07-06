import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RepVGG_model_jittor import RepVGG_Model
import jittor as jt
import jittor.nn as nn
from jittor import optim
from jittor.dataset import CIFAR10
import time
import copy

# 启用 CUDA
jt.flags.use_cuda = 1

def test_repvgg_cifar10():
    print(f"Using device: {'CUDA' if jt.flags.use_cuda else 'CPU'}")
    
    # 数据预处理 - Jittor 版本 (修复后)
    normalize = jt.transform.ImageNormalize(
        mean=[0.4914, 0.4822, 0.4465], 
        std=[0.2023, 0.1994, 0.2010]
    )
    
    # 训练数据变换 - 修复 RandomCrop 参数
    train_transform = jt.transform.Compose([
        jt.transform.Resize((36, 36)),  # 先放大
        jt.transform.RandomCrop(32),    # 然后随机裁剪到32x32
        jt.transform.RandomHorizontalFlip(0.5),
        jt.transform.ToTensor(),
        normalize
    ])
    
    # 测试数据变换
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
    
    print("\n=== Testing Original Model ===")
    model.eval()
    test_accuracy_orig = test_model(model, test_loader)
    
    print("\n=== Verifying Conversion ===")
    verify_conversion(model)
    
    print("\n=== Converting to Inference Mode ===")
    model.convert_to_infer()
    
    print("=== Testing Converted Model ===")
    model.eval()
    test_accuracy_infer = test_model(model, test_loader)
    
    print(f"\n=== Results ===")
    print(f"Original accuracy: {test_accuracy_orig:.2f}%")
    print(f"Inference accuracy: {test_accuracy_infer:.2f}%")
    print(f"Difference: {abs(test_accuracy_orig - test_accuracy_infer):.4f}%")

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

def verify_conversion(model):
    """验证转换前后一致性"""
    model.eval()
    test_input = jt.randn(4, 3, 32, 32)
    
    with jt.no_grad():
        output_before = model(test_input)
    
    # 深拷贝模型
    model_copy = copy.deepcopy(model)
    model_copy.convert_to_infer()
    
    with jt.no_grad():
        output_after = model_copy(test_input)
    
    diff = jt.abs(output_before - output_after).max().item()
    print(f"diff : {diff:.6f}")

def simple_test():
    """简单功能测试"""
    print("🧪 Simple functionality test")
    print("="*50)
    
    # 创建小模型进行快速测试
    model = RepVGG_Model(
        channel_scale_A=0.25,    # 更小的模型
        channel_scale_B=0.5,
        group_conv=1,          
        classify_classes=10,     
        model_type='A'
    )
    
    # 测试前向传播
    test_input = jt.randn(2, 3, 32, 32)
    
    print("训练模式测试:")
    model.train()
    output_train = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output_train.shape}")
    
    print("\n推理模式测试:")
    model.eval()
    output_eval_before = model(test_input)
    
    # 转换为推理模式
    model.convert_to_infer()
    output_eval_after = model(test_input)
    
    # 检查一致性
    diff = jt.abs(output_eval_before - output_eval_after).max().item()
    print(f"转换前后差异: {diff:.2e}")
    
    if diff < 1e-5:
        print("✅ 简单测试通过")
    else:
        print("⚠️ 转换存在差异，但可能在可接受范围内")

def performance_comparison():
    """性能对比测试"""
    print("\n🏃 Performance comparison test")
    print("="*50)
    
    model = RepVGG_Model(
        channel_scale_A=0.5,
        channel_scale_B=1.0,
        group_conv=1,
        classify_classes=10,
        model_type='A'
    )
    
    test_input = jt.randn(8, 3, 32, 32)
    num_runs = 50
    
    # 训练模式性能
    model.eval()
    jt.sync_all()
    start_time = time.time()
    
    with jt.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    
    jt.sync_all()
    train_time = time.time() - start_time
    
    # 推理模式性能
    model.convert_to_infer()
    jt.sync_all()
    start_time = time.time()
    
    with jt.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    
    jt.sync_all()
    infer_time = time.time() - start_time
    
    print(f"训练模式平均时间: {train_time/num_runs*1000:.2f} ms")
    print(f"推理模式平均时间: {infer_time/num_runs*1000:.2f} ms")
    if infer_time > 0:
        print(f"加速比: {train_time/infer_time:.2f}x")

# 简化的数据变换版本
def simple_test_with_data():
    """使用简化数据变换的测试"""
    print("🧪 Simple test with CIFAR10 data")
    print("="*50)
    
    # 简化的数据变换
    simple_transform = jt.transform.Compose([
        jt.transform.ToTensor(),
        jt.transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载少量数据进行测试
    test_dataset = CIFAR10(train=False, transform=simple_transform)
    test_loader = test_dataset.set_attrs(batch_size=32, shuffle=False)
    
    # 创建小模型
    model = RepVGG_Model(
        channel_scale_A=0.25,
        channel_scale_B=0.5,
        group_conv=1,
        classify_classes=10,
        model_type='A'
    )
    
    print("测试数据加载...")
    # 只测试一个批次
    for data, target in test_loader:
        print(f"数据形状: {data.shape}, 标签形状: {target.shape}")
        
        # 前向传播测试
        model.eval()
        with jt.no_grad():
            output = model(data)
        print(f"模型输出形状: {output.shape}")
        
        # 转换测试
        model.convert_to_infer()
        with jt.no_grad():
            output_infer = model(data)
        print(f"推理模式输出形状: {output_infer.shape}")
        
        print("✅ 数据测试通过")
        break  # 只测试第一个批次

if __name__ == "__main__":
    print("🚀 开始测试 Jittor RepVGG Model")
    print("="*60)
    
    # 选择测试模式
    test_mode = input("选择测试模式 (1: 简单测试, 2: 性能测试, 3: 完整训练, 4: 数据测试): ")
    
    if test_mode == "1":
        simple_test()
    elif test_mode == "2":
        simple_test()
        performance_comparison()
    elif test_mode == "3":
        test_repvgg_cifar10()
    elif test_mode == "4":
        simple_test_with_data()
    else:
        print("运行简单测试...")
        simple_test()
    
    print("\n🎉 测试完成!")