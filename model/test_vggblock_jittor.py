from RepVGG_block_jittor import RepVGG_Block
import jittor as jt

# 启用 CUDA（如果可用）
jt.flags.use_cuda = 1

def test_case(in_c, out_c, groups, stride, H, W):
    print(f"Test: in_c={in_c}, out_c={out_c}, groups={groups}, stride={stride}, input=({H},{W})")
    
    # 创建 RepVGG Block
    block = RepVGG_Block(in_c, out_c, groups, stride)
    block.eval()
    
    # 创建测试输入
    x = jt.randn(2, in_c, H, W)
    
    # 训练模式前向传播
    y1 = block(x)
    
    # 转换为推理模式
    block.convert_to_infer()
    block.eval()
    
    # 推理模式前向传播
    y2 = block(x)
    
    # 计算差异
    diff = jt.abs(y1 - y2).sum().item()
    print("diff:", diff)
    print("output shape:", y1.shape, y2.shape)
    
    # 验证输出形状是否一致
    assert y1.shape == y2.shape, f"输出形状不一致: {y1.shape} vs {y2.shape}"
    
    return diff


# 标准分组
test_case(4, 4, 2, 1, 32, 32)

# 不同输入输出通道
test_case(8, 8, 4, 1, 16, 16)

# stride > 1
test_case(4, 4, 2, 2, 32, 32)

# groups=1（普通卷积）
test_case(4, 4, 1, 1, 32, 32)

# 输入输出通道不等
test_case(6, 12, 2, 1, 28, 28)

# 输入尺寸不是32的倍数
test_case(4, 4, 2, 1, 17, 19)

# ImageNet 尺寸
test_case(4, 4, 2, 1, 224, 224)

# 更复杂的测试用例
print("\n🔍 额外测试用例:")
print("="*50)

# 大通道数测试
test_case(64, 128, 1, 2, 56, 56)

# 深度可分离卷积风格
test_case(32, 32, 32, 1, 32, 32)

# 单通道测试
test_case(1, 8, 1, 1, 28, 28)
