from RepVGG_block_jittor import RepVGG_Block
import jittor as jt

jt.flags.use_cuda = 1

def test_case(in_c, out_c, groups, stride, H, W):
    print(f"Test: in_c={in_c}, out_c={out_c}, groups={groups}, stride={stride}, input=({H},{W})")
    
    block = RepVGG_Block(in_c, out_c, groups, stride)
    block.eval()
    
    x = jt.randn(2, in_c, H, W)
    
    y1 = block(x)
    
    block.convert_to_infer()
    block.eval()
    
    y2 = block(x)
    
    diff = jt.abs(y1 - y2).sum().item()
    
    assert (y1.shape == y2.shape)
    
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

# 大通道数测试
test_case(64, 128, 1, 2, 56, 56)

# 单通道测试
test_case(1, 8, 1, 1, 28, 28)
