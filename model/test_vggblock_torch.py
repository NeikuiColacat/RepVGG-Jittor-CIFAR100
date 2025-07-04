from model.RepVGG_block_torch import RepVGG_Block
import torch

def test_case(in_c, out_c, groups, stride, H, W):
    print(f"Test: in_c={in_c}, out_c={out_c}, groups={groups}, stride={stride}, input=({H},{W})")
    block = RepVGG_Block(in_c, out_c, groups, stride)
    block.eval()
    x = torch.randn(2, in_c, H, W)
    y1 = block(x)
    block.convert_to_infer()
    block.eval()
    y2 = block(x)
    diff = torch.abs(y1 - y2).sum().item()
    print("diff:", diff)
    print("output shape:", y1.shape, y2.shape)
    print("="*40)

# 标准分组
test_case(4, 4, 2, 1,  32, 32)
# 不同输入输出通道
test_case(8, 8, 4, 1,  16, 16)
# stride > 1
test_case(4, 4, 2, 2,  32, 32)
# padding = 0
test_case(4, 4, 2, 1,  32, 32)
# groups=1（普通卷积）
test_case(4, 4, 1, 1,  32, 32)
# 输入输出通道不等
test_case(6, 12, 2, 1, 28, 28)
# 输入尺寸不是32的倍数
test_case(4, 4, 2, 1,  17, 19)
test_case(4, 4, 2, 1,  224, 224)



