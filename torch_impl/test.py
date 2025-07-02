from model.repVGG_block import RepVGG_Block
import torch

test_block = RepVGG_Block(4, 4, 1, 1, 1)
test_block.eval()

input = torch.randn(1, 4, 5, 5)

output = test_block(input)

test_block.convert_to_infer()


new_out = test_block(input)

diff = torch.abs(new_out - output).sum()

print(diff.item())