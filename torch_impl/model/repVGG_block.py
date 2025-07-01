import torch.nn as nn
import torch.nn.functional as F


class RepVGG_Block(nn.Module):

    def __init__(self, in_channels, out_channels, conv_group,activate_func):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_group = conv_group
        self.activate_func = activate_func

        get_conv_with_bn = (
            lambda in_channels, out_channels, kernel_size, groups=conv_group: nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    groups=conv_group,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.res_add_with_bn = nn.BatchNorm2d(in_channels)
        self.conv_1x1_with_bn = get_conv_with_bn(in_channels, out_channels, 1)
        self.conv_3x3_with_bn = get_conv_with_bn(in_channels, out_channels, 3)
    
    def forward(self,input):
        output = self.res_add_with_bn(input) + self.conv_1x1_with_bn(input) + self.conv_3x3_with_bn(input)
        output = self.activate_func(output)
        return output 



