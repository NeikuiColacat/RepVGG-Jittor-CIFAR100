import jittor as jt
import jittor.nn as nn
from jittor import init

class RepVGG_Block(nn.Module):

    def __init__(self, in_channels, out_channels, conv_group, stride):
        super().__init__()
        assert(in_channels % conv_group == 0 and out_channels % conv_group == 0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_group = conv_group
        self.stride = stride
        self.infer = False
        self.use_identity =  (in_channels == out_channels and stride == 1)
        self.id_with_bn = None

        def get_conv_with_bn(kernel_size, padding):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=conv_group,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        
        if self.use_identity:
            self.id_with_bn = nn.BatchNorm2d(in_channels)
        self.conv_1x1_with_bn = get_conv_with_bn(1, 0)
        self.conv_3x3_with_bn = get_conv_with_bn(3, 1)
        self.infer_conv = None
    
    def execute(self, input):
        if self.infer:
            return nn.relu(self.infer_conv(input))
        
        output = self.conv_1x1_with_bn(input) + self.conv_3x3_with_bn(input)
        if self.use_identity: 
            output = output + self.id_with_bn(input)

        output = nn.relu(output)
        return output 
    
    def convert_to_infer(self):
        self.infer = True
        self.infer_conv = self.combine_3_branch()

        delattr(self,'conv_1x1_with_bn')
        delattr(self,'conv_3x3_with_bn')
        if self.use_identity : delattr(self,'id_with_bn')

        jt.clean()

    def combine_3_branch(self):
        weight_3x3, bias_3x3 = self.combine_conv_bn(self.conv_3x3_with_bn)
        weight_1x1, bias_1x1 = self.combine_conv_bn(self.conv_1x1_with_bn)

        # Jittor 中使用 jt.nn.pad 进行填充
        weight_1x1 = jt.nn.pad(weight_1x1, [1, 1, 1, 1])

        new_weight = weight_3x3 + weight_1x1
        new_bias = bias_1x1 + bias_3x3

        if self.use_identity: 
            weight_id, bias_id = self.combine_conv_bn(self.id_with_bn)
            weight_id = jt.nn.pad(weight_id, [1, 1, 1, 1])

            new_weight = new_weight + weight_id
            new_bias = new_bias + bias_id
        
        new_conv = nn.Conv2d(
            self.in_channels, 
            self.out_channels, 
            3, 
            stride=self.stride, 
            groups=self.conv_group, 
            padding=1, 
            bias=True
        )
        new_conv.weight = new_weight
        new_conv.bias = new_bias

        return new_conv
        
    def combine_conv_bn(self, branch):
        if isinstance(branch, nn.Sequential):
            conv, bn = branch[0], branch[1]
            conv_weight = conv.weight
        else:
            bn = branch
            channels = self.in_channels
            dtype= bn.weight.dtype 

            conv_weight = jt.zeros((channels, channels//self.conv_group, 1, 1), dtype=dtype)
            idx = jt.arange(channels)
            conv_weight[idx, idx%(channels//self.conv_group), 0, 0] = 1

        gamma, beta, mu, sigma = bn.weight, bn.bias, bn.running_mean, (bn.running_var + bn.eps) ** 0.5

        new_conv_weight = conv_weight * (gamma / sigma).reshape((-1, 1, 1, 1))
        new_conv_bias = beta - (mu * gamma / sigma)

        return new_conv_weight, new_conv_bias