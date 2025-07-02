import torch.nn as nn
import torch.nn.functional as F
import torch

class RepVGG_Block(nn.Module):

    def __init__(self, in_channels, out_channels, conv_group, stride,padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_group = conv_group
        self.stride = stride
        self.pad = padding
        self.infer = False

        get_conv_with_bn = lambda kernel_size , padding : nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride = stride,
                padding=padding,
                groups=conv_group,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        
        if in_channels == out_channels and stride == 1 : self.id_with_bn= nn.BatchNorm2d(in_channels)
        self.conv_1x1_with_bn = get_conv_with_bn(1,padding-1)
        self.conv_3x3_with_bn = get_conv_with_bn(3,padding)
        self.infer_conv = None
    
    def forward(self,input):
        if (self.infer):
            return F.relu(self.infer_conv(input))
        
        output = self.conv_1x1_with_bn(input) + self.conv_3x3_with_bn(input)
        if hasattr(self,'id_with_bn'):
            output = output + self.id_with_bn(input)

        output = F.relu(output) 
        return output 
    
    def convert_to_infer(self):
        self.infer = True
        self.infer_conv = self.combine_3_branch()

        del self.conv_1x1_with_bn
        del self.conv_3x3_with_bn
        if hasattr(self,'id_with_bn') : del self.id_with_bn

    def combine_3_branch(self):
        weight_3x3, bias_3x3 = self.combine_conv_bn(self.conv_3x3_with_bn)
        weight_1x1, bias_1x1 = self.combine_conv_bn(self.conv_1x1_with_bn)

        weight_1x1 = F.pad(weight_1x1,[1,1,1,1])

        new_weight = weight_3x3 + weight_1x1
        new_bias = bias_1x1 + bias_3x3

        if hasattr(self,'id_with_bn') : 
            weight_id , bias_id = self.combine_conv_bn(self.id_with_bn)
            weight_id = F.pad(weight_id,[1,1,1,1])

            print(weight_id.shape , '\n' , weight_1x1.shape)
            exit()

            new_weight = new_weight + weight_id
            new_bias = new_bias + bias_id
        
        new_conv = nn.Conv2d(self.in_channels,self.out_channels,3,self.stride,bias=True)
        new_conv.weight.data = new_weight
        new_conv.bias.data = new_bias
        
        return new_conv
        
    def combine_conv_bn(self,branch):
        if isinstance(branch,nn.Sequential) :
            conv , bn = branch
            conv_weight = conv.weight
        else :
            bn = branch
            channels = self.in_channels
            conv_weight = torch.ones((channels,channels,1,1),dtype=bn.weight.dtype,device=bn.weight.device) 

        gamma , beta , mu , sigma = bn.weight,bn.bias,bn.running_mean,(bn.running_var + bn.eps) ** 0.5

        new_conv_weight = conv_weight * (gamma / sigma).reshape(-1,1,1,1)
        new_conv_bias = beta - (mu * gamma / sigma).reshape(-1,1,1,1)

        return new_conv_weight , new_conv_bias


        