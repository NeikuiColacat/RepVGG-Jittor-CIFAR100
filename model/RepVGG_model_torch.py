from model.RepVGG_block_torch import RepVGG_Block
import torch.nn as nn
import torch.nn.functional as F
import torch


class RepVGG_Model(nn.Module):
    
    def __init__(self, channel_scale_A, channel_scale_B, group_conv , classify_classes , model_type):
        super().__init__()
        self.channel_scale_A = channel_scale_A
        self.channel_scale_B = channel_scale_B
        self.group_conv = group_conv 
        self.classify_classes = classify_classes
        self.model_type = model_type
        self.layer_idx = 0
        self.pre_stage_channels = 3
        self.conv_group_idx = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

        assert(model_type == 'A' or model_type == 'B')

        channel_scale_base = [64,64,128,256,512] 
        channel_scale_base = [num // 2 for num in channel_scale_base]

        blocks_per_stage = [1,2,4,14,1] if model_type == 'A' else [1,4,6,16,1]

        RepVGG_Blcoks = []
        for stage_id in range(5):
            channel_size = int(channel_scale_A * channel_scale_base[stage_id])
            if stage_id == 0 : channel_size = min(channel_size , channel_scale_base[0])
            if stage_id == 4 : channel_size = int(channel_scale_B * channel_scale_base[stage_id])

            stride = 2 if stage_id > 1 else 1
            RepVGG_Blcoks = RepVGG_Blcoks + self.get_a_stage(channel_size,blocks_per_stage[stage_id],stride)
            
        self.RepVGG_Blocks = nn.ModuleList(RepVGG_Blcoks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(channel_scale_B * channel_scale_base[-1]) , self.classify_classes)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def get_a_stage(self, channels_size, blocks_size,stride=1):
        res = []
        for i in range(blocks_size):
            res.append(
                RepVGG_Block(self.pre_stage_channels,
                             channels_size,
                             self.group_conv if self.layer_idx in self.conv_group_idx else 1,
                             stride if i == 0 else 1
                             )
            )

            self.layer_idx += 1
            self.pre_stage_channels = channels_size
        return res
    
    def forward(self,input):
        output = input
        for block in self.RepVGG_Blocks : 
            output = block(output)

        output = self.pool(output)
        output = self.linear(output.reshape(output.shape[0],-1))
        return output

    def convert_to_infer(self):
        for block in self.RepVGG_Blocks:
            block.convert_to_infer()








    