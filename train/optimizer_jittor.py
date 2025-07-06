import jittor as jt
from jittor import nn
from jittor import optim
import yaml
from math import cos, pi

def filter_param(model: nn.Module):
    weight_decay_param = []
    no_decay_param = []

    get_param = lambda module : (
        [param for param in module.parameters() if param.requires_grad]
    )

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_decay_param += get_param(module)
        else:
            no_decay_param += get_param(module)
    
    return [
        {'params': weight_decay_param},
        {'params': no_decay_param, 'weight_decay': 0}
    ]

def get_optimizer(model, config):
    weight_decay = config['weight_decay']
    lr = config['lr']
    momentum = config['momentum']

    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )


def get_scheduler(optimizer, config):
    epochs = config['epochs']
    warmup_epochs = config['warmup_epochs']
    min_lr = config['min_lr']
    lr = config['lr']

    ratio = min_lr / lr

    def get_factor(epoch):
        if epoch < warmup_epochs:
            return max(epoch / warmup_epochs, ratio)
        else:
            scale = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return ratio + (1 - ratio) / 2 * (1 + cos(pi * scale))

    return optim.LambdaLR(optimizer, get_factor)


