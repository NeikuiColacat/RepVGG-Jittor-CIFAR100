import torch 
import yaml
from torch import nn

def filter_param(model:nn.Module):
    weight_decay_param = []
    no_decay_param = []

    get_param_list = lambda module : [
        param for param in module.parameters(False)
        if param.requires_grad
    ]

    for module in model.modules():
        if isinstance(module , (nn.Conv2d,nn.Linear)) : 
            weight_decay_param = weight_decay_param + get_param_list(module)
        else :
            no_decay_param = no_decay_param + get_param_list(module)
    
    return [
        {'params':weight_decay_param},
        {'params':no_decay_param,'weight_decay':0.}
    ]


def get_optimizer(model,config):
    
    weight_decay = config['weight_decay']
    lr = config['lr']
    momentum = config['momentum'] 

    res = torch.optim.SGD(
        filter_param(model),
        lr = lr,
        momentum= momentum,
        nesterov=True,
        weight_decay=weight_decay
    )

    return res

def get_scheduler(optimizer,config):
    
    epochs = config['epochs']

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0
    )

    
