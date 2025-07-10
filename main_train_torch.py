import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import yaml
import random
from tqdm import tqdm

from model.RepVGG_model_torch import RepVGG_Model
from train.train_torch import  train_one_epoch, val_one_epoch
from train.data_loader_torch import get_cifar100_dataloaders , get_imagenet_dataloaders
from train.optimizer_torch import get_optimizer, get_scheduler
from utils.train_logger import Logger

def create_model(config):

    model_name = config['model_name']
    if 'resnet' in model_name :
        # import torchvision.models as models 
        # if 'cifar' in model_name:
        #     return models.resnet18(pretrained=False, num_classes=100)  
        # else :
        #     return models.resnet18(pretrained=False, num_classes=1000)
        from model import ResNet_model_torch
        return ResNet_model_torch.resnet18()

    else:
        # import model.tmp_model 
        # return model.tmp_model.create_RepVGG_A0(num_classes=100)
        return RepVGG_Model(
            channel_scale_A=config['scale_a'],
            channel_scale_B=config['scale_b'],
            group_conv=config['group_conv'],
            classify_classes=config['num_classes'],
            model_type=config['model_type']
        )

def save_chk_point(state, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)


def load_chk_point(model, optimizer, scheduler, scaler,  checkpoint_path , USE_AMP):
    assert(os.path.exists(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if USE_AMP:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['acc']

def train_model(config_path, resume_path = None):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    USE_AMP = config['use_amp'] 

    epochs = config['epochs']
    save_dir = config['chk_point_dir']
    model_name = config['model_name']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(config)
    model = model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    if 'cifar' in model_name: 
        train_loader, val_loader = get_cifar100_dataloaders(config)
    else :
        train_loader, val_loader = get_imagenet_dataloaders(config)

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    start_epoch = 0
    best_top1_acc = 0.0

    if USE_AMP:
        scaler = torch.amp.GradScaler("cuda")
    else :
        scaler = None

    logger = Logger(
        log_dir=config['log_dir'],
        framework='torch',
        model_name=model_name
    )
    

    if resume_path: 
        start_epoch, best_top1_acc = load_chk_point(
            model, optimizer, scheduler, scaler, resume_path, USE_AMP)
    get_save_dict  = lambda : ({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'acc': val_top1_acc,
            'config': config,
            'scaler_state_dict' : scaler.state_dict() if USE_AMP else None
        }
    )

    for epoch in range(start_epoch, epochs):
        
        logger.start_epoch_monitoring()
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_lodaer=train_loader, 
            optimizer=optimizer,
            loss_func=loss_func,
            epoch_idx=epoch ,
            scaler = scaler
        )
        
        val_loss, val_top1_acc, val_top5_acc = val_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_func=loss_func,
            USE_AMP=USE_AMP
        )

        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc = train_acc,
            val_loss=val_loss,
            top1_acc=val_top1_acc,
            top5_acc=val_top5_acc,
            epoch_time=epoch_time,
            learning_rate=cur_lr,
        )
        

        if val_top1_acc > best_top1_acc:
            best_top1_acc = val_top1_acc 
            save_dict = get_save_dict()
            save_chk_point(save_dict, save_dir, 'best_model.pth')
        
        if epoch % 10 == 0:
            save_dict = get_save_dict()
            save_chk_point(save_dict, save_dir, f'checkpoint_epoch_{epoch}.pth')


    save_dict = get_save_dict()
    save_chk_point(save_dict,save_dir,'final_model.pth')


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    config_path , resume_path = args.config , args.resume
    assert (os.path.exists(config_path) and os.path.isfile(config_path))
    if resume_path : assert (os.path.exists(args.resume) and os.path.isfile(args.resume))

    train_model(
        config_path=config_path,
        resume_path=resume_path
    )
    
if __name__ == "__main__":

    LUCK_NUMBER = 998244353

    random.seed(LUCK_NUMBER)
    np.random.seed(LUCK_NUMBER)    
    torch.manual_seed(LUCK_NUMBER)
    torch.cuda.manual_seed(LUCK_NUMBER)
    torch.cuda.manual_seed_all(LUCK_NUMBER)

    torch.set_float32_matmul_precision('high')

    main()
    