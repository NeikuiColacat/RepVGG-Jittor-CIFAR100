import jittor as jt
import jittor.nn as nn
import os
import time

import yaml
from model.RepVGG_model_jittor import RepVGG_Model 
from train.train_jittor import train_one_epoch , val_one_epoch
from train.data_loader_jittor import get_cifar100_dataloaders , get_imagenet_dataloaders
from train.optimizer_jittor import get_optimizer, get_scheduler
from utils.train_logger import Logger
from jittor.models import resnet18
import model.ResNet_model_jittor

def create_model(config):

    model_name = config['model_name']
    if 'resnet' in model_name :
        return model.ResNet_model_jittor.resnet18()

    else:
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
    jt.save(state, filepath)

def load_chk_point(model, optimizer, checkpoint_path):
    assert(os.path.exists(checkpoint_path))
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = jt.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint.get('acc', 0.0)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def execute(self, pred, target):
        n_class = pred.shape[1]
        log_probs = nn.log_softmax(pred, dim=1)
        with jt.no_grad():
            true_dist = jt.zeros_like(pred)
            true_dist += self.smoothing / (n_class - 1)
            fill_value = jt.array([1.0 - self.smoothing]).float32()   
            true_dist = true_dist.scatter(1, target.unsqueeze(1), fill_value)
        return (-true_dist * log_probs).sum(dim=1).mean()

def train_model(config_path, resume_path = None):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    epochs = config['epochs']
    save_dir = config['chk_point_dir']
    model_name = config['model_name']

    model = create_model(config)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    if 'cifar' in model_name:
        train_loader, val_loader = get_cifar100_dataloaders(config)
    else :
        train_loader, val_loader = get_imagenet_dataloaders(config)

    loss_func = LabelSmoothingCrossEntropy(smoothing=0.1) 
    start_epoch = 0
    best_top1_acc = 0.0

    logger = Logger(
        log_dir=config['log_dir'],
        framework='jittor',
        model_name=model_name
    )
    

    if resume_path: 
        start_epoch, best_top1_acc = load_chk_point(model, optimizer, resume_path)
        scheduler.last_epoch = start_epoch

    get_save_dict  = lambda : ({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': val_top1_acc,
            'config': config,
        }
    )
    for epoch in range(start_epoch, epochs):
        
        logger.start_epoch_monitoring()
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_func=loss_func,
            epoch_idx=epoch ,
        )
        
        val_loss, val_top1_acc, val_top5_acc = val_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_func=loss_func
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
            save_chk_point(get_save_dict(), save_dir, 'best_model.jt')
        
        if epoch % 10 == 0:
            save_chk_point(get_save_dict(), save_dir, f'checkpoint_epoch_{epoch}.jt')


    save_chk_point(get_save_dict(), save_dir, 'final_model.jt')


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
    jt.misc.set_global_seed(LUCK_NUMBER)
    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1

    main()
    