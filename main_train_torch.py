import torch
import torch.nn as nn
import os
import sys
import time
import yaml
from tqdm import tqdm

from model.RepVGG_model_torch import RepVGG_Model
from train.train_torch import get_imagenet_dataloaders, train_one_epoch, val_one_epoch
from train.optimizer_torch import get_optimizer, get_scheduler
from utils.train_logger import Logger

def create_model(config):
    model = RepVGG_Model(
        channel_scale_A=config['scale_a'],
        channel_scale_B=config['scale_b'],
        group_conv=config['group_conv'],
        classify_classes=1000,  
        model_type=config['model_type']
    )
    
    return model

def save_chk_point(state, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

def load_chk_point(model, optimizer, scheduler, checkpoint_path):
    assert(os.path.exists(checkpoint_path))
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('best_acc', 0.0)

def train_model(config_path, resume_path = None):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    epochs = config['epochs']
    save_dir = config['chk_point_dir']
    model_name = config['model_name']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config)
    model = model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    loss_func = nn.CrossEntropyLoss()
    train_loader, val_loader = get_imagenet_dataloaders(config)
    start_epoch = 0
    best_top1_acc = 0.0
    logger = Logger(
        log_dir=config['log_dir'],
        framework='torch',
        model_name=model_name
    )
    

    if resume_path: start_epoch, best_top1_acc = load_chk_point(model, optimizer, scheduler,resume_path)
    for epoch in range(start_epoch, epochs):
        print(f"eopch : {epoch}" + "="*80 + "\n")
        
        logger.start_epoch_monitoring()
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_lodaer=train_loader, 
            optimizer=optimizer,
            loss_func=loss_func,
            epoch_idx=epoch 
        )
        
        val_loss, val_top1_acc, val_top5_acc = val_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_func=loss_func
        )
        
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        
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
        
        if val_top1_acc > best_top1_acc or epoch % 5 == 0:
            filename = 'best_model.pth' if val_top1_acc > best_top1_acc else f'checkpoint_epoch_{epoch}.pth'
            best_top1_acc = max(val_top1_acc,best_top1_acc)

            save_chk_point({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': val_top1_acc,
                'config': config,
            }, save_dir, filename)

    save_chk_point({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_top1_acc,
        'config': config, }, 
        save_dir, 
        'final_model.pth'
    )

    logger.print_summary()
    
    

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
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # 设置CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    main()
    