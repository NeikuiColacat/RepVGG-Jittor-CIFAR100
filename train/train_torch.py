import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_imagenet_dataloaders(config):
    
    data_path = config['data_path']
    batch_size = config['batch_size']
    img_size = config['image_size']
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(f"{data_path}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_path}/val", transform=val_transform)
    
    num_workers = config['num_workers']
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    return train_loader, val_loader



def train_one_epoch(model: nn.Module, train_lodaer: DataLoader, optimizer: torch.optim.SGD, loss_func : nn.CrossEntropyLoss , epoch_idx , scaler):
    model.train()
    cur_loss = 0.
    ok_num = 0
    tt = 0
    process_bar = tqdm(train_lodaer,desc=f'epoch {epoch_idx} trainning')


    for batch_idx , (data, target) in enumerate(process_bar):

        data , target = data.to(device,non_blocking=True) , target.to(device,non_blocking=True) ,
        data = data.to(memory_format=torch.channels_last)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast('cuda'):
            output = model(data)
            loss = loss_func(output , target) 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        cur_loss += loss.item()
        pred_tar = output.argmax(dim = 1)
        ok_num += pred_tar.eq(target).sum().item()
        tt += target.shape[0]

        if batch_idx % 100 == 0 :
            acc = ok_num / tt * 100
            process_bar.set_postfix({
                'loss' : f'{loss.item():.4f}',
                'acc' : f'{acc:.2f}',
            })
    
    tt_loss = cur_loss / len(train_lodaer)
    tt_acc = ok_num / tt * 100

    return tt_loss , tt_acc


@torch.no_grad()
def val_one_epoch(model:nn.Module,val_loader,loss_func:nn.CrossEntropyLoss):
    model.eval()
    cur_loss = 0.
    top1_ok_num = 0
    top5_ok_num = 0
    tt = 0

    process_bar = tqdm(val_loader,desc=f'validating')

    for data , target in process_bar:
        data , target = data.to(device,non_blocking=True) , target.to(device,non_blocking=True) ,
        data = data.to(memory_format=torch.channels_last) 


        with torch.cuda.amp.autocast('cuda'):
            output = model(data)
            loss = loss_func(output, target)
        
        cur_loss += loss.item()
        
        top1_num, top5_num = cal_topk_num(output, target , (1,5))
        top1_ok_num += top1_num.item() 
        top5_ok_num += top5_num.item()
        tt += target.shape[0]
        
        current_top1 = top1_ok_num / tt * 100.0
        current_top5 = top5_ok_num / tt * 100.0
        process_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Top1': f'{current_top1:.2f}%',
            'Top5': f'{current_top5:.2f}%'
        })
    
    cur_loss /= len(val_loader)
    top1_acc = top1_ok_num / tt * 100.0
    top5_acc = top5_ok_num / tt * 100.0

    return cur_loss , top1_acc , top5_acc



@torch.no_grad()
def cal_topk_num(output : torch.Tensor , target:torch.Tensor , topk):

    mxk = max(topk)
    # pred b*k , tar b * 1
    _ , pred = output.topk(mxk,dim=1)
    target = target.reshape(-1,1).expand_as(pred)

    #ok b * 1
    ok = pred.eq(target)
    res = []
    for k in topk:
        ok_k = ok[:, :k].reshape(-1).float().sum()
        res.append(ok_k)
    return res








        



