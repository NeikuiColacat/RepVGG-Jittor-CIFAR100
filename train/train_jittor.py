import jittor as jt
import jittor.nn as nn
from jittor.dataset import ImageFolder
import jittor.transform as transform
from tqdm import tqdm
import os

jt.flags.use_cuda = 1

def get_imagenet_dataloaders(config):
    data_path = config['data_path']
    batch_size = config['batch_size']
    img_size = config['image_size']
    
    train_transform = transform.Compose([
        transform.RandomResizedCrop(img_size),
        transform.RandomHorizontalFlip(0.5),
        transform.ToTensor(),
        transform.ImageNormalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transform.Compose([
        transform.Resize(256),
        transform.CenterCrop(img_size),
        transform.ToTensor(),
        transform.ImageNormalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = ImageFolder(
        os.path.join(data_path, "train"), 
        transform=train_transform
    )
    val_dataset = ImageFolder(
        os.path.join(data_path, "val"), 
        transform=val_transform
    )
    
    num_workers = config['num_workers']
    train_loader = train_dataset.set_attrs(
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = val_dataset.set_attrs(
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

def train_one_epoch(model: nn.Module, train_loader, optimizer, loss_func, epoch_idx):
    model.train()
    cur_loss = 0.
    ok_num = 0
    tt = 0
    
    process_bar = tqdm(train_loader, desc=f'epoch {epoch_idx} training')

    for batch_idx, (data, target) in enumerate(process_bar):
        
        output = model(data)
        loss = loss_func(output, target)

        optimizer.step(loss)

        cur_loss += loss.item()
        pred = jt.argmax(output, dim=1)[0]
        ok_num += (pred == target).sum().item()
        tt += target.shape[0]

        if batch_idx % 100 == 0:
            acc = ok_num / tt * 100
            process_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%',
            })
    
    train_loss = cur_loss / len(train_loader)
    train_acc = ok_num / tt * 100

    return  train_loss , train_acc 

@jt.no_grad()
def val_one_epoch(model: nn.Module, val_loader, loss_func):
    model.eval()
    cur_loss = 0.
    top1_ok_num = 0
    top5_ok_num = 0
    tt = 0

    process_bar = tqdm(val_loader, desc='validating')
    for data, target in process_bar:
        output = model(data)
        loss = loss_func(output, target)
        
        cur_loss += loss.item()
        
        top1_num, top5_num = cal_topk_num(output, target, (1, 5))
        top1_ok_num += top1_num.item()
        top5_ok_num += top5_num.item()
        tt += target.shape[0]
        
        cur_t1 = top1_ok_num / tt * 100.0
        cur_t5 = top5_ok_num / tt * 100.0
        process_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Top1': f'{cur_t1:.2f}%',
            'Top5': f'{cur_t5:.2f}%'
        })
    
    cur_loss /= len(val_loader)
    top1_acc = top1_ok_num / tt * 100.0
    top5_acc = top5_ok_num / tt * 100.0

    return cur_loss , top1_acc, top5_acc

@jt.no_grad()
def cal_topk_num(output: jt.Var, target: jt.Var, topk):
    maxk = max(topk)
    
    _, pred = jt.topk(output, maxk, dim=1)
    target = target.reshape(-1, 1).expand_as(pred)

    correct = (pred == target)
    res = []
    for k in topk:
        correct_k = correct[:, :k].reshape(-1).float().sum()
        res.append(correct_k)
    
    return res
