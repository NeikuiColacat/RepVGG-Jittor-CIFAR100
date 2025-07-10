import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
import numpy as np


jt.flags.use_cuda = 1

def train_one_epoch(model: nn.Module, train_loader, optimizer, loss_func, epoch_idx):
    model.train()
    cur_loss = 0.
    ok_num = 0
    tt = 0
    
    process_bar = tqdm(train_loader, desc=f'epoch {epoch_idx} training')
    for batch_idx, (data, target) in enumerate(process_bar):
        
        # input , tar_a , tar_b , lam = mixup_data(data,target)
        output = model(data)
        # loss = mixup_loss_func(loss_func,output,tar_a,tar_b,lam)
        loss = loss_func(output , target)

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
    
    train_loss = cur_loss / (batch_idx + 1) 
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
    batch_cnt = 0
    for data, target in process_bar:
        batch_cnt += 1
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
    
    cur_loss /= batch_cnt 
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
