# Jittor-RepVGG-CIFAR100

## 简介

这是一个使用Jittor框架实现的RepVGG ，并在CIFAR100上进行测试，带有与Jittor实现对齐的Torch版本

以jittor为后缀命名的源代码文件为jittor 实现 ， 以torch为后缀命名的为torch实现

模型结构搭建遵循原作者论文[Ding, Xiaohan, et al. "Repvgg: Making vgg-style convnets great again." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.](https://arxiv.org/abs/2101.03697)

为了适应CIFAR-100数据集做出以下改动

- 将5个stage的卷积通道缩放系数[64,64,128,256,512] 更换为 [32,32,64,128,256]
- 将前两个stage的第一个RepVGG_block，卷积stride由2改为1

具体结构变为如下表所示

| Stage | Output size | RepVGG-A                | RepVGG-B                |
|-------|-------------|-------------------------|-------------------------|
| 1     | 32× 32| 1 × min(32, 32a)              | 1 × min(32, 32a)            |
| 2     | 32× 32| 2 × 32a                       | 4 × 32a                     |
| 3     | 16× 16| 4 × 64a                       | 6 × 64a                     |
| 4     | 8× 8  | 14 × 128a                     | 16 × 128a                   |
| 5     | 4 × 4 | 1 × 256b                      | 1 × 256b                    |

缩放系数a、b保持原作者设置，如下表所示

| Name       | Layers of each stage | a    | b    |
|------------|-----------------------|------|------|
| RepVGG-A0  | 1, 2, 4, 14, 1        | 0.75 | 2.5  |
| RepVGG-A1  | 1, 2, 4, 14, 1        | 1    | 2.5  |
| RepVGG-A2  | 1, 2, 4, 14, 1        | 1.5  | 2.75 |
| RepVGG-B0  | 1, 4, 6, 16, 1        | 1    | 2.5  |
| RepVGG-B1  | 1, 4, 6, 16, 1        | 2    | 4    |
| RepVGG-B1g4| 1, 4, 6, 16, 1        | 2    | 4    |


## 环境配置

**注意Jittor目前(截止到1.3.9.14版本) 对 conda支持并不完善 [点击查看issue](https://github.com/Jittor/jittor/issues/298) , 因而脚本使用venv进行环境管理** 

**如遇到Jittor自带的Cutlass库下载链接失效 [点击查看issue](https://discuss.jittor.org/t/topic/936) 使用该链接替换 `https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip`**

环境配置脚本 `repo/env_create.sh`: 

```bash
#!/bin/bash

sudo apt update

sudo apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv repvgg 

source repvgg/bin/activate

pip install -r requirements.txt

python3.11 -m jittor.test.test_example

python3.11 -m jittor.test.test_cudnn_op
```

##  数据准备

使用 Jittor 官方提供的 `jittor.dataset.CIFAR100`

数据增强方法包括 :
- RandAugment , `N=2,M=9`
- RandomCrop , `size=32,padding=4`
- RandomHorizontalFlip , `p=0.5`
- ImageNormalize , `mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]`
        
数据准备脚本 `repo/train/data_loader_jittor.py`

```python
import jittor as jt
import jittor.nn as nn
from jittor.dataset import ImageFolder
import jittor.transform as transform
from jittor.dataset.cifar import CIFAR100
from torchvision import transforms
from jittor import transform

def get_cifar100_dataloaders(config):
    batch_size = config['batch_size']
    img_size = config['image_size']
    
    train_transform = transform.Compose([
        transforms.RandAugment(),
        transforms.RandomCrop(32,padding=4),
        transform.Resize(img_size),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    val_transform = transform.Compose([
        transform.Resize(img_size),
        transform.ToTensor(),
        transform.ImageNormalize(
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    num_workers = config['num_workers']
    
    train_dataset = CIFAR100(
        root=config['data_path'],
        train=True,
        transform=train_transform,
        download= True
    )
    
    val_dataset = CIFAR100(
        root=config['data_path'],
        train=False,
        transform=val_transform,
        download=True
    )

    train_loader = train_dataset.set_attrs(
        batch_size=batch_size, shuffle=True, num_workers=num_workers,buffer_size=1024**3)
    val_loader = val_dataset.set_attrs(
        batch_size=batch_size, shuffle=False, num_workers=num_workers,buffer_size=1024**3)

    return train_loader, val_loader 
```

## 训练

遵循原作者使用 CosineAnnealing , Nesterov动量法

超参数
- batch_size : `512`
- epochs : `250` 
- lr : `0.4`
- weight_decay : `5e-4`
- Nesterov momentum : `0.9`
- CosineAnnealing eta_min: `1e-6`

训练脚本 `/repo/main_train_jittor.py`

```python
import jittor as jt
import jittor.nn as nn
import os
import time

import yaml
from model.RepVGG_model_jittor import RepVGG_Model 
from train.train_jittor import train_one_epoch , val_one_epoch
from train.data_loader_jittor import get_cifar100_dataloaders 
from train.optimizer_jittor import get_optimizer, get_scheduler
from utils.train_logger import Logger
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

    train_loader, val_loader = get_cifar100_dataloaders(config)

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
```

## 测试 

脚本可查看 RepVGG 模型三路分支合并前后的 

- `模型参数数量(M)` 
-  `推理速度(KFPS)`
- `转换前后CIFAR100测试集准确率`

测试脚本 `/repo/infer_test_jittor.py`


```python
import jittor as jt
import os
import yaml
import time
import csv

from model.RepVGG_model_jittor import RepVGG_Model
from train.data_loader_jittor import get_cifar100_dataloaders

jt.flags.use_cuda = 1  

def create_model(config):
    model_name = config['model_name']
    if 'resnet' in model_name:
        from model import ResNet_model_jittor
        return ResNet_model_jittor.resnet18()
    else:
        return RepVGG_Model(
            channel_scale_A=config['scale_a'],
            channel_scale_B=config['scale_b'],
            group_conv=config['group_conv'],
            classify_classes=config['num_classes'],
            model_type=config['model_type']
        )

def speed_test(model):
    model.eval()
    batch_size = 512
    img_size = 32

    data = jt.randn((batch_size, 3, img_size, img_size))
    warm_up = 50
    test_rounds = 1000 

    jt.sync_all(True)
    for _ in range(warm_up):
        output = model(data)
        output.sync()
    jt.sync_all(True)

    st = time.time()
    for _ in range(test_rounds):
        output = model(data)
        output.sync() 
    jt.sync_all(True)
    ed = time.time()

    res = test_rounds * batch_size / (ed - st)
    res = round(res / 1000 , 2)
    return res

def acc_test(config, chk_point_path):
    import train.train_jittor as tool

    model = create_model(config)
    checkpoint = jt.load(os.path.join(chk_point_path, 'best_model.jt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, val_loader = get_cifar100_dataloaders(config)

    _, before_top1_acc, _ = tool.val_one_epoch(model, val_loader, jt.nn.CrossEntropyLoss())

    if 'resnet' not in config['model_name']:
        model.convert_to_infer()

    _, after_top1_acc, _ = tool.val_one_epoch(model, val_loader, jt.nn.CrossEntropyLoss())

    return before_top1_acc, after_top1_acc

def infer_test(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model_name']

    model = create_model(config)

    before_params = sum(p.numel() for p in model.parameters())
    before_params = round(before_params / 1e6,2)
    before_speed = speed_test(model)

    chk_point_path = config['chk_point_dir']

    if 'resnet' not in config['model_name']:
        model.convert_to_infer()

    after_params = sum(p.numel() for p in model.parameters())
    after_params = round(after_params / 1e6,2)
    after_speed = speed_test(model)

    b_acc, a_acc = acc_test(config, chk_point_path)
    b_acc , a_acc = round(b_acc , 2) , round(a_acc , 2)

    path = os.path.join('.', model_name + '_infer_log.csv')
    with open(path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Model Name', 'Before Params(M)', 'Before Speed(KFPS)', 'Before acc',
                         'After Params(M)', 'After Speed(KFPS)', 'After acc'])
        writer.writerow([model_name, before_params, before_speed, b_acc,
                         after_params, after_speed, a_acc])

    print(f"Inference results saved to {path}")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    config_path = args.config
    assert os.path.exists(config_path), f"Config file {config_path} does not exist!"

    infer_test(config_path=config_path)

if __name__ == "__main__":
    main()
```

## 结果Log

测试硬件环境 ： autodl云服务器

CPU : 18 vCPU AMD EPYC 9754 128-Core Processor

GPU : RTX 4090D

RAM : 60G

--- 

### Jittor框架实现:


| 模型          |   转换前参数量(M) |   转换前速度(KFPS) |   转换前准确率(%) |   转换后参数量(M) |   转换后速度(KFPS) |   转换后准确率(%) |
|:------------|------------:|--------------:|------------:|------------:|--------------:|------------:|
| RepVGG_A0   |        2.04 |         77.58 |       75.06 |        1.82 |        157.47 |       75.04 |
| RepVGG_A1   |        3.29 |         53.74 |       76.35 |        2.94 |        107.09 |       76.35 |
| RepVGG_A2   |        6.8  |         35.3  |       77.78 |        6.1  |         55.37 |       77.78 |
| RepVGG_B0   |        3.73 |         44.54 |       77.04 |        3.33 |         79.83 |       77.03 |
| RepVGG_B1   |       14    |         21.18 |       79.44 |       12.55 |         29.14 |       79.43 |
| RepVGG_B1g4 |        9.63 |         23.5  |       79.01 |        8.63 |         34.71 |       79.01 |
| resnet18    |        2.83 |         94.89 |       75.84 |        2.83 |         94.8  |       75.84 |


### Torch框架实现：
| 模型          |   转换前参数量(M) |   转换前速度(KFPS) |   转换前准确率(%) |   转换后参数量(M) |   转换后速度(KFPS) |   转换后准确率(%) |
|:------------|------------:|--------------:|------------:|------------:|--------------:|------------:|
| RepVGG_A0   |        2.03 |         60.02 |       75.65 |        1.82 |        127.86 |       75.7  |
| RepVGG_A1   |        3.27 |         44.2  |       76.27 |        2.94 |         94.59 |       76.25 |
| RepVGG_A2   |        6.78 |         25.68 |       77.38 |        6.1  |         48.42 |       77.36 |
| RepVGG_B0   |        3.71 |         31.19 |       77.04 |        3.33 |         69.36 |       76.98 |
| RepVGG_B1   |       13.96 |         12.28 |       79.75 |       12.55 |         22.99 |       79.78 |
| RepVGG_B1g4 |        9.6  |         13.07 |       78.84 |        8.63 |         26.19 |       78.83 |
| resnet18    |        2.82 |         81.57 |       75.70 |        2.82 |         81.65  |      75.70 |


### RepVGG-A0 训练过程对比

#### 损失函数变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG A0 Jittor Loss](train_logs/jittor_A0_cifar/RepVGG_Jittor_A0_cifar_loss.png) | ![RepVGG A0 Torch Loss](train_logs/torch_A0_cifar/RepVGG_Torch_A0_cifar_loss.png) |

#### 准确率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG A0 Jittor Accuracy](train_logs/jittor_A0_cifar/RepVGG_Jittor_A0_cifar_accuracy.png) | ![RepVGG A0 Torch Accuracy](train_logs/torch_A0_cifar/RepVGG_Torch_A0_cifar_accuracy.png) |

### RepVGG-A1 训练过程对比

#### 损失函数变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG A1 Jittor Loss](train_logs/jittor_A1_cifar/RepVGG_Jittor_A1_cifar_loss.png) | ![RepVGG A1 Torch Loss](train_logs/torch_A1_cifar/RepVGG_Torch_A1_cifar_loss.png) |

#### 准确率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG A1 Jittor Accuracy](train_logs/jittor_A1_cifar/RepVGG_Jittor_A1_cifar_accuracy.png) | ![RepVGG A1 Torch Accuracy](train_logs/torch_A1_cifar/RepVGG_Torch_A1_cifar_accuracy.png) |

### RepVGG-A2 训练过程对比

#### 损失函数变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG A2 Jittor Loss](train_logs/jittor_A2_cifar/RepVGG_Jittor_A2_cifar_loss.png) | ![RepVGG A2 Torch Loss](train_logs/torch_A2_cifar/RepVGG_Torch_A2_cifar_loss.png) |

#### 准确率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG A2 Jittor Accuracy](train_logs/jittor_A2_cifar/RepVGG_Jittor_A2_cifar_accuracy.png) | ![RepVGG A2 Torch Accuracy](train_logs/torch_A2_cifar/RepVGG_Torch_A2_cifar_accuracy.png) |

### RepVGG-B0 训练过程对比

#### 损失函数变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG B0 Jittor Loss](train_logs/jittor_B0_cifar/RepVGG_Jittor_B0_cifar_loss.png) | ![RepVGG B0 Torch Loss](train_logs/torch_B0_cifar/RepVGG_Torch_B0_cifar_loss.png) |

#### 准确率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG B0 Jittor Accuracy](train_logs/jittor_B0_cifar/RepVGG_Jittor_B0_cifar_accuracy.png) | ![RepVGG B0 Torch Accuracy](train_logs/torch_B0_cifar/RepVGG_Torch_B0_cifar_accuracy.png) |

### RepVGG-B1 训练过程对比

#### 损失函数变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG B1 Jittor Loss](train_logs/jittor_B1_cifar/RepVGG_Jittor_B1_cifar_loss.png) | ![RepVGG B1 Torch Loss](train_logs/torch_B1_cifar/RepVGG_Torch_B1_cifar_loss.png) |

#### 准确率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG B1 Jittor Accuracy](train_logs/jittor_B1_cifar/RepVGG_Jittor_B1_cifar_accuracy.png) | ![RepVGG B1 Torch Accuracy](train_logs/torch_B1_cifar/RepVGG_Torch_B1_cifar_accuracy.png) |

### RepVGG-B1g4 训练过程对比

#### 损失函数变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG B1g4 Jittor Loss](train_logs/jittor_B1g4_cifar/RepVGG_Jittor_B1g4_cifar_loss.png) | ![RepVGG B1g4 Torch Loss](train_logs/torch_B1g4_cifar/RepVGG_Torch_B1g4_cifar_loss.png) |

#### 准确率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![RepVGG B1g4 Jittor Accuracy](train_logs/jittor_B1g4_cifar/RepVGG_Jittor_B1g4_cifar_accuracy.png) | ![RepVGG B1g4 Torch Accuracy](train_logs/torch_B1g4_cifar/RepVGG_Torch_B1g4_cifar_accuracy.png) |


### ResNet18 训练过程对比


#### 损失函数变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![ResNet18 Jittor Loss](train_logs/jittor_resnet_cifar/resnet_jittor_cifar_loss.png) | ![ResNet18 Torch Loss](train_logs/torch_resnet_cifar/resnet_torch_cifar_loss.png) |

#### 准确率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![ResNet18 Jittor Accuracy](train_logs/jittor_resnet_cifar/resnet_jittor_cifar_accuracy.png) | ![ResNet18 Torch Accuracy](train_logs/torch_resnet_cifar/resnet_torch_cifar_accuracy.png) |

#### 学习率变化
| Jittor框架 | PyTorch框架 |
|-----------|-------------|
| ![ResNet18 Jittor LR](train_logs/jittor_resnet_cifar/resnet_jittor_cifar_learning_rate.png) | ![ResNet18 Torch LR](train_logs/torch_resnet_cifar/resnet_torch_cifar_learning_rate.png) |