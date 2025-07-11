import torch
import os
import yaml
import time
import csv

from model.RepVGG_model_torch import RepVGG_Model
from train.data_loader_torch import get_cifar100_dataloaders 
from utils.train_logger import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create_model(config):

    model_name = config['model_name']
    if 'resnet' in model_name :
        from model import ResNet_model_torch
        return ResNet_model_torch.resnet18()

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

    data = torch.randn((batch_size,3,img_size,img_size)).to(device)
    warm_up = 50 
    test_rounds = 1000 

    torch.cuda.synchronize()
    for _ in range(warm_up):
        output = model(data)

    torch.cuda.synchronize()
    
    st = time.time()
    for _ in range(test_rounds):
        output = model(data)

    torch.cuda.synchronize()
    ed = time.time()

    res = test_rounds * batch_size / (ed - st)
    res = round(res / 1000 , 2)
    return res

def acc_test(config , chk_point_path):
    import train.train_torch as tool
    model : torch.nn.Module = create_model(config).to(device)
    chk_point = torch.load(os.path.join(chk_point_path,'best_model.pth')) 
    model.load_state_dict(chk_point['model_state_dict'])
    model.eval()

    _ , val_loader = get_cifar100_dataloaders(config)
    _ , before_top1_acc , _ = tool.val_one_epoch(model,val_loader,torch.nn.CrossEntropyLoss(),False)

    if 'resnet' not in config['model_name']:
        model.convert_to_infer()

    _ , after_top1_acc , _ = tool.val_one_epoch(model,val_loader,torch.nn.CrossEntropyLoss(),False)


    return before_top1_acc, after_top1_acc

def infer_test(config_path):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model_name']

    model = create_model(config)
    model = model.to(device)
    chk_point_path = config['chk_point_dir']

    before_params = sum(p.numel() for p in model.parameters())
    before_params = round(before_params / 1e6,2)
    before_speed = speed_test(model)

    if 'resnet' not in config['model_name']:
        model.convert_to_infer()

    after_params = sum(p.numel() for p in model.parameters())
    after_params = round(after_params / 1e6,2)
    after_speed = speed_test(model)

    b_acc , a_acc = acc_test(config,chk_point_path)
    b_acc , a_acc = round(b_acc , 2) , round(a_acc , 2)
    path = os.path.join('.',model_name + '_infer_log.csv')
    with open(path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Model Name', 'Before Params(M)',
                        'Before Speed(KFPS)', 'Before acc' , 'After Params(M)', 'After Speed(KFPS)' , 'After acc'])
        writer.writerow([model_name, before_params, before_speed, b_acc , after_params, after_speed,a_acc])

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config_path = args.config 
    assert (os.path.exists(config_path))

    infer_test(config_path=config_path)
    
if __name__ == "__main__":
    main()
    