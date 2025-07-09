import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

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

def get_cifar100_dataloaders(config):
    batch_size = config['batch_size']
    img_size = config['image_size']
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size,padding=4),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    num_workers = config['num_workers']
    
    train_dataset = datasets.CIFAR100(
        root=config['data_path'],
        train=True,
        transform=train_transform,
        download=True
    )
    
    val_dataset = datasets.CIFAR100(
        root=config['data_path'],
        train=False,
        transform=val_transform,
        download=True
    )
    
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