import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class DiscDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Máscara binária

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Garante que máscara seja binária
        mask = (mask > 0).float()

        return image, mask

def get_data_loaders(image_dir, mask_dir, batch_size=8, img_size=(256, 256)):
    # Lista de todos os arquivos de imagem
    all_images = sorted(os.listdir(image_dir))

    # Definindo imagens fixas para teste e validação
    test_images = all_images[:10]
    val_images = all_images[10:15]
    train_images = all_images[15:]

    def build_paths(image_list):
        img_paths = [os.path.join(image_dir, img) for img in image_list]
        mask_paths = [os.path.join(mask_dir, img.replace('.jpg', '.png')) for img in image_list]
        return img_paths, mask_paths

    train_img_paths, train_mask_paths = build_paths(train_images)
    val_img_paths, val_mask_paths = build_paths(val_images)
    test_img_paths, test_mask_paths = build_paths(test_images)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    train_dataset = DiscDataset(train_img_paths, train_mask_paths, transform=img_transform, mask_transform=mask_transform)
    val_dataset = DiscDataset(val_img_paths, val_mask_paths, transform=img_transform, mask_transform=mask_transform)
    test_dataset = DiscDataset(test_img_paths, test_mask_paths, transform=img_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
