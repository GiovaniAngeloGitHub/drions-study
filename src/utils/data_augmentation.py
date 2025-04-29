import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def expand_channels(image, **kwargs):
    """
    Converte imagens de 1 canal (grayscale) para 3 canais (RGB falso),
    repetindo o mesmo canal 3 vezes.
    """
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return image

def get_train_transforms():
    """
    Transforms para treino: aumenta a base de dados com rotações, flips,
    brilho/contraste, e pequenas deformações (Affine).
    """
    return A.Compose([
        A.Lambda(image=expand_channels),  # Expande canais antes de tudo
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Affine(translate_percent=(0.1, 0.1), scale=(0.85, 1.15), rotate=0, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def get_val_transforms():
    """
    Transforms para validação/teste: apenas normalização, sem augmentação.
    """
    return A.Compose([
        A.Lambda(image=expand_channels),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
