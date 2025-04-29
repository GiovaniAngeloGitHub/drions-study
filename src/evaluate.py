import os
import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def predict_and_save(model, test_loader, device, output_dir='outputs/predictions', img_size=(256, 256)):
    """
    Gera predições usando o modelo treinado e salva as máscaras previstas em um diretório.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (images, _) in enumerate(test_loader):
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            # Binariza a saída
            pred_mask = (probs > 0.5).astype(np.uint8) * 255

            # Salva a máscara como imagem PNG
            mask_img = Image.fromarray(pred_mask)
            mask_img = mask_img.resize(img_size)
            save_path = os.path.join(output_dir, f"pred_mask_{idx:03d}.png")
            mask_img.save(save_path)

            print(f"✅ Máscara salva: {save_path}")

    print(f"🎯 Todas as máscaras previstas foram salvas em '{output_dir}'")


def predict_and_visualize(model, test_loader, image_dir, mask_dir,
                          output_dir='outputs/comparisons', img_size=(256, 256), device='cpu'):
    """
    Gera figuras comparativas entre ground-truth, predição e diferença.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    transform_inv = transforms.Compose([
        transforms.Resize(img_size)
    ])

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)

            # Nome do arquivo da imagem original
            image_name = sorted(os.listdir(image_dir))[idx]
            original_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.replace('.jpg', '.png'))

            # Carregar imagem original (RGB)
            original_img = Image.open(original_path).convert("RGB")
            original_img = transform_inv(original_img)
            original_np = np.array(original_img)

            # Ground-truth mask
            gt_mask = Image.open(mask_path).convert("L")
            gt_mask = transform_inv(gt_mask)
            gt_mask = np.array(gt_mask)
            gt_mask_bin = (gt_mask > 0).astype(np.uint8) * 255

            # Predição do modelo
            output = model(images)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

            # Máscara de diferença
            diff_mask = cv2.absdiff(gt_mask_bin, pred_mask_bin)

            # --- VISUALIZAÇÃO ---
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].imshow(original_np)
            axs[0].imshow(gt_mask_bin, cmap='Greens', alpha=0.4)
            axs[0].set_title("Ground Truth")
            axs[0].axis('off')

            axs[1].imshow(original_np)
            axs[1].imshow(pred_mask_bin, cmap='Reds', alpha=0.4)
            axs[1].set_title("Predição do Modelo")
            axs[1].axis('off')

            axs[2].imshow(original_np)
            axs[2].imshow(diff_mask, cmap='Blues', alpha=0.6)
            axs[2].set_title("Diferença")
            axs[2].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(output_dir, f'comparison_{idx:03d}.png')
            plt.savefig(save_path)
            plt.close()

            print(f"✅ Figura comparativa salva: {save_path}")

    print(f"🎯 Comparações salvas em: {output_dir}")