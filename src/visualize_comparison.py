import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_predictions(model, test_loader, original_images_dir, output_dir='outputs/comparisons', img_size=(256, 256), device='cpu'):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    idx = 0
    image_files = sorted(os.listdir(original_images_dir))

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()

            batch_size = images.size(0)

            for b in range(batch_size):
                if idx >= len(image_files):
                    break

                # Carregar imagem original
                original_img_path = os.path.join(original_images_dir, image_files[idx])
                original_img = cv2.imread(original_img_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                original_img = cv2.resize(original_img, img_size)

                # Processar as máscaras
                pred_mask = (preds[b].squeeze() * 255).astype(np.uint8)
                gt_mask = (masks[b].squeeze() * 255).astype(np.uint8)

                # Criar visualização
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(original_img)
                axs[0].set_title('Imagem Original')
                axs[0].axis('off')

                axs[1].imshow(original_img)
                axs[1].imshow(pred_mask, cmap='Reds', alpha=0.5)
                axs[1].set_title('Máscara Prevista (Modelo)')
                axs[1].axis('off')

                axs[2].imshow(original_img)
                axs[2].imshow(gt_mask, cmap='Greens', alpha=0.5)
                axs[2].set_title('Máscara Ground Truth (Especialista)')
                axs[2].axis('off')

                output_path = os.path.join(output_dir, f'comparison_{idx:03d}.png')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()

                idx += 1

    print(f"✅ Imagens comparativas salvas em: {output_dir}")


def save_predicted_masks(model, test_loader, output_dir='outputs/predicted_masks', img_size=(256, 256), device='cpu'):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    idx = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            preds = preds.cpu().numpy()

            batch_size = images.size(0)

            for b in range(batch_size):
                pred_mask = (preds[b].squeeze() * 255).astype(np.uint8)

                output_path = os.path.join(output_dir, f'predicted_mask_{idx:03d}.png')
                cv2.imwrite(output_path, pred_mask)

                idx += 1

    print(f"✅ Máscaras previstas salvas em: {output_dir}")



def save_raw_outputs(model, test_loader, output_dir='outputs/raw_outputs', img_size=(256, 256), device='cpu'):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    idx = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            outputs = model(images)  # <- sem sigmoid
            outputs = outputs.cpu().numpy()

            batch_size = images.size(0)

            for b in range(batch_size):
                raw_output = outputs[b].squeeze()

                # Normalizar o raw_output para 0-255 apenas para salvar visualmente
                raw_min = np.min(raw_output)
                raw_max = np.max(raw_output)

                if raw_max - raw_min > 1e-6:
                    norm_output = (raw_output - raw_min) / (raw_max - raw_min)
                    norm_output = (norm_output * 255).astype(np.uint8)
                else:
                    # Se quase tudo igual (tipo saída constante), força tudo zero
                    norm_output = np.zeros_like(raw_output, dtype=np.uint8)

                output_path = os.path.join(output_dir, f'raw_output_{idx:03d}.png')
                cv2.imwrite(output_path, norm_output)

                idx += 1

    print(f"✅ Saída bruta do modelo salva em: {output_dir}")
