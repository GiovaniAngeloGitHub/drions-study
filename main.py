import torch
import os

from src.data_loader import get_data_loaders
from src.model import get_model, load_trained_model
from src.train import train_model
from src.evaluate import predict_and_save, predict_and_visualize

def main():
    # Configurações principais
    image_dir = 'data/images/'
    mask_dir = 'data/masks/'
    model_path = 'best_model.pth'
    batch_size = 8
    num_epochs = 50
    img_size = (256, 256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Usando dispositivo: {device}")

    # Carrega os dataloaders
    train_loader, val_loader, test_loader = get_data_loaders(
        image_dir, mask_dir, batch_size=batch_size, img_size=img_size
    )

    # Verifica se o modelo já foi treinado
    if os.path.exists(model_path):
        print("🔍 Modelo encontrado. Carregando...")
        model = load_trained_model(model_path, device)
    else:
        print("🛠️ Treinando novo modelo...")
        model = get_model()
        model.to(device)

        train_model(model, train_loader, val_loader, device, num_epochs=num_epochs)

        print("✅ Modelo treinado e salvo.")

    # Avaliação e geração de predições
    print("🖼️ Gerando predições...")
    predict_and_save(model, test_loader, device, output_dir='outputs/predictions', img_size=img_size)
    predict_and_visualize(model, test_loader, image_dir=image_dir, mask_dir=mask_dir, device=device)
if __name__ == '__main__':
    main()
