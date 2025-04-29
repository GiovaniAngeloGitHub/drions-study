import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp

# Configura dispositivo e paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_img_dir = 'data/val/images'
output_dir = 'data/test/predictions'
os.makedirs(output_dir, exist_ok=True)

# Parâmetros (idem ao treinamento)
img_height, img_width = 256, 256
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Carrega o modelo (mesma arquitetura) e pesos do melhor estado salvo
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# Percorre imagens de teste, faz predição e salva máscara binária
for img_name in sorted(os.listdir(test_img_dir)):
    img_path = os.path.join(test_img_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Converte para máscara binária (0 ou 255)
    mask = (prob > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    save_path = os.path.join(output_dir, f"pred_{img_name}")
    mask_img.save(save_path)
    print(f"Salvo: {save_path}")

print("Predição concluída.")
