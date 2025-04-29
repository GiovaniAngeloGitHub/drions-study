import segmentation_models_pytorch as smp
import torch

def get_model():
    """
    Inicializa o modelo UnetPlusPlus com EfficientNet-b0.
    """
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",    # Backbone
        encoder_weights="imagenet",        # Pesos pré-treinados
        in_channels=3,                     # 3 canais (RGB)
        classes=1,                         # 1 classe (disco óptico)
        activation=None                    # Sem ativação, aplicamos sigmoid depois
    )
    return model

def load_trained_model(model_path, device):
    """
    Carrega o modelo treinado a partir do arquivo salvo.
    """
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
