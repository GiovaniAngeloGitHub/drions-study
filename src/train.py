import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def train_model(model, train_loader, val_loader, device, num_epochs=50, learning_rate=1e-4):
    """
    Treina o modelo usando BCEWithLogitsLoss + DiceLoss e scheduler ReduceLROnPlateau.
    Salva o melhor modelo com base na maior média de Dice na validação.
    """
    # Critérios de perda
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode='binary', smooth=1.0)

    # Otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler que reduz o learning rate se o Dice não melhorar
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_dice = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward + Loss
            outputs = model(images)
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_bce + loss_dice

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validação
        model.eval()
        val_dice_total = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss_bce = criterion_bce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                val_loss += (loss_bce + loss_dice).item()

                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                intersection = (preds * masks).sum(dim=(2, 3))
                total = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (2 * intersection + 1) / (total + 1)
                val_dice_total += dice.mean().item()

        val_loss_avg = val_loss / len(val_loader)
        val_dice_avg = val_dice_total / len(val_loader)

        print(f"Época [{epoch}/{num_epochs}]  "
              f"Train Loss: {train_loss/len(train_loader):.4f}  "
              f"Val Loss: {val_loss_avg:.4f}  Val Dice: {val_dice_avg:.4f}")

        scheduler.step(val_dice_avg)

        # Salva o melhor modelo
        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> Novo melhor modelo salvo (Dice={best_dice:.4f}).")

    print(f"✅ Treinamento finalizado. Melhor Dice na validação: {best_dice:.4f}")
