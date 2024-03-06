import os
import time

import torch

import wandb


def train_epoch(model, criterion, device, dataloader, optimizer):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for data, *target in dataloader:
        data = data.to(device)
        target = (
            [t.to(device) for t in target] if len(target) > 1 else target[0].to(device)
        )
        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches


def val_epoch(model, criterion, device, dataloader):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    with torch.inference_mode():
        for data, *target in dataloader:
            data = data.to(device)
            target = (
                [t.to(device) for t in target]
                if len(target) > 1
                else target[0].to(device)
            )
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / num_batches


def train(
    epochs,
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    save_path,
    device,
    logger,
    val_iterations=1,
):
    """Trains the model and saves the best weights.

    Args:
        epochs (int): Number of epochs to train.
        dataloaders (tuple): Tuple containing the training and validation dataloaders.
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function to use.
        optimizer (torch.nn.Module): Optimizer to use.
        scheduler (torch.nn.Module): Learning rate scheduler to use.
        save_path (str): Path to save the best model weights.
        device (torch.device): Device to use.
        logger (logging.Logger): Logger to use.
        val_iterations (int, optional): Number of validation epochs to average. Defaults to 1.
    """
    train_loader, val_loader = dataloaders
    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = train_epoch(model, criterion, device, train_loader, optimizer)
        val_loss = 0
        if val_loader is not None:
            # each validation epoch is unique due to data augmentation, so we can average multiple
            for _ in range(val_iterations):
                val_loss += val_epoch(model, criterion, device, val_loader)
            val_loss /= val_iterations
        if scheduler is not None:
            scheduler.step()
        if epoch >= epochs * 0.9 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best.pt"))
        logger.info(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            + f" | EPOCH {(epoch+1):0{len(str(epochs))}}/{epochs}"
            + f" | TRAIN LOSS: {train_loss:.5f}"
            + f" | VAL LOSS: {val_loss:.5f}"
        )
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
    wandb.log({"best_val_loss": best_val_loss})
