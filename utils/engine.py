import torch
from models.model import CustomNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    model.to(device=device)

    total_batches = len(train_loader)
    progress_interval = max(1, total_batches // 10)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        completed_batches = batch_idx + 1
        if completed_batches % progress_interval == 0 or completed_batches == total_batches:
            print(
                f"[Training {epoch}] Batch {completed_batches}/{total_batches} "
                f"({completed_batches / total_batches:.0%}) | "
                f"loss medio: {running_loss / completed_batches:.4f} | "
                f"accuratezza: {100. * correct / total:.2f}%",
                flush=True,
            )

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    model.to(device=device)

    total_batches = len(val_loader)
    progress_interval = max(1, total_batches // 10)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass per calcolare le uscite del modello
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            completed_batches = batch_idx + 1
            if completed_batches % progress_interval == 0 or completed_batches == total_batches:
                print(
                    f"[Validazione] Batch {completed_batches}/{total_batches} "
                    f"({completed_batches / total_batches:.0%})",
                    flush=True,
                )

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy
