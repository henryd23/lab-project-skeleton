import torch
import wandb
from torch import nn
from models.model import CustomNet
from utils.engine import train, validate
from dataset.dataset import make_loaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Training] Preparo i dataset e i DataLoader...", flush=True)
    train_loader, val_loader = make_loaders()
    print(f"[Training] Device selezionato: {device}", flush=True)
    print("[Training] Creo modello, loss function e optimizer...", flush=True)

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    learning_rate = 0.001
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9
    )

    best_acc = 0.0

    print("[Training] Inizializzo il run W&B...", flush=True)
    with wandb.init(
        project="tiny-imagenet-cnn",
        config={
            "epochs": num_epochs,
            "batch_size": 32,
            "learning_rate": learning_rate,
            "optimizer": "SGD",
            "momentum": 0.9,
            "model": "CustomNet",
            "device": str(device),
        },
        save_code=False,
    ) as run:
        print(f"[W&B] Run avviato: {run.url}", flush=True)
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("validation/*", step_metric="epoch")
        run.define_metric("best/*", step_metric="epoch")

        for epoch in range(1, num_epochs + 1):
            print(f"\n===== Epoca {epoch}/{num_epochs}: training =====", flush=True)
            train_loss, train_accuracy = train(
                epoch, model, train_loader, criterion, optimizer, device
            )

            print(f"===== Epoca {epoch}/{num_epochs}: validazione =====", flush=True)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "checkpoints/best_model.pth")
                print(
                    f"[Checkpoint] Salvato nuovo modello migliore "
                    f"(accuratezza {best_acc:.2f}%).",
                    flush=True,
                )

            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "validation/loss": val_loss,
                    "validation/accuracy": val_acc,
                    "best/validation_accuracy": best_acc,
                },
                step=epoch,
            )
            print(f"[W&B] Metriche dell'epoca {epoch} inviate.", flush=True)

        run.summary["best_validation_accuracy"] = best_acc
        print("[Training] Tutte le epoche sono terminate.", flush=True)

if __name__ == "__main__":
    main()
