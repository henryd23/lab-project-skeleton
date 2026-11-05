import torch
# nn provides the loss function used to measure the model's errors.
from torch import nn
from models.model import CustomNet
from utils.engine import validate
from dataset.dataset import make_loaders

def main():
    # Use the GPU if available; otherwise, run evaluation on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = make_loaders()

    # Create the network and move it to the device selected above.
    model = CustomNet().to(device)

    # Load the weights saved during training; map_location also allows loading
    # the checkpoint if it was created on a different device.
    state_dict = torch.load("checkpoints/best_model.pth", map_location=device)

    # Copy the checkpoint weights into the layers of the newly created network.
    model.load_state_dict(state_dict)

    # Use cross entropy to compute the loss, as during training.
    criterion = nn.CrossEntropyLoss()

    # Compute loss and accuracy on the test set; the model weights are not updated.
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    # Print the results
    print(f"Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%")


# Run main only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()