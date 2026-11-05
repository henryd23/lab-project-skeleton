import torch
from torch import nn
import torch.nn.functional as F

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Add more layers...
        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.conv2(x).relu() # B x 128 x 224 x 224
        x = F.adaptive_avg_pool2d(x, (2, 1)) # B x 128 x 2 x 1

        # Appiattimento del tensore da 4D (spaziale) a 2D (batch_size, features)
        x = torch.flatten(x, 1) # B x 256

        # Passaggio nel livello lineare per ottenere i logits delle classi
        x = self.fc1(x)

        return x