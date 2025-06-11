import torch.nn as nn

class FeedForwardDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)
