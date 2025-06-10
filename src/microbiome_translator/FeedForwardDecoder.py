class FeedForwardDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)