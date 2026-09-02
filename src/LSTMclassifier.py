class LSTMClassifier(nn.Module):
    '''Single LSTM model'''

    def __init__(self, n_channels=N_CHANNELS, hidden=64, dense=64, n_classes=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, dense)
        self.fc2 = nn.Linear(dense, n_classes)

        init_lstm(self.lstm)
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc2(torch.relu(self.fc1(h_n[-1])))

    @torch.no_grad()
    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=1)
