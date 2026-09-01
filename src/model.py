class CNNLSTMClassifier(nn.Module):
    """1D CNN front end + LSTM.

    Shapes, for (batch, 128, 9) in:

        permute      -> (B,  9, 128)   Conv1d wants channels-first
        conv1 k=5    -> (B, 64, 124)
        conv2 k=5    -> (B, 64, 120)
        maxpool 2    -> (B, 64,  60)
        permute      -> (B, 60,  64)   LSTM wants (batch, time, feature)
        lstm         -> (B, 64)        final hidden state only
        head         -> (B,  6)        logits

    Convs carry no bias — BatchNorm's shift parameter makes it redundant.
    """

    def __init__(self, n_channels=N_CHANNELS, filters=64, hidden=64, dense=64,
                 n_classes=N_CLASSES, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, filters, kernel_size=5, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, kernel_size=5, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(filters, hidden, batch_first=True)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, dense),
            nn.ReLU(inplace=True),
            nn.Linear(dense, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        init_lstm(self.lstm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.transpose(1, 2))
        _, (h_n, _) = self.lstm(x.transpose(1, 2))
        return self.head(h_n[-1])

    @torch.no_grad()
    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=1)
