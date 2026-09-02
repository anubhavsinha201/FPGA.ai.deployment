def init_lstm(lstm: nn.LSTM) -> None:
    """Xavier input kernel, orthogonal recurrent kernel, forget-gate bias = 1.

    PyTorch defaults every LSTM weight to uniform(±1/√hidden). The orthogonal
    recurrent kernel keeps gradients from vanishing over 60+ timesteps, and the
    unit forget bias stops the cell from erasing its state early in training. This is set to default both by PyTorch and Keras

    Gate order is (input, forget, cell, output), so the forget slice is the
    second quarter. PyTorch's two bias vectors simply sum, so setting the 1.0
    on bias_ih alone is enough.
    """

  
    hidden = lstm.hidden_size
    for name, param in lstm.named_parameters():
        if name.startswith("weight_ih"):
            nn.init.xavier_uniform_(param)
        elif name.startswith("weight_hh"):
            nn.init.orthogonal_(param)
        elif name.startswith("bias"):
            nn.init.zeros_(param)
    with torch.no_grad():
        lstm.bias_ih_l0[hidden : 2 * hidden].fill_(1.0)

'''standard protocol for a classic LSTM. Will change them to improve after during High Granularity Quantization and Quantization Aware Training
to improve batch validation and model precision'''
