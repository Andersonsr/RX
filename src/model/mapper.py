import torch.nn as nn


class SimpleMapper(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMapper, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.GELU())

    def forward(self, x):
        return self.model(x)
