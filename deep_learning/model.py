import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, depth=3, width=256, input_channels=65, output_ch=6, use_dropout=False, dropout_p=0.5):
        super(MLP, self).__init__()

        self.linear_layers = nn.ModuleList([nn.Linear(input_channels, width)] + [nn.Linear(width, width) for i in range(depth - 1)])
        self.output_layer = nn.Linear(width, output_ch)
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p

    def forward(self, x, testing=False):
        if len(x.shape) == 3:
            x = x.view([x.shape[0], -1])
        for layer in self.linear_layers:
            x = F.relu(layer(x))
            if self.use_dropout and not testing:
                x = F.dropout(x, self.dropout_p)

        return self.output_layer(x)