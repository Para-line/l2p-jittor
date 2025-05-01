from jittor import nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features = None, bias = True, act_layer = nn.GELU):
        super().__init__()

        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x