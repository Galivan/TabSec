import torch.nn as nn

class Net(nn.Module):
    def __init__(self, D_in, H, D_out, layers_size):
        """
        Neural Network
        :param D_in: input layer dimension
        :param H: hidden layer dimension
        :param D_out: output layer dimension
        :param layers_size: amount of hidden layers
        """

        super(Net, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(D_in, H, dtype=float)])
        self.linears.extend(nn.ModuleList([nn.Linear(H, H, dtype=float) for i in range(1,layers_size - 1)]))
        self.last_linear = nn.Linear(H, D_out, dtype=float)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for linear in self.linears:
            x = self.relu(linear(x))
        x = self.last_linear(x)
        y = self.softmax(x)
        return y