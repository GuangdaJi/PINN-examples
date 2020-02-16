import torch.nn as nn
from collections import OrderedDict


class Net(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers=5, width=100):
        super(Net, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.width = width

        #  input layer to first hidden layer
        self.features = nn.Sequential(
            OrderedDict([
                ('fc0', nn.Linear(in_features=input_dim, out_features=width, bias=True)),
                ('softplus0', nn.Softplus()),
            ])
        )

        # between hidden layers
        for i in range(hidden_layers - 1):
            self.features.add_module(
                'fc%d' % (i+1), nn.Linear(in_features=width, out_features=width, bias=True)
            )
            self.features.add_module('softplus%d' % (i+1), nn.Softplus())

        # output layers
        self.features.add_module(
            'fc%d' % hidden_layers,
            nn.Linear(in_features=width, out_features=output_dim, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # the weight have already been kaiming-uniform initialized according to pytorch's source code
                # nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.features(x)
