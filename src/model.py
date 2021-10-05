import torch.nn as nn
import torch


class NeuralNet(nn.Module):
    def __init__(self, cont_input_size, emb_dims, output_size=1):
        super(NeuralNet, self).__init__()
        self.cont_input_size = cont_input_size
        self.emb_dims = emb_dims
        self.output_size = output_size

        self.bn1 = nn.BatchNorm1d(self.cont_input_size)
        self.emb_drop = nn.Dropout(0.6)

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(category, size) for category, size in self.emb_dims]
        )
        n_emb = sum(emb.embedding_dim for emb in self.embedding_layers)
        self.embedding_layers.apply(self.init_layers)

        # Combined layers
        self.combined_layer = nn.Sequential(
            nn.Linear(in_features=n_emb + cont_input_size, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.6),
            # nn.Linear(in_features=128, out_features=64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.combined_layer.apply(self.init_layers)

        # Output layer
        self.output_layer = nn.Linear(in_features=32, out_features=self.output_size)
        nn.init.kaiming_normal_(self.output_layer.weight)

    def init_layers(self, m):
        if type(m) == nn.Linear or type(m) == nn.Embedding:
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x_cont, x_emb):

        # Embedding layers
        x_emb = [
            self.emb_drop(f(x_emb[:, i])) for i, f in enumerate(self.embedding_layers)
        ]
        x_emb = torch.cat(x_emb, 1)
        # x_emb = self.emb_drop(x_emb)

        # Continuous layer
        x_cont = self.bn1(x_cont)

        # Combine embedding output and continuous output
        x = torch.cat((x_cont, x_emb), 1)

        # Combine layers
        x = self.combined_layer(x)

        # Output layer
        x = self.output_layer(x)
        return x
