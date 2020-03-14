import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVAE(nn.Module):


    def __init__(self, encoder_dims, decoder_dims=[], dropout_rate=0.2):
        super(MultiVAE, self).__init__()
        if decoder_dims:
            assert encoder_dims[0] == decoder_dims[-1] and encoder_dims[-1] == decoder_dims[0]
        self.latent_dim = encoder_dims[-1]
        self.encoder_dims = encoder_dims[:]
        if decoder_dims:
            self.decoder_dims = decoder_dims[:]
        else:
            self.decoder_dims = self.encoder_dims[::-1]

        dims = encoder_dims[:-1] + [encoder_dims[-1] * 2]
        self.encoder_layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
        )

        self.decoder_layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(self.decoder_dims[:-1], self.decoder_dims[1:])]
        )

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.initialize_weights()


    def initialize_weights(self):

        for layer in self.encoder_layers:
            torch.nn.init.xavier_normal_(layer.weight.data)
            torch.nn.init.normal_(layer.bias.data)

        for layer in self.decoder_layers:
            torch.nn.init.xavier_normal_(layer.weight.data)
            torch.nn.init.normal_(layer.bias.data)


    def encode(self, x: torch.tensor):
        """
        :param x: (batch_size, input_dim)
        :return:
          mu: (batch_size, latent_dim)
          logvar: (batch_size, latent_dim)
        """
        h = F.normalize(x)
        h = self.dropout_layer(h)
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if i != len(self.encoder_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.latent_dim]
                logvar = h[:, self.latent_dim:]

        return mu, logvar


    def decode(self, z: torch.tensor):
        """
        :param a: (batch_size, latent_dim)
        :return: x_tilde: (batch_size, input_dim)
        """
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if i != len(self.decoder_layers) - 1:
                h = torch.tanh(h)
        return h


    def reparametrize(self, mu: torch.tensor, logvar: torch.tensor):
        """
        :param mu: (batch_size, latent_dim)
        :param logvar: (batch_size, latent_dim)
        :return:
        """
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
            return z
        else:
            return mu


    def forward(self, x: torch.tensor):
        """
        :param x: (batch_size, input_dim)
        :return: x_tilde, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_tilde = self.decode(z)
        return x_tilde, mu, logvar


def loss_function(x: torch.tensor, x_tilde: torch.tensor, mu: torch.tensor, logvar: torch.tensor, beta: float):
    BCE = -torch.mean(
        torch.sum(
            x * F.log_softmax(x_tilde, 1),
            dim=1
        ),
    )

    KL = -0.5 * torch.mean(
        torch.sum(
            1.0 + logvar - logvar.exp() - mu.pow(2),
            dim=1
        )
    )

    return BCE + beta * KL
