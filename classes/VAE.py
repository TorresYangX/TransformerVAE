import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class VAE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, latent_dim, vocab_size, BATCH_SIZE, trajectory_length):
        super(VAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.BATCH_SIZE = BATCH_SIZE
        
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.encoder_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.encoder_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.outfc = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.hidden_init = (torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(device)),
                            torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(device)))

        self.softsign = nn.Softsign()

    def encode(self, x):
        x = self.embedding(x.to(torch.int64))
        h, _ = self.encoder_lstm(x, self.hidden_init)
        h = h[:, -1:, :]
        mu = self.softsign(self.encoder_mu(h))
        logvar = self.softsign(self.encoder_logvar(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = z.repeat(1, self.trajectory_length, 1)
        h, _ = self.decoder_lstm(z, self.hidden_init)
        x_hat = self.softsign(self.decoder_fc(h))
        x_hat = self.outfc(x_hat)
        x_hat = self.softmax(x_hat)
        return x_hat

    def loss_fn(self, x_hat, x, mu, logvar):
        KL_WEIGHT = 0.01
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        targets = torch.flatten(x)
        logits = torch.flatten(x_hat, start_dim=0, end_dim=1)
        BCE = criterion(logits.float(), targets.long()).mean()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL_WEIGHT * KLD

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {
            "x_hat": x_hat,
            "mu": mu,
            "logvar": logvar,
        }






