import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, latent_dim, vocab_size, BATCH_SIZE, trajectory_length):
        super(AE, self).__init__()

        self.embedding_dim = embedding_dim
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.BATCH_SIZE = BATCH_SIZE

        self.embedding = nn.Embedding(num_embeddings= self.vocab_size, embedding_dim= self.embedding_dim)
        self.encoder_lstm = nn.LSTM(input_size= self.embedding_dim, hidden_size= self.hidden_dim, batch_first=True)
        self.encoder = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_lstm = nn.LSTM(input_size= self.latent_dim, hidden_size= self.hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.outfc = nn.Linear( self.embedding_dim,  self.vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.hidden_init = (torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(device)),
                            torch.autograd.Variable(torch.zeros(1, self.BATCH_SIZE, self.hidden_dim).to(device)))

        self.softsign = nn.Softsign()

    def encode(self, x):
        x = self.embedding(x.to(torch.int64))
        h, _ = self.encoder_lstm(x, self.hidden_init)
        h = h[:, -1:, :]
        h = self.softsign(self.encoder(h))
        return h

    def decode(self, h):
        h = h.repeat(1, self.trajectory_length, 1)
        h, _ = self.decoder_lstm(h, self.hidden_init)
        x_hat = self.softsign(self.decoder_fc(h))
        x_hat = self.outfc(x_hat)
        x_hat = self.softmax(x_hat)
        return x_hat
    
    def loss_fn(self, logits, targets, prob):
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        targets = torch.flatten(targets)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)
        BCE = criterion(logits.float(), targets.long()).mean()
        return {
            "Loss": BCE,
            "CrossEntropy": BCE,
        }

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return {
            'logits': x_hat,
            'prob': h
        }