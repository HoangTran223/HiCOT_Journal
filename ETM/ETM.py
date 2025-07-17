import numpy as np
import torch_kmeans
import torch
import torch.nn as nn
from .OT import OT
import torch.nn.functional as F
import logging
import sentence_transformers

class ETM(nn.Module):
    '''
        Topic Modeling in Embedding Spaces. TACL 2020

        Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei.
    '''
    def __init__(self, vocab_size, embed_size=200, num_topics=50, num_groups=10, en_units=800, dropout=0., 
                    pretrained_WE=None, train_WE=False, theta_train=False,
                    device='cuda'):
        super().__init__()
        self.device = device
        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))

        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.theta_train = theta_train
        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.ReLU(),
            nn.Linear(en_units, en_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # # Thêm 
        self.num_topics = num_topics
        self.num_groups = num_groups

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False

        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, x):
        x = x.to(self.device)
        e1 = self.encoder1(x).to(self.device)
        return self.fc21(e1), self.fc22(e1)
    

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost


    def get_theta(self, input):
        # Warn: normalize the input if use Relu.
        # https://github.com/adjidieng/ETM/issues/3
        norm_input = input / input.sum(1, keepdim=True)
        norm_input = norm_input.to(self.device)
        mu, logvar = self.encode(norm_input)
        z = self.reparameterize(mu, logvar).to(self.device)
        theta = F.softmax(z, dim=-1).to(self.device)
        if self.training or self.theta_train == True:
            return theta, mu, logvar
        else:
            return theta

    def get_beta(self):
        beta = F.softmax(torch.matmul(self.topic_embeddings.to(self.device), self.word_embeddings.T.to(self.device)), dim=1)
        return beta

    def forward(self, indices, input, avg_loss=True, epoch_id=None):
        bow = input[0].to(self.device)
        theta, mu, logvar = self.get_theta(bow)
        beta = self.get_beta()
        recon_input = torch.matmul(theta, beta)
        loss = self.loss_function(bow, recon_input, mu, logvar, avg_loss)

        rst_dict = {
            'loss': loss,
        }
        return rst_dict

    def loss_function(self, bow, recon_input, mu, logvar, avg_loss=True):
        recon_loss = -(bow * (recon_input + 1e-12).log()).sum(1)
        KLD = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        loss = (recon_loss + KLD)
        if avg_loss:
            loss = loss.mean()
        return loss