import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from .OT import OT


class ECRTM(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200, is_OT=False,
                    cluster_distribution=None, cluster_mean=None, cluster_label=None, sinkhorn_alpha = 20.0, weight_OT=100.0,
                    beta_temp=0.2, weight_loss_ECR=250.0, alpha_ECR=20.0, sinkhorn_max_iter=1000,
                    device='cuda'):
        super().__init__()

        self.device = device
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.weight_OT = weight_OT
        self.is_OT = is_OT

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))


        # Add OT
        self.cluster_mean = nn.Parameter(torch.from_numpy(cluster_mean).float(), requires_grad=False)
        self.cluster_distribution = nn.Parameter(torch.from_numpy(cluster_distribution).float(), requires_grad=False)
        self.cluster_label = cluster_label
        if not isinstance(self.cluster_label, torch.Tensor):
            self.cluster_label = torch.tensor(self.cluster_label, dtype=torch.long, device='cuda')
        else:
            self.cluster_label = self.cluster_label.to(device='cuda', dtype=torch.long)
        
        self.map_t2c = nn.Linear(self.word_embeddings.shape[1], self.cluster_mean.shape[1], bias=False)
        self.OT = OT(weight_OT, sinkhorn_alpha, sinkhorn_max_iter)
        # # 

        

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)

        self.to(self.device)

    # Same
    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    # Same
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    # Same
    def encode(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_KL(mu, logvar)

        return theta, loss_KL


    # Same
    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    # Same
    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    # Same
    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    # Same
    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost


    # Thêm
    def get_loss_OT(self, input, indices):
        bow = input[0]
        theta, _ = self.encode(bow)
        cd_batch = self.cluster_distribution[indices]  
        cost = self.pairwise_euclidean_distance(self.cluster_mean, self.map_t2c(self.topic_embeddings))  
        loss_OT = self.weight_OT * self.OT(theta, cd_batch, cost)  
        return loss_OT


    def forward(self, indices, input, epoch_id=None):
        # input = input['data']
        bow = input[0].to(self.device)
        theta, loss_KL = self.encode(bow)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()

        if self.is_OT:
            loss_OT = self.get_loss_OT(input, indices)
        else:
            loss_OT = 0.0

        loss = loss_TM + loss_ECR + loss_OT

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'loss_OT': loss_OT
        }

        return rst_dict

