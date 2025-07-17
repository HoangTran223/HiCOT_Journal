import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .HiCOT import HiCOT

class HiCOT_Plus(HiCOT):
    def __init__(self, weight_loss_DTC=1.0, threshold_epoch=10, threshold_cluster=30, **kwargs):
        # threshold_epoch: epoch to start contrastive losses
        # threshold_cluster: interval for updating topic groups
        super().__init__(threshold_epoch=threshold_epoch, threshold_cluster=threshold_cluster, **kwargs)
        self.weight_loss_DTC = weight_loss_DTC
        print(f"HiCOT_Plus model initialized with weight_loss_DTC: {self.weight_loss_DTC}, threshold_epoch: {self.threshold_epoch}, threshold_cluster: {self.threshold_cluster}")

    def get_loss_DTC(self, doc_embeddings, theta, margin=0.2, k=5, metric='cosine'):
        # Sử dụng Triplet Loss giống CLC/CLT
        anchor_doc_emb = self.document_emb_prj(doc_embeddings)
        _, top_positive_indices = torch.topk(theta, k=1, dim=1)
        _, bottom_negative_indices = torch.topk(theta, k=k, dim=1, largest=False)
        pos_emb = self.topic_embeddings[top_positive_indices].squeeze(1)  # shape: [batch, embed]
        neg_emb = self.topic_embeddings[bottom_negative_indices]          # shape: [batch, k, embed]
        neg_emb = neg_emb.mean(dim=1)                                    # shape: [batch, embed]

        if metric == 'cosine':
            pos_similarity = F.cosine_similarity(anchor_doc_emb, pos_emb)
            neg_similarity = F.cosine_similarity(anchor_doc_emb, neg_emb)
            pos_distance = 1 - pos_similarity
            neg_distance = 1 - neg_similarity
        elif metric == 'euclidean':
            pos_distance = F.pairwise_distance(anchor_doc_emb, pos_emb)
            neg_distance = F.pairwise_distance(anchor_doc_emb, neg_emb)
        else:
            raise ValueError(f"Invalid metric: {metric}")

        loss = torch.clamp(pos_distance - neg_distance + margin, min=0.0)
        return loss.mean() * self.weight_loss_DTC

    def forward(self, indices, input, epoch_id=None, doc_embeddings=None):
        bow = input[0]
        doc_embeddings = doc_embeddings.to(self.topic_embeddings.device)
        self.current_epoch = epoch_id if epoch_id is not None else 0

        rep, mu, logvar = self.get_representation(bow)
        loss_KL = self.compute_loss_KL(mu, logvar)
        theta = rep
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()
        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()
        loss_TP = self.get_loss_TP(doc_embeddings, indices)
        # loss_DT = self.get_loss_DT(doc_embeddings)

        loss_DT = 0.0
        loss_DTC = 0.0
        loss_CLC = 0.0
        loss_CLT = 0.0

        # Run all contrastive losses together after threshold_epoch
        # Update topic groups every threshold_cluster epochs
        if epoch_id >= self.threshold_epoch and (
            epoch_id == self.threshold_epoch or 
            (epoch_id > self.threshold_epoch and epoch_id % self.threshold_cluster == 0)
        ):
            self.create_group_topic()

        if epoch_id >= self.threshold_epoch:
            if self.weight_loss_CLC != 0:
                loss_CLC = self.get_loss_CLC()
            if self.weight_loss_CLT != 0:
                loss_CLT = self.get_loss_CLT()
            if self.weight_loss_DTC != 0:
                loss_DTC = self.get_loss_DTC(doc_embeddings, theta)

        loss = loss_TM + loss_ECR + loss_TP + loss_DT + loss_CLC + loss_CLT + loss_DTC

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'loss_DT': loss_DT,
            'loss_TP': loss_TP,
            'loss_CLC': loss_CLC,
            'loss_CLT': loss_CLT,
            'loss_DTC': loss_DTC
        }

        return rst_dict
