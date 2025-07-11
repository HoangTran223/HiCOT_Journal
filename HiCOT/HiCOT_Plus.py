import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .HiCOT import HiCOT

class HiCOT_Plus(HiCOT):
    def __init__(self, weight_loss_DTC=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight_loss_DTC = weight_loss_DTC
        print(f"HiCOT_Plus model initialized with weight_loss_DTC: {self.weight_loss_DTC}")

    def get_loss_DTC(self, doc_embeddings, theta, margin=0.2):
        """
        Document-Topic Contrastive Loss.
        Pulls document embeddings closer to their main topic embeddings and pushes them away from irrelevant topics.
        """
        loss_DTC = 0.0
        
        # Anchor: document embeddings (projected)
        anchor_doc_emb = self.document_emb_prj(doc_embeddings)

        # Find positive and negative topics based on theta
        _, top_positive_indices = torch.topk(theta, k=1, dim=1)
        
        # For simplicity, we sample one negative topic for each document.
        # A good negative topic is one with a low theta value.
        _, bottom_negative_indices = torch.topk(theta, k=1, dim=1, largest=False)

        # Positive: topic embeddings corresponding to the highest theta
        # Use .squeeze() to remove the extra dimension from topk
        positive_topic_emb = self.topic_embeddings[top_positive_indices.squeeze(1)]

        # Negative: topic embeddings corresponding to the lowest theta
        negative_topic_emb = self.topic_embeddings[bottom_negative_indices.squeeze(1)]

        # Using cosine similarity for contrastive loss
        pos_similarity = F.cosine_similarity(anchor_doc_emb, positive_topic_emb)
        neg_similarity = F.cosine_similarity(anchor_doc_emb, negative_topic_emb)

        # We want to maximize positive similarity and minimize negative similarity.
        # Loss = max(0, margin - pos_similarity + neg_similarity)
        loss = torch.clamp(margin - pos_similarity + neg_similarity, min=0.0)
        
        loss_DTC = loss.mean()
        
        return loss_DTC * self.weight_loss_DTC


    def forward(self, indices, input, epoch_id=None, doc_embeddings=None):
        bow = input[0]
        doc_embeddings = doc_embeddings.to(self.topic_embeddings.device)

        rep, mu, logvar = self.get_representation(bow)
        loss_KL = self.compute_loss_KL(mu, logvar)
        theta = rep
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()
        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()
        loss_TP = self.get_loss_TP(doc_embeddings, indices)
        loss_DT = self.get_loss_DT(doc_embeddings)
        
        # New Document-Topic Contrastive Loss
        loss_DTC = 0.0
        loss_CLC = 0.0
        loss_CLT = 0.0

        if epoch_id >= self.threshold_epoch and (epoch_id == self.threshold_epoch or (epoch_id > self.threshold_epoch and epoch_id % self.threshold_cluster == 0)):
            self.create_group_topic()

        if epoch_id >= self.threshold_epoch and self.weight_loss_CLC != 0:
            loss_CLC = self.get_loss_CLC()
        
        if epoch_id >= self.threshold_epoch and self.weight_loss_CLT != 0:
            loss_CLT = self.get_loss_CLT()
        
        if epoch_id >= self.threshold_epoch and self.weight_loss_DTC != 0:
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
            'loss_DTC': loss_DTC, # Add new loss to results
        }

        return rst_dict
