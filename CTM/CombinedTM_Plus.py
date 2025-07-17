import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CombinedTM import CombinedTM
from HiCOT.TP import TP
import torch_kmeans
import logging
import sentence_transformers
import hdbscan
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from utils import static_utils


class CombinedTM_Plus(CombinedTM):
    def __init__(self, vocab=None, weight_loss_CLT=1.0, weight_loss_CLC=1.0, weight_loss_DTC=1.0, weight_loss_TP=1.0,
                 threshold_epoch=10, threshold_cluster=30, metric_CL='cosine', max_clusters=9, doc_embeddings=None,
                 alpha_TP=20.0, sinkhorn_max_iter=1000, doc2vec_size=384, **kwargs):
        super().__init__(**kwargs)
        
        # Fixed embed_size for CombinedTM_Plus
        embed_size = 200
        
        self.weight_loss_CLT = weight_loss_CLT
        self.weight_loss_CLC = weight_loss_CLC
        self.weight_loss_DTC = weight_loss_DTC
        self.weight_loss_TP = weight_loss_TP
        self.alpha_TP = alpha_TP
        self.threshold_epoch = threshold_epoch
        self.threshold_cluster = threshold_cluster
        self.metric_CL = metric_CL
        self.max_clusters = max_clusters
        
        # Store contextual_embed_size from parent class
        self.contextual_embed_size = kwargs.get('contextual_embed_size', 768)
        
        self.doc_embeddings = doc_embeddings.to(self.fcd1.weight.device) if doc_embeddings is not None else None
        self.vocab = vocab
        self.group_topic = None
        self.topics = []
        self.topic_index_mapping = {}
        self.matrixP = None

        # Initialize topic embeddings for CombinedTM
        self.topic_embeddings = torch.empty((self.num_topics, embed_size))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        # Initialize word embeddings for contrastive losses
        self.word_embeddings = torch.empty((self.vocab_size, embed_size))
        nn.init.trunc_normal_(self.word_embeddings, std=0.1)
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        # Add document_emb_prj layer like HiCOT - project contextual embeddings to embed_size
        self.document_emb_prj = nn.Sequential(
            nn.Linear(self.contextual_embed_size, embed_size), 
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.))
        ).to(self.fcd1.weight.device)

        # Add TP class like HiCOT
        self.TP = TP(self.weight_loss_TP, self.alpha_TP, sinkhorn_max_iter)

    def create_group_topic(self):
        with torch.no_grad():
            distances = torch.cdist(self.topic_embeddings, self.topic_embeddings, p=2)
            distances = distances.detach().cpu().numpy()

        Z = linkage(distances, method='average', optimal_ordering=True)
        group_id = fcluster(Z, t=self.max_clusters, criterion='maxclust') - 1

        self.group_topic = [[] for _ in range(self.max_clusters)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i]].append(i)

        topic_idx_counter = 0
        word_topic_assignments = self.get_word_topic_assignments()
        for topic_idx in range(self.num_topics):
            self.topics.append(word_topic_assignments[topic_idx])
            self.topic_index_mapping[topic_idx] = topic_idx_counter
            topic_idx_counter += 1

    def get_word_topic_assignments(self):
        word_topic_assignments = [[] for _ in range(self.num_topics)]
        if self.vocab is not None:
            for word_idx, word in enumerate(self.vocab):
                topic_idx = self.word_to_topic_by_similarity(word)
                word_topic_assignments[topic_idx].append(word_idx)
        else:
            for word_idx in range(self.word_embeddings.shape[0]):
                topic_idx = self.word_to_topic_by_similarity(word_idx)
                word_topic_assignments[topic_idx].append(word_idx)
        return word_topic_assignments

    def word_to_topic_by_similarity(self, word):
        if self.vocab is not None and isinstance(word, str):
            word_idx = self.vocab.index(word)
        else:
            word_idx = word
        word_embedding = self.word_embeddings[word_idx].unsqueeze(0)
        similarity_scores = F.cosine_similarity(word_embedding, self.topic_embeddings)
        topic_idx = torch.argmax(similarity_scores).item()
        return topic_idx

    def get_loss_CLC(self, margin=0.2, num_negatives=10):
        loss_CLC = 0.0
        for group_idx, group_topics in enumerate(self.group_topic):
            if len(group_topics) < 1:
                continue
            anchor = torch.mean(self.topic_embeddings[group_topics], dim=0, keepdim=True)
            positive_topic_idx = np.random.choice(group_topics)
            positive = self.topic_embeddings[positive_topic_idx].unsqueeze(0)
            negative_candidates = []
            for neg_group_idx, neg_group_topics in enumerate(self.group_topic):
                if neg_group_idx != group_idx:
                    negative_candidates.extend(neg_group_topics)
            if len(negative_candidates) < num_negatives:
                continue
            negative_topic_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
            negatives = self.topic_embeddings[negative_topic_idxes]
            if self.metric_CL == 'euclidean':
                pos_distance = F.pairwise_distance(anchor, positive)
                neg_distances = F.pairwise_distance(anchor.repeat(num_negatives, 1), negatives)
            elif self.metric_CL == 'cosine':
                pos_similarity = F.cosine_similarity(anchor, positive)
                neg_similarities = F.cosine_similarity(anchor.repeat(num_negatives, 1), negatives)
                pos_distance = 1 - pos_similarity
                neg_distances = 1 - neg_similarities
            else:
                raise ValueError(f"Invalid metric_CL: {self.metric_CL}")
            loss = torch.clamp(pos_distance - neg_distances + margin, min=0.0)
            loss_CLC += loss.mean()
        loss_CLC *= self.weight_loss_CLC
        return loss_CLC

    def get_loss_CLT(self, margin=0.2, num_negatives=10):
        loss_CLT = 0.0
        for group_idx, group_topics in enumerate(self.group_topic):
            for anchor_topic_idx in group_topics:
                anchor_words_idxes = self.topics[self.topic_index_mapping[anchor_topic_idx]]
                if len(anchor_words_idxes) < 1:
                    continue
                anchor = torch.mean(self.word_embeddings[anchor_words_idxes], dim=0, keepdim=True)
                positive_word_idx = np.random.choice(anchor_words_idxes)
                positive = self.word_embeddings[positive_word_idx].unsqueeze(0)
                negative_candidates = []
                for neg_topic_idx in range(self.num_topics):
                    if neg_topic_idx not in group_topics:
                        negative_candidates.extend(self.topics[self.topic_index_mapping[neg_topic_idx]])
                if len(negative_candidates) < num_negatives:
                    continue
                negative_word_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
                negatives = self.word_embeddings[negative_word_idxes]
                if self.metric_CL == 'euclidean':
                    pos_distance = F.pairwise_distance(anchor, positive)
                    neg_distances = F.pairwise_distance(anchor.repeat(num_negatives, 1), negatives)
                elif self.metric_CL == 'cosine':
                    pos_similarity = F.cosine_similarity(anchor, positive)
                    neg_similarities = F.cosine_similarity(anchor.repeat(num_negatives, 1), negatives)
                    pos_distance = 1 - pos_similarity
                    neg_distances = 1 - neg_similarities
                else:
                    raise ValueError(f"Invalid metric_CL: {self.metric_CL}")
                loss = torch.clamp(pos_distance - neg_distances + margin, min=0.0)
                loss_CLT += loss.mean()
        loss_CLT *= self.weight_loss_CLT
        return loss_CLT

    def get_loss_TP(self, doc_embeddings, indices):
        indices = indices.to(self.doc_embeddings.device)
        minibatch_embeddings = self.doc_embeddings[indices]

        cost = self.pairwise_euclidean_distance(minibatch_embeddings, minibatch_embeddings) \
           + 1e1 * torch.ones(minibatch_embeddings.size(0), minibatch_embeddings.size(0)).to(minibatch_embeddings.device)

        self.matrixP = self.create_matrixP(minibatch_embeddings, indices)
        loss_TP = self.TP(cost, self.matrixP)
        return loss_TP

    def create_matrixP(self, minibatch_embeddings, indices):
        num_minibatch = len(indices)
        self.matrixP = torch.ones((num_minibatch, num_minibatch), device=self.topic_embeddings.device) / num_minibatch
        norm_embeddings = F.normalize(minibatch_embeddings, p=2, dim=1).clamp(min=1e-6)
        self.matrixP = torch.matmul(norm_embeddings, norm_embeddings.T)
        self.matrixP = self.matrixP.clamp(min=1e-4)
        return self.matrixP

    def get_loss_DTC(self, doc_embeddings, theta, margin=0.2, k=5, metric='cosine'):
        # Project contextual embeddings to same dimension as topic embeddings
        contextual_part = doc_embeddings[:, self.vocab_size:]  # Extract contextual part
        anchor_doc_emb = self.document_emb_prj(contextual_part)  # Project to embed_size (200)
        
        _, top_positive_indices = torch.topk(theta, k=1, dim=1)
        _, bottom_negative_indices = torch.topk(theta, k=k, dim=1, largest=False)
        pos_emb = self.topic_embeddings[top_positive_indices].squeeze(1)
        neg_emb = self.topic_embeddings[bottom_negative_indices]
        neg_emb = neg_emb.mean(dim=1)
        
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

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, indices, input, epoch_id=None, doc_embeddings=None):
        x = input[0]  # This should be combined BOW + contextual data
        
        # Check input shape - should be [batch_size, vocab_size + contextual_embed_size]
        expected_size = self.vocab_size + self.contextual_embed_size
        if x.shape[1] != expected_size and epoch_id == 1:  # Only warn once
            print(f"Warning: Expected input shape [batch, {expected_size}], got {x.shape}")
            print(f"vocab_size: {self.vocab_size}, contextual_embed_size: {self.contextual_embed_size}")
        
        # Use CombinedTM's get_theta method properly
        if self.training:
            theta, mu, logvar = self.get_theta(x)
        else:
            theta = self.get_theta(x)
            mu, logvar = None, None
        
        # CombinedTM reconstruction loss
        recon_x = self.decode(theta)
        if mu is not None and logvar is not None:
            loss_TM = self.loss_function(x[:, :self.vocab_size], recon_x, mu, logvar)
        else:
            # Fallback for inference mode
            loss_TM = -(x[:, :self.vocab_size] * (recon_x + 1e-10).log()).sum(axis=1).mean()

        # Progressive loss addition based on epoch
        loss_TP = 0.0
        loss_CLC = 0.0
        loss_CLT = 0.0
        loss_DTC = 0.0

        # Phase 1 (epochs 1-30): Only TM loss to establish basic topic structure
        if epoch_id is None or epoch_id <= 30:
            loss = loss_TM
        
        # Phase 2 (epochs 31-60): Add very small TP loss
        elif epoch_id <= 60:
            loss_TP = self.get_loss_TP(doc_embeddings, indices) * 0.0001  # Very small
            loss = loss_TM + loss_TP
            
        # Phase 3 (epochs 61+): Add contrastive losses gradually
        else:
            # Create topic groups
            if epoch_id >= self.threshold_epoch and (
                epoch_id == self.threshold_epoch or 
                (epoch_id > self.threshold_epoch and epoch_id % self.threshold_cluster == 0)
            ):
                self.create_group_topic()

            # Add contrastive losses gradually
            if epoch_id >= self.threshold_epoch:
                if self.weight_loss_CLC != 0:
                    loss_CLC = self.get_loss_CLC() * 0.1  # Scale down
                if self.weight_loss_CLT != 0:
                    loss_CLT = self.get_loss_CLT() * 0.1  # Scale down
                if self.weight_loss_DTC != 0:
                    loss_DTC = self.get_loss_DTC(x, theta) * 0.1  # Scale down
                    
            loss_TP = self.get_loss_TP(doc_embeddings, indices) * 0.001  # Still small
            loss = loss_TM + loss_TP + loss_CLC + loss_CLT + loss_DTC

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_TP': loss_TP,
            'loss_CLC': loss_CLC,
            'loss_CLT': loss_CLT,
            'loss_DTC': loss_DTC,
        }

        return rst_dict