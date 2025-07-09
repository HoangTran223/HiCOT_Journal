import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from .DT import DT
from .TP import TP
import torch_kmeans
import logging
import sentence_transformers
import hdbscan
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from utils import static_utils
from .glove_cooccurrence import build_cooccurrence_matrix
import os

class HiCOT_C(nn.Module):
    def __init__(self, vocab_size, data_name = '20NG', num_topics=50, en_units=200, dropout=0.,
                threshold_epoch = 10, doc2vec_size=384, pretrained_WE=None, embed_size=200, beta_temp=0.2, 
                weight_loss_CLT=1.0, threshold_cluster=30, weight_loss_ECR=250.0, alpha_ECR=20.0, weight_loss_TP = 250.0, 
                alpha_TP = 20.0, alpha_DT: float=3.0, weight_loss_DT = 10.0, vocab = None, doc_embeddings=None,
                weight_loss_CLC= 1.0, max_clusters = 50, method_CL = 'HAC', metric_CL = 'euclidean', sinkhorn_max_iter=5000,
                weight_loss_coherence=1.0, cooc_norm='log', cooc_window_size=10, train_data=None):

        super().__init__()

        self.method_CL = method_CL
        self.metric_CL = metric_CL
        self.threshold_epoch = threshold_epoch
        self.threshold_cluster = threshold_cluster
        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.data_name = data_name

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor(
            (np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor(
            (((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(
                torch.empty(vocab_size, embed_size))

        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))
        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)
        self.weight_loss_CLT = weight_loss_CLT
        self.weight_loss_CLC = weight_loss_CLC

        self.max_clusters = max_clusters
        self.vocab = vocab
        self.matrixP = None
        self.DT = DT(weight_loss_DT, alpha_TP)

        self.doc_embeddings = doc_embeddings.to(self.topic_embeddings.device)
        self.group_topic = None

        self.TP = TP(weight_loss_TP, alpha_TP)

        self.document_emb_prj = nn.Sequential(
            nn.Linear(doc2vec_size, embed_size), 
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.topic_embeddings.device)

        self.topics = []
        self.topic_index_mapping = {}

        # GloVe-style topic coherence regularization
        self.weight_loss_coherence = weight_loss_coherence
        self.cooc_norm = cooc_norm
        self.cooc_window_size = cooc_window_size
        self.cooc_matrix = None
        
        # FIXED: Proper co-occurrence matrix building
        if train_data is not None and vocab is not None:
            precomputed_dir = 'precomputed'
            os.makedirs(precomputed_dir, exist_ok=True)
            cooc_matrix_path = os.path.join(precomputed_dir, f'cooc_{data_name}_{cooc_norm}_w{cooc_window_size}.npy')

            if os.path.exists(cooc_matrix_path):
                logging.info(f"Loading co-occurrence matrix from {cooc_matrix_path}")
                cooc_matrix_np = np.load(cooc_matrix_path)
                self.cooc_matrix = torch.tensor(cooc_matrix_np, dtype=torch.float32)
            else:
                logging.info(f"Building co-occurrence matrix for {data_name}...")
                cooc_matrix_np = build_cooccurrence_matrix(
                    train_data, vocab, norm=cooc_norm, window_size=cooc_window_size
                )
                
                # CRITICAL: Ensure matrix is not all zeros
                if np.sum(cooc_matrix_np) == 0:
                    logging.warning("Co-occurrence matrix is all zeros! Using simple word co-occurrence.")
                    cooc_matrix_np = self._build_simple_cooccurrence(train_data, vocab)
                
                logging.info(f"Saving co-occurrence matrix to {cooc_matrix_path}")
                np.save(cooc_matrix_path, cooc_matrix_np)
                self.cooc_matrix = torch.tensor(cooc_matrix_np, dtype=torch.float32)
                
            logging.info(f"Co-occurrence matrix stats: max={torch.max(self.cooc_matrix):.6f}, mean={torch.mean(self.cooc_matrix):.6f}, non_zero={torch.count_nonzero(self.cooc_matrix)}")

    def _build_simple_cooccurrence(self, train_data, vocab):
        """Fallback: simple document-level co-occurrence when sliding window fails"""
        V = len(vocab)
        cooc_matrix = np.zeros((V, V), dtype=np.float32)
        
        if isinstance(train_data, np.ndarray) and train_data.ndim == 2:
            for doc in train_data:
                words = np.where(doc > 0)[0]  # Get words present in document
                for i in range(len(words)):
                    for j in range(len(words)):
                        if i != j and words[i] < V and words[j] < V:
                            cooc_matrix[words[i], words[j]] += 1
        
        # Normalize by max value
        if np.max(cooc_matrix) > 0:
            cooc_matrix = cooc_matrix / np.max(cooc_matrix)
        
        return cooc_matrix

    def create_group_topic(self):
        with torch.no_grad():  
            distances = torch.cdist(self.topic_embeddings, self.topic_embeddings, p=2)
            distances = distances.detach().cpu().numpy()

        if self.method_CL == 'HAC':
            Z = linkage(distances, method='average', optimal_ordering=True) 
            group_id = fcluster(Z, t= self.max_clusters, criterion='maxclust') - 1
        
        elif self.method_CL == 'HDBSCAN':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
            group_id = clusterer.fit_predict(distances)
        
        else:
            raise ValueError(f"Invalid method_CL: {self.method_CL}")
        
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

        for word_idx, word in enumerate(self.vocab):
            topic_idx = self.word_to_topic_by_similarity(word)
            word_topic_assignments[topic_idx].append(word_idx)
        return word_topic_assignments


    def word_to_topic_by_similarity(self, word):
        word_idx = self.vocab.index(word)
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


    def get_beta(self):
        dist = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def get_loss_coherence(self):
        """
        COMPLETELY FIXED: GloVe-style coherence loss with proper gradients and validation
        """
        if self.weight_loss_coherence <= 1e-8 or self.cooc_matrix is None:
            return torch.tensor(0.0, device=self.topic_embeddings.device)
            
        X = self.cooc_matrix.to(self.topic_embeddings.device)
        
        # Check if co-occurrence matrix is meaningful
        if torch.sum(X) < 1e-6:
            logging.warning("Co-occurrence matrix is nearly empty, skipping coherence loss")
            return torch.tensor(0.0, device=self.topic_embeddings.device)
        
        # Get beta WITH gradients - this is crucial!
        beta = self.get_beta()  # shape: [num_topics, vocab_size] - WITH GRADIENTS!
        
        # Compute predicted co-occurrence from topic-word distributions
        pred = torch.matmul(beta.t(), beta)  # shape: [vocab_size, vocab_size]
        pred = pred.clamp(min=1e-8)  # avoid division by zero
        
        # Normalize X to have reasonable scale
        X = X.clamp(min=1e-8)
        
        # Use Frobenius norm loss instead of KL divergence for stability
        coherence_loss = F.mse_loss(pred, X)
        
        # Alternative: Use only upper triangle to avoid diagonal dominance
        # mask = torch.triu(torch.ones_like(X), diagonal=1).bool()
        # coherence_loss = F.mse_loss(pred[mask], X[mask])
        
        final_loss = self.weight_loss_coherence * coherence_loss
        
        # Log for debugging
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            logging.warning(f"Invalid coherence loss: {final_loss}, setting to 0")
            return torch.tensor(0.0, device=self.topic_embeddings.device)
        
        return final_loss

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def get_representation(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        return theta, mu, logvar

    def encode(self, input):
        theta, mu, logvar = self.get_representation(input)
        loss_KL = self.compute_loss_KL(mu, logvar)
        return theta, loss_KL

    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term +
                     logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR
    
    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def create_matrixP(self, minibatch_embeddings, indices):
        num_minibatch = len(indices)
        self.matrixP = torch.ones(
            (num_minibatch, num_minibatch), device=self.topic_embeddings.device) / num_minibatch

        norm_embeddings = F.normalize(minibatch_embeddings, p=2, dim=1).clamp(min=1e-6)
        self.matrixP = torch.matmul(norm_embeddings, norm_embeddings.T)
        self.matrixP = self.matrixP.clamp(min=1e-4)
        return self.matrixP

    def get_loss_TP(self, doc_embeddings, indices):
        indices = indices.to(self.doc_embeddings.device)
        minibatch_embeddings = self.doc_embeddings[indices]

        cost = self.pairwise_euclidean_distance(minibatch_embeddings, minibatch_embeddings) \
           + 1e1 * torch.ones(minibatch_embeddings.size(0), minibatch_embeddings.size(0)).to(minibatch_embeddings.device)

        self.matrixP = self.create_matrixP(minibatch_embeddings, indices)
        loss_TP = self.TP(cost, self.matrixP)
        return loss_TP
    
    
    def get_loss_DT(self, doc_embeddings):
        document_prj = self.document_emb_prj(doc_embeddings)

        loss_DT, transp_DT = self.DT(document_prj, self.topic_embeddings)
        return loss_DT


    def forward(self, indices, input, epoch_id=None, doc_embeddings=None):
        # FIXED: Proper input handling
        bow = input[0]
        
        # FIXED: Proper device handling for doc_embeddings
        if doc_embeddings is not None:
            doc_embeddings = doc_embeddings.to(self.topic_embeddings.device)
        else:
            # Use stored doc_embeddings if available
            if hasattr(self, 'doc_embeddings') and self.doc_embeddings is not None:
                doc_embeddings = self.doc_embeddings

        rep, mu, logvar = self.get_representation(bow)
        loss_KL = self.compute_loss_KL(mu, logvar)
        theta = rep
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()
        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()
        
        # FIXED: Proper conditional loss calculation
        loss_TP = 0.0
        loss_DT = 0.0
        if doc_embeddings is not None:
            loss_TP = self.get_loss_TP(doc_embeddings, indices)
            loss_DT = self.get_loss_DT(doc_embeddings)

        loss_CLC = 0.0
        loss_CLT = 0.0

        # FIXED: Proper epoch-based loss activation
        if epoch_id is not None and epoch_id >= self.threshold_epoch:
            if epoch_id == self.threshold_epoch or (epoch_id > self.threshold_epoch and epoch_id % self.threshold_cluster == 0):
                self.create_group_topic()

            if self.weight_loss_CLC > 1e-8:
                loss_CLC = self.get_loss_CLC()
            
            if self.weight_loss_CLT > 1e-8:
                loss_CLT = self.get_loss_CLT()
        
        # FIXED: Always compute coherence loss (not epoch-dependent)
        loss_coherence = self.get_loss_coherence()

        loss = loss_TM + loss_ECR + loss_TP + loss_DT + loss_CLC + loss_CLT + loss_coherence
        
        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'loss_DT': loss_DT,
            'loss_TP': loss_TP,
            'loss_CLC': loss_CLC,
            'loss_CLT': loss_CLT,
            'loss_coherence': loss_coherence
        }

        return rst_dict

