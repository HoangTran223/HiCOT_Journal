import numpy as np
import torch
from collections import defaultdict
from scipy.sparse import csr_matrix
import logging

class AdvancedCooccurrenceBuilder:
    """
    Advanced co-occurrence matrix builder with multiple coherence metrics
    """
    def __init__(self, window_sizes=[2, 5, 10], min_count=5, subsampling_threshold=1e-3):
        self.window_sizes = window_sizes
        self.min_count = min_count
        self.subsampling_threshold = subsampling_threshold
        
    def build_multi_scale_cooccurrence(self, train_data, vocab, norm='pmi'):
        """
        Build multi-scale co-occurrence matrices for different window sizes
        """
        V = len(vocab)
        cooc_matrices = {}
        
        # Convert train_data to word sequences if it's bow format
        if isinstance(train_data, np.ndarray) and train_data.ndim == 2:
            word_sequences = self._bow_to_sequences(train_data)
        else:
            word_sequences = train_data
            
        # Calculate word frequencies for subsampling
        word_counts = self._calculate_word_counts(word_sequences, V)
        word_freqs = word_counts / np.sum(word_counts)
        
        # Build matrices for different window sizes
        for window_size in self.window_sizes:
            logging.info(f"Building co-occurrence matrix for window size {window_size}")
            
            # Initialize co-occurrence matrix
            cooc_matrix = np.zeros((V, V), dtype=np.float32)
            
            # Process each document
            for doc in word_sequences:
                # Apply subsampling to frequent words
                doc_filtered = self._subsample_frequent_words(doc, word_freqs)
                
                # Build co-occurrence for this document
                for i, center_word in enumerate(doc_filtered):
                    # Define context window
                    start = max(0, i - window_size)
                    end = min(len(doc_filtered), i + window_size + 1)
                    
                    for j in range(start, end):
                        if i != j and center_word < V and doc_filtered[j] < V:
                            # Distance-weighted co-occurrence
                            distance = abs(i - j)
                            weight = 1.0 / distance
                            cooc_matrix[center_word, doc_filtered[j]] += weight
            
            # Normalize matrix
            if norm == 'pmi':
                cooc_matrix = self._compute_pmi_matrix(cooc_matrix, word_counts)
            elif norm == 'npmi':
                cooc_matrix = self._compute_npmi_matrix(cooc_matrix, word_counts)
            elif norm == 'log':
                cooc_matrix = np.log1p(cooc_matrix)
                cooc_matrix = cooc_matrix / (np.max(cooc_matrix) + 1e-8)
            
            cooc_matrices[window_size] = cooc_matrix
            
        return cooc_matrices
    
    def _bow_to_sequences(self, bow_data):
        """Convert bag-of-words to word sequences"""
        sequences = []
        for doc in bow_data:
            sequence = []
            for word_idx, count in enumerate(doc):
                sequence.extend([word_idx] * int(count))
            sequences.append(sequence)
        return sequences
    
    def _calculate_word_counts(self, word_sequences, vocab_size):
        """Calculate word frequencies across all documents"""
        word_counts = np.zeros(vocab_size)
        for doc in word_sequences:
            for word in doc:
                if word < vocab_size:
                    word_counts[word] += 1
        return word_counts
    
    def _subsample_frequent_words(self, doc, word_freqs):
        """Apply subsampling to frequent words"""
        filtered_doc = []
        for word in doc:
            if word < len(word_freqs):
                freq = word_freqs[word]
                prob_keep = min(1.0, np.sqrt(self.subsampling_threshold / freq))
                if np.random.random() < prob_keep:
                    filtered_doc.append(word)
        return filtered_doc
    
    def _compute_pmi_matrix(self, cooc_matrix, word_counts):
        """Compute Pointwise Mutual Information matrix"""
        total_count = np.sum(word_counts)
        word_probs = word_counts / total_count
        
        # Add smoothing
        cooc_matrix_smooth = cooc_matrix + 1e-8
        joint_probs = cooc_matrix_smooth / np.sum(cooc_matrix_smooth)
        
        pmi_matrix = np.zeros_like(cooc_matrix)
        for i in range(len(word_counts)):
            for j in range(len(word_counts)):
                if joint_probs[i, j] > 0 and word_probs[i] > 0 and word_probs[j] > 0:
                    pmi_matrix[i, j] = np.log(joint_probs[i, j] / (word_probs[i] * word_probs[j]))
        
        # Clip negative PMI values
        pmi_matrix = np.maximum(pmi_matrix, 0)
        return pmi_matrix
    
    def _compute_npmi_matrix(self, cooc_matrix, word_counts):
        """Compute Normalized Pointwise Mutual Information matrix"""
        pmi_matrix = self._compute_pmi_matrix(cooc_matrix, word_counts)
        
        # Normalize by negative log of joint probability
        total_count = np.sum(word_counts)
        cooc_matrix_smooth = cooc_matrix + 1e-8
        joint_probs = cooc_matrix_smooth / np.sum(cooc_matrix_smooth)
        
        npmi_matrix = np.zeros_like(pmi_matrix)
        for i in range(pmi_matrix.shape[0]):
            for j in range(pmi_matrix.shape[1]):
                if joint_probs[i, j] > 0:
                    npmi_matrix[i, j] = pmi_matrix[i, j] / (-np.log(joint_probs[i, j]))
        
        return npmi_matrix