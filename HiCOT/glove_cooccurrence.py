import numpy as np

def build_cooccurrence_matrix(train_data, vocab, norm='log'):
    """
    Build co-occurrence matrix X from train_data.
    train_data: bag-of-words matrix [num_docs, vocab_size] or list of word index lists
    vocab: list of vocab words (for size)
    Returns: X' (normalized co-occurrence matrix), shape [V, V]
    """
    V = len(vocab)
    X = np.zeros((V, V), dtype=np.float32)
    # If train_data is bag-of-words matrix
    if isinstance(train_data, np.ndarray) and train_data.ndim == 2:
        for doc in train_data:
            words = np.where(doc > 0)[0]
            for i in range(len(words)):
                for j in range(len(words)):
                    if i != j:
                        X[words[i], words[j]] += 1
    else:
        # Assume train_data is list of word index lists
        for doc in train_data:
            unique_words = set(doc)
            for wi in unique_words:
                for wj in unique_words:
                    if wi != wj:
                        X[wi, wj] += 1

    # Normalize X to [0,1]
    if norm == 'max':
        X_norm = X / (np.max(X) + 1e-8)
    elif norm == 'log':
        X_norm = np.log1p(X) / (np.max(np.log1p(X)) + 1e-8)
    else:
        X_norm = X
    return X_norm
