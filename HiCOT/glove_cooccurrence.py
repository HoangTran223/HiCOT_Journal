import numpy as np

def build_cooccurrence_matrix(train_data, vocab, norm='log', window_size=5):
    """
    IMPROVED: Build co-occurrence matrix X from train_data with proper windowing.
    train_data: bag-of-words matrix [num_docs, vocab_size] or list of word index lists
    vocab: list of vocab words (for size)
    window_size: context window size for co-occurrence
    Returns: X' (normalized co-occurrence matrix), shape [V, V]
    """
    V = len(vocab)
    X = np.zeros((V, V), dtype=np.float32)
    
    # If train_data is bag-of-words matrix
    if isinstance(train_data, np.ndarray) and train_data.ndim == 2:
        print("Building co-occurrence matrix from BOW data...")
        for doc_idx, doc in enumerate(train_data):
            if doc_idx % 1000 == 0:
                print(f"Processing doc {doc_idx}/{len(train_data)}")
                
            # Convert BOW to word sequence
            word_sequence = []
            for word_idx, count in enumerate(doc):
                if count > 0 and word_idx < V:
                    # Add words proportional to count (but limit to avoid huge sequences)
                    word_sequence.extend([word_idx] * min(int(count), 3))
            
            if len(word_sequence) < 2:
                continue
                
            # Sliding window co-occurrence
            for i in range(len(word_sequence)):
                center_word = word_sequence[i]
                start = max(0, i - window_size)
                end = min(len(word_sequence), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j and center_word < V and word_sequence[j] < V:
                        # Distance-weighted co-occurrence
                        distance = abs(i - j)
                        weight = 1.0 / distance if distance > 0 else 1.0
                        X[center_word, word_sequence[j]] += weight
    else:
        # Assume train_data is list of word index lists
        print("Building co-occurrence matrix from token sequences...")
        for doc_idx, doc in enumerate(train_data):
            if doc_idx % 1000 == 0:
                print(f"Processing doc {doc_idx}/{len(train_data)}")
                
            for i, wi in enumerate(doc):
                if wi >= V:
                    continue
                start = max(0, i - window_size)
                end = min(len(doc), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j and doc[j] < V:
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        X[wi, doc[j]] += weight

    print(f"Co-occurrence matrix built. Non-zero entries: {np.count_nonzero(X)}")
    print(f"Max co-occurrence value: {np.max(X)}")
    
    # CRITICAL: Check if matrix is meaningful
    if np.count_nonzero(X) == 0:
        print("WARNING: Co-occurrence matrix is all zeros! Building simple document-level co-occurrence.")
        # Fallback to simple document-level co-occurrence
        for doc_idx, doc in enumerate(train_data[:min(1000, len(train_data))]):  # Limit to first 1000 docs
            if isinstance(doc, np.ndarray):
                words = np.where(doc > 0)[0]
            else:
                words = list(set(doc))
            
            for i in range(len(words)):
                for j in range(len(words)):
                    if i != j and words[i] < V and words[j] < V:
                        X[words[i], words[j]] += 1
    
    # Make symmetric
    X = (X + X.T) / 2
    
    # Normalize X to meaningful range
    if norm == 'max':
        if np.max(X) > 0:
            X_norm = X / np.max(X)
        else:
            X_norm = X + 1e-6 * np.ones_like(X)  # Add small constant if all zeros
    elif norm == 'log':
        X_log = np.log1p(X)  # log(1 + X)
        if np.max(X_log) > 0:
            X_norm = X_log / np.max(X_log)
        else:
            X_norm = X_log + 1e-6 * np.ones_like(X_log)
    elif norm == 'sqrt':
        X_norm = np.sqrt(X)
        if np.max(X_norm) > 0:
            X_norm = X_norm / np.max(X_norm)
        else:
            X_norm = X_norm + 1e-6 * np.ones_like(X_norm)
    else:
        X_norm = X
        if np.max(X_norm) == 0:
            X_norm = X_norm + 1e-6 * np.ones_like(X_norm)
        
    print(f"Normalized matrix - Max: {np.max(X_norm):.6f}, Mean: {np.mean(X_norm):.6f}, Non-zero: {np.count_nonzero(X_norm)}")
    return X_norm
