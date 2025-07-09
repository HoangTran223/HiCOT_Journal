import numpy as np

def build_cooccurrence_matrix(train_data, vocab, norm='log', window_size=5):
    """
    Build co-occurrence matrix from BOW data using sliding window approach.
    
    CÔNG THỨC:
    1. Chuyển BOW → word sequence: [word_idx] * min(count, 3)
    2. Sliding window: X[i,j] += weight, weight = 1/distance
    3. Symmetric: X = (X + X.T) / 2
    4. Normalize: log(1+X) / max(log(1+X))
    
    Args:
        train_data: BOW matrix [num_docs, vocab_size] 
        vocab: vocabulary list
        norm: normalization method ('log', 'max', 'sqrt')
        window_size: context window size
    
    Returns:
        X_norm: normalized co-occurrence matrix [vocab_size, vocab_size]
    """
    V = len(vocab)
    X = np.zeros((V, V), dtype=np.float32)
    
    print(f"Building co-occurrence matrix from {len(train_data)} documents...")
    
    # Main sliding window approach
    for doc_idx, doc in enumerate(train_data):
        if doc_idx % 1000 == 0:
            print(f"Processing doc {doc_idx}/{len(train_data)}")
        
        # Convert BOW to word sequence
        word_sequence = []
        for word_idx, count in enumerate(doc):
            if count > 0 and word_idx < V:
                # Giới hạn số lần lặp để tránh sequence quá dài
                word_sequence.extend([word_idx] * min(int(count), 3))
        
        if len(word_sequence) < 2:
            continue
        
        # Sliding window co-occurrence counting
        for i in range(len(word_sequence)):
            center_word = word_sequence[i]
            start = max(0, i - window_size)
            end = min(len(word_sequence), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_word = word_sequence[j]
                    if center_word < V and context_word < V:
                        # Weight theo khoảng cách: càng xa thì weight càng nhỏ
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        X[center_word, context_word] += weight
    
    print(f"Co-occurrence matrix built. Non-zero entries: {np.count_nonzero(X)}")
    print(f"Max co-occurrence value: {np.max(X):.4f}")
    
    # Make symmetric
    X = (X + X.T) / 2
    
    # Normalize
    if norm == 'log':
        X_log = np.log1p(X)  # log(1 + X)
        X_norm = X_log / np.max(X_log) if np.max(X_log) > 0 else X_log + 1e-6
    elif norm == 'max':
        X_norm = X / np.max(X) if np.max(X) > 0 else X + 1e-6
    elif norm == 'sqrt':
        X_sqrt = np.sqrt(X)
        X_norm = X_sqrt / np.max(X_sqrt) if np.max(X_sqrt) > 0 else X_sqrt + 1e-6
    else:
        X_norm = X if np.max(X) > 0 else X + 1e-6
    
    print(f"Normalized matrix - Max: {np.max(X_norm):.6f}, Mean: {np.mean(X_norm):.6f}, Non-zero: {np.count_nonzero(X_norm)}")
    return X_norm