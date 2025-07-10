import numpy as np

def build_cooccurrence_matrix(train_data, vocab, norm='log', window_size=5):
    """
    Build co-occurrence matrix from BOW data using sliding window approach (tối ưu tốc độ).

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

    for doc_idx, doc in enumerate(train_data):
        if doc_idx % 10000 == 0:
            print(f"Processing doc {doc_idx}/{len(train_data)}")
        # Vector hóa tạo word_sequence
        word_indices = np.arange(len(doc))
        counts = np.minimum(doc, 3).astype(int)
        word_sequence = np.repeat(word_indices, counts)
        if len(word_sequence) < 2:
            continue
        for i in range(len(word_sequence)):
            center = word_sequence[i]
            start = max(0, i - window_size)
            end = min(len(word_sequence), i + window_size + 1)
            context = word_sequence[start:i]
            context = np.append(context, word_sequence[i+1:end])
            distances = np.abs(np.arange(start, end)[np.arange(start, end) != i] - i)
            if len(context) > 0:
                X[center, context] += 1.0 / distances
    print(f"Co-occurrence matrix built. Non-zero entries: {np.count_nonzero(X)}")
    print(f"Max co-occurrence value: {np.max(X):.4f}")
    X = (X + X.T) / 2
    if norm == 'log':
        X_log = np.log1p(X)
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