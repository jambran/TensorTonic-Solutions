import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    padded_seqs = []
    for seq in seqs:
        num_pads = max_len - len(seq)
        padded_seq = seq + [pad_value] * num_pads
        padded_seqs.append(padded_seq[:max_len])
        
    return padded_seqs