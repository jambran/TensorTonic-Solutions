import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"


    def _preprocess(self, text: str):
        return [token.lower() for token in text.split()]

    def _add_token_to_vocab(self, token: str):
        self.word_to_id[token] = self.vocab_size
        self.id_to_word[self.vocab_size] = token
        self.vocab_size += 1
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # add special tokens to vocab
        for special_token in [
            self.pad_token, 
            self.unk_token, 
            self.bos_token, 
            self.eos_token,
        ]:
            self._add_token_to_vocab(special_token)

        token_set = set(
            [
                token
                for text in texts
                for token in self._preprocess(text)
            ]
        )
        for token in token_set:
            if token not in self.word_to_id:
                self._add_token_to_vocab(token)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = self._preprocess(text)
        token_ids = [
            self.word_to_id.get(token, self.word_to_id[self.unk_token])
            for token in tokens
        ]
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join([
            self.id_to_word[id_]
            for id_ in ids
        ])
        