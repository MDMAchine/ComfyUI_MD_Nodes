import sentencepiece as spm
import os

class SPieceTokenizer:
    def __init__(self, model_path, **kwargs):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SentencePiece model not found at {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def encode(self, text):
        return self.sp.encode_as_ids(text)
    
    def decode(self, tokens):
        return self.sp.decode_ids(tokens)
    
    def get_vocab(self):
        """Return vocabulary as a dict mapping tokens to IDs"""
        vocab = {}
        for i in range(self.sp.get_piece_size()):
            vocab[self.sp.id_to_piece(i)] = i
        return vocab
    
    @staticmethod
    def from_pretrained(path, **kwargs):
        # This method is often expected by ComfyUI's loader
        return SPieceTokenizer(path, **kwargs)
    
    def __call__(self, text, **kwargs):
        # Process text and return in the expected dictionary format
        tokens = self.encode(text)
        return {"input_ids": tokens}
    
    def serialize_model(self):
        # Return the serialized model proto
        return self.sp.serialized_model_proto()