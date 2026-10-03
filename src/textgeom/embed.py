"""Contextual token embeddings from XLM-RoBERTa-base (multilingual) (needs the `embed` extra)."""
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "xlm-roberta-base", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def token_ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def embed_ids(self, ids: list[int]) -> np.ndarray:
        """Embeddings (len(ids), 768) for one chunk, special tokens dropped."""
        t = self.torch
        full = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]
        with t.no_grad():
            out = self.model(input_ids=t.tensor([full], device=self.device))
        return out.last_hidden_state[0, 1:-1].cpu().numpy()
