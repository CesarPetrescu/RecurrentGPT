import os
import numpy as np
from openai import OpenAI
from utils import load_env

load_env()

api_key = os.getenv("ApiKey") or os.getenv("OPENAI_API_KEY")
api_base = os.getenv("ApiBase") or os.getenv("OPENAI_API_BASE")
embedding_model = os.getenv("EmbeddingModel", "text-embedding-ada-002")

class OpenAIEmbedder:
    def __init__(self, model=None):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model or embedding_model

    def encode(self, texts, convert_to_tensor=False):
        if isinstance(texts, str):
            texts = [texts]
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

def cosine_scores(vec, matrix):
    vec = np.array(vec)
    matrix = np.array(matrix)
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
    return matrix.dot(vec) / denom
