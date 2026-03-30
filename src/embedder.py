import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from src.chunker import TextChunk
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

class VectorStore:

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index: faiss.Index = None
        self.chunks: List[TextChunk] = []

    def build(self, chunks: List[TextChunk], batch_size: int=64) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc='Embedding chunks'):
            batch = texts[i:i + batch_size]
            embs = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.append(embs)
        embeddings = np.vstack(all_embeddings).astype('float32')
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f'[embedder] Built FAISS index: {len(chunks)} chunks | dim={dim}')

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'index': faiss.serialize_index(self.index)}, f)
        print(f'[embedder] Saved vector store to {path}')

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.index = faiss.deserialize_index(data['index'])
        print(f'[embedder] Loaded vector store: {len(self.chunks)} chunks')

    def search(self, query: str, top_k: int=5) -> List[Tuple[TextChunk, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for j, i in enumerate(indices[0]):
            if i >= 0:
                results.append((self.chunks[i], float(scores[0][j])))
        return results