from __future__ import annotations
from typing import List
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dims: int = 1536):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dims = dims

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.dims),
        )
        return [e.values for e in res.embeddings]

    def embed_query(self, text: str) -> List[float]:
        res = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.dims),
        )
        return res.embeddings[0].values
