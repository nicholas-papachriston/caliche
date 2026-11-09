"""
LangChain Embedding Provider Adapter for Caliche.
"""

from __future__ import annotations

import asyncio
from typing import Any


class LangChainEmbeddingProvider:
    """Adapter that wraps a LangChain embedding model."""

    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings
        self._model_name = (
            getattr(embeddings, "model", None)
            or getattr(embeddings, "model_name", None)
            or "langchain-embeddings"
        )

    async def embed(self, request: dict[str, Any]) -> dict[str, Any]:
        request_obj = request if isinstance(request, dict) else {}
        texts = request_obj.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return {"embeddings": [], "model": self._model_name, "usage": None}

        try:
            if hasattr(self.embeddings, "aembed_documents"):
                embeddings_list = await self.embeddings.aembed_documents(texts)
            elif hasattr(self.embeddings, "embed_documents"):
                loop = asyncio.get_event_loop()
                embeddings_list = await loop.run_in_executor(
                    None, lambda: self.embeddings.embed_documents(texts)
                )
            else:
                raise ValueError("Embeddings model does not support document embedding")
        except Exception as exc:
            raise RuntimeError(f"LangChain embedding failed: {exc}") from exc

        embeddings = []
        for emb in embeddings_list:
            if isinstance(emb, list):
                embeddings.append([float(x) for x in emb])
            else:
                embeddings.append([float(x) for x in list(emb)])

        total_chars = sum(len(text) for text in texts)
        estimated_tokens = max(1, total_chars // 4)

        return {
            "embeddings": embeddings,
            "model": request_obj.get("model") or self._model_name,
            "usage": {
                "prompt_tokens": estimated_tokens,
                "completion_tokens": 0,
                "total_tokens": estimated_tokens,
                "embedding_tokens": estimated_tokens,
            },
        }
