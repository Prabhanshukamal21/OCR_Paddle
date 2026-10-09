from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingService:

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL
    ):

        self.model_name = model_name

        self.model = self._load_model(
            model_name
        )

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(
        model_name: str
    ):

        return SentenceTransformer(
            model_name
        )

    def embed_text(
        self,
        text: str
    ) -> list[float]:

        if not text or not text.strip():
            raise ValueError(
                "Cannot create an embedding from empty text."
            )

        vector = self.model.encode(
            text.strip(),
            normalize_embeddings=True
        )

        return vector.tolist()

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        cleaned = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned:
            return []

        vectors = self.model.encode(
            cleaned,
            normalize_embeddings=True
        )

        return [
            vector.tolist()
            for vector in vectors
        ]