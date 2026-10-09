import hashlib
from pathlib import Path
from typing import Any

import chromadb


class ChromaService:

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_service
    ):

        self.persist_directory = str(
            Path(persist_directory)
        )

        Path(
            self.persist_directory
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        self.embedding_service = (
            embedding_service
        )

        self.client = (
            chromadb.PersistentClient(
                path=self.persist_directory
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

    @staticmethod
    def _make_id(
        text: str,
        source: str,
        chunk_index: int = 0
    ) -> str:

        seed = (
            f"{source}|"
            f"{chunk_index}|"
            f"{text}"
        )

        return hashlib.sha256(
            seed.encode("utf-8")
        ).hexdigest()

    def add_text(
        self,
        text: str,
        source: str,
        metadata: dict[str, Any] | None = None
    ) -> dict:

        clean_text = text.strip()

        if not clean_text:
            raise ValueError(
                "Cannot store empty text."
            )

        metadata = dict(
            metadata or {}
        )

        metadata["source"] = (
            source or "unknown"
        )

        record_id = self._make_id(
            clean_text,
            source or "unknown",
            0
        )

        embedding = (
            self.embedding_service
            .embed_text(clean_text)
        )

        self.collection.upsert(
            ids=[record_id],
            documents=[clean_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        return {
            "id": record_id,
            "embedding_dimensions": len(
                embedding
            )
        }

    def add_chunks(
        self,
        chunks: list[str],
        source: str,
        base_metadata: dict[str, Any] | None = None
    ) -> dict:

        clean_chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk and chunk.strip()
        ]

        if not clean_chunks:
            raise ValueError(
                "No valid chunks to store."
            )

        base_metadata = dict(
            base_metadata or {}
        )

        embeddings = (
            self.embedding_service
            .embed_texts(clean_chunks)
        )

        ids = []
        metadatas = []

        for index, chunk in enumerate(
            clean_chunks
        ):

            record_id = self._make_id(
                chunk,
                source or "unknown",
                index
            )

            metadata = dict(
                base_metadata
            )

            metadata["source"] = (
                source or "unknown"
            )

            metadata["chunk_index"] = index

            ids.append(record_id)
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=clean_chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return {
            "count": len(clean_chunks),
            "embedding_dimensions": (
                len(embeddings[0])
                if embeddings
                else 0
            )
        }

    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        count = self.collection.count()

        if count == 0:
            return []

        query_embedding = (
            self.embedding_service
            .embed_text(query)
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=min(
                max(1, n_results),
                count
            ),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        ids = (
            results.get("ids") or [[]]
        )[0]

        docs = (
            results.get("documents") or [[]]
        )[0]

        metadatas = (
            results.get("metadatas") or [[]]
        )[0]

        distances = (
            results.get("distances") or [[]]
        )[0]

        matches = []

        for i, record_id in enumerate(ids):

            metadata = (
                metadatas[i]
                if i < len(metadatas)
                and metadatas[i]
                else {}
            )

            matches.append(
                {
                    "id": record_id,
                    "text": (
                        docs[i]
                        if i < len(docs)
                        else ""
                    ),
                    "source": metadata.get(
                        "source",
                        "unknown"
                    ),
                    "metadata": metadata,
                    "distance": (
                        distances[i]
                        if i < len(distances)
                        else None
                    )
                }
            )

        return matches

    def list_records(
        self,
        limit: int = 100
    ) -> list[dict]:

        count = self.collection.count()

        if count == 0:
            return []

        result = self.collection.get(
            limit=min(
                max(1, limit),
                count
            ),
            include=[
                "documents",
                "metadatas"
            ]
        )

        ids = result.get(
            "ids",
            []
        )

        docs = result.get(
            "documents",
            []
        ) or []

        metadatas = result.get(
            "metadatas",
            []
        ) or []

        records = []

        for i, record_id in enumerate(ids):

            metadata = (
                metadatas[i]
                if i < len(metadatas)
                and metadatas[i]
                else {}
            )

            records.append(
                {
                    "id": record_id,
                    "text": (
                        docs[i]
                        if i < len(docs)
                        else ""
                    ),
                    "source": metadata.get(
                        "source",
                        "unknown"
                    ),
                    "metadata": metadata
                }
            )

        return records