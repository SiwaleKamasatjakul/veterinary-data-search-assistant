"""Thin search facade kept for API compatibility with the original project."""

from __future__ import annotations

import logging
from typing import List

from tools.ImportDB2Faiss import VetFAISS

logger = logging.getLogger(__name__)


class VetDocumentSearch:
    @staticmethod
    def search_documents(query: str, top_k: int | None = None) -> List[dict]:
        results = VetFAISS.search_vet_doc(query, top_k=top_k)
        logger.info("Vet search %r -> %d hit(s)", query, len(results))
        return results
