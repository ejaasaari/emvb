"""
EMVB: Efficient Multi-Vector Retrieval with Bit Vectors

Python wrapper for the EMVB C++ library providing high-performance
multi-vector dense retrieval using bit vectors and Product Quantization.
"""

import os
import numpy as np
from typing import List, Tuple

try:
    from emvb import (
        DocumentScorer,
        search as _search,
        load_qids,
        set_num_threads
    )
except ImportError as e:
    raise ImportError(
        "Could not import EMVB C++ extension. "
        "Make sure the module is built and in your Python path. "
        f"Error: {e}"
    )

os.environ.setdefault('OMP_NUM_THREADS', '1')
set_num_threads(1)


class EMVBSearcher:
    """
    High-level Python interface for EMVB search.

    Parameters
    ----------
    index_dir_path : str
        Path to the directory containing index files (centroids.npy, residuals.npy, etc.)
    doclens_path : str
        Path to the .npy file containing document lengths
    max_query_terms : int
        Maximum number of terms per query (e.g., 32)

    Attributes
    ----------
    scorer : DocumentScorer
        The underlying C++ DocumentScorer instance
    """

    def __init__(
        self,
        index_dir_path: str,
        doclens_path: str,
        max_query_terms: int
    ):
        self.index_dir_path = index_dir_path
        self.doclens_path = doclens_path
        self.max_query_terms = max_query_terms
        self.scorer = DocumentScorer(doclens_path, index_dir_path, max_query_terms)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int = 2,
        thresh: float = 0.4,
        thresh_query: float = 0.4,
        out_second_stage: int = 128,
        n_doc_to_score: int = 500
    ) -> List[Tuple[int, float]]:
        """
        Perform search on a single query.

        Parameters
        ----------
        query : np.ndarray
            Query embedding with shape (vec_per_query, dim)
        k : int, default=10
            Number of top results to return
        nprobe : int, default=2
            Number of inverted lists to probe per query term
        thresh : float, default=0.4
            Centroid filtering threshold
        thresh_query : float, default=0.4
            Query term threshold for Phase 4
        out_second_stage : int, default=128
            Number of documents advancing from Phase 3 to Phase 4
        n_doc_to_score : int, default=500
            Number of documents advancing from Phase 2 to Phase 3

        Returns
        -------
        List[Tuple[int, float]]
            List of (doc_id, score) tuples for top-k results
        """
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        if query.ndim != 2:
            raise ValueError("query must be 2-dimensional: (vec_per_query, dim)")

        results = _search(
            self.scorer,
            query,
            k,
            nprobe,
            thresh,
            thresh_query,
            out_second_stage,
            n_doc_to_score
        )

        return results

    def load_queries(self, queries_path: str) -> np.ndarray:
        """
        Load queries from a .npy file.

        Parameters
        ----------
        queries_path : str
            Path to the queries .npy file

        Returns
        -------
        np.ndarray
            Queries array with shape (n_queries, vec_per_query, dim)
        """
        queries = np.load(queries_path)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        return queries

    @property
    def num_vectors_scored(self) -> int:
        """Get the number of document vectors scored (globalCounter)."""
        return self.scorer.globalCounter

    def reset_counter(self):
        """Reset the globalCounter to 0."""
        self.scorer.globalCounter = 0


__all__ = [
    'EMVBSearcher',
    'DocumentScorer',
    'search',
    'load_qids',
    'set_num_threads'
]
