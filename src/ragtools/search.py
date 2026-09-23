import itertools
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import faiss
import numpy as np
from scipy.sparse import csr_matrix


class Index[K: Hashable, Q](Protocol):
    def add(self, key: K, point: Q) -> None: ...
    def drop(self, key: K) -> None: ...
    def query(self, point: Q, count: int) -> list[tuple[K, float]]: ...


@dataclass(slots=True)
class DenseIndex[K: Hashable]:
    """Dense Euclidean / L2 vector metric index using FAISS."""

    dimensions: int
    _index: faiss.IndexIDMap2 = field(init=False)
    _key_to_id: dict[K, int] = field(init=False, default_factory=dict)
    _id_to_key: dict[int, K] = field(init=False, default_factory=dict)
    _id_counter: itertools.count[int] = field(
        init=False, default_factory=itertools.count
    )

    def __post_init__(self) -> None:
        self._index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dimensions))

    def add(self, key: K, point: Sequence[float]) -> None:
        self.drop(key)

        vector = np.ascontiguousarray([point], dtype=np.float32)
        if vector.shape[1] != self.dimensions:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimensions}, got {vector.shape[1]}"
            )

        internal_id = next(self._id_counter)
        self._index.add_with_ids(vector, np.array([internal_id], dtype=np.int64))
        self._key_to_id[key] = internal_id
        self._id_to_key[internal_id] = key

    def drop(self, key: K) -> None:
        if key not in self._key_to_id:
            return
        internal_id = self._key_to_id.pop(key)
        del self._id_to_key[internal_id]

        ids = np.array([internal_id], dtype=np.int64)
        self._index.remove_ids(faiss.IDSelectorBatch(len(ids), faiss.swig_ptr(ids)))

    def query(self, point: Sequence[float], count: int) -> list[tuple[K, float]]:
        if count <= 0 or self._index.ntotal == 0:
            return []

        vector = np.ascontiguousarray([point], dtype=np.float32)
        if vector.shape[1] != self.dimensions:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimensions}, got {vector.shape[1]}"
            )

        k_search = min(count, self._index.ntotal)
        distances, ids = self._index.search(vector, k_search)

        results: list[tuple[K, float]] = []
        for internal_id, distance in zip(ids[0], distances[0], strict=True):
            if internal_id != -1:
                results.append((self._id_to_key[int(internal_id)], float(distance)))

        return results

    def __len__(self) -> int:
        return self._index.ntotal

    def __contains__(self, key: K) -> bool:
        return key in self._key_to_id


@dataclass(slots=True)
class SparseIndex[K: Hashable]:
    _data: np.ndarray = field(
        init=False, default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    _indices: np.ndarray = field(
        init=False, default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    _indptr: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros(1, dtype=np.int64)
    )
    _rows: list[K] = field(init=False, default_factory=list)
    _row_of: dict[K, int] = field(init=False, default_factory=dict)
    _vocab: dict[str, int] = field(init=False, default_factory=dict)
    _tokens: list[str] = field(init=False, default_factory=list)

    def add(self, key: K, point: dict[str, float]) -> None:
        self.drop(key)

        clean = {t: w for t, w in point.items() if w != 0.0}
        cols: list[int] = []
        for token in clean:
            if token not in self._vocab:
                self._vocab[token] = len(self._tokens)
                self._tokens.append(token)
            cols.append(self._vocab[token])

        self._data = np.concatenate(
            [self._data, np.fromiter(clean.values(), np.float32, len(clean))]
        )
        self._indices = np.concatenate(
            [self._indices, np.fromiter(cols, np.int32, len(clean))]
        )
        self._indptr = np.append(self._indptr, len(self._data))
        self._row_of[key] = len(self._rows)
        self._rows.append(key)

    def drop(self, key: K) -> None:
        if (r := self._row_of.pop(key, None)) is None:
            return

        start = int(self._indptr[r])
        end = int(self._indptr[r + 1])
        self._data = np.delete(self._data, slice(start, end))
        self._indices = np.delete(self._indices, slice(start, end))

        counts = np.diff(self._indptr)
        keep = np.ones(counts.size, dtype=bool)
        keep[r] = False
        self._indptr = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts[keep], dtype=np.int64)]
        )

        del self._rows[r]
        self._row_of = {k: i for i, k in enumerate(self._rows)}

    def query(self, point: dict[str, float], count: int) -> list[tuple[K, float]]:
        if count <= 0 or len(self._rows) == 0 or len(self._vocab) == 0:
            return []

        q_cols: list[int] = []
        q_vals: list[float] = []
        for token, weight in point.items():
            if token in self._vocab and weight != 0.0:
                q_cols.append(self._vocab[token])
                q_vals.append(weight)

        if not q_cols:
            return []

        q = np.zeros(len(self._vocab), dtype=np.float32)
        q[q_cols] = q_vals

        matrix = csr_matrix(
            (self._data, self._indices, self._indptr),
            shape=(len(self._rows), len(self._vocab)),
        )
        scores = matrix @ q

        valid_indices = np.flatnonzero(scores > 0.0)
        if valid_indices.size == 0:
            return []

        k_eff = min(count, valid_indices.size)
        partitioned = valid_indices[
            np.argpartition(-scores[valid_indices], k_eff - 1)[:k_eff]
        ]
        top_indices = partitioned[np.argsort(-scores[partitioned], kind="stable")]

        return [(self._rows[idx], float(scores[idx])) for idx in top_indices]

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, key: K) -> bool:
        return key in self._row_of


def rerank[K](
    items: Sequence[K],
    scorer: Callable[[K], float],
    count: int,
) -> list[tuple[K, float]]:
    if count <= 0 or not items:
        return []
    scored = [(key, scorer(key)) for key in items]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:count]


def fuse_ranks[K: Hashable](
    rankings: Sequence[Sequence[K]],
    decay: float,
    count: int,
) -> list[tuple[K, float]]:
    if count <= 0 or not rankings:
        return []

    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            weight = 1.0 / (decay + rank)
            scores[key] = scores.get(key, 0.0) + weight

    sorted_results = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return sorted_results[:count]
