import builtins
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol

import faiss
import numpy as np
from scipy.sparse import csr_matrix

from ragtools.stores import Store


class Index[K, V](Store[K, V], Protocol):
    def closest(self, key: K, k: int) -> tuple[K]: ...


@dataclass(slots=True)
class FaissEmbeddingIndex:
    def __init__(self, dimensions: int) -> None:
        self._index = faiss.IndexIDMap2(faiss.IndexFlatL2(dimensions))
        self._ids: builtins.set[int] = builtins.set()

    def set(self, key: int, value: list[float]) -> None:
        self.delete(key)
        self._index.add_with_ids(
            np.ascontiguousarray([value], dtype=np.float32),
            np.array([key], dtype=np.int64),
        )
        self._ids.add(key)

    def get(self, key: int) -> list[float]:
        if key not in self._ids:
            raise KeyError(key)
        return self._index.reconstruct(key).tolist()

    def delete(self, key: int) -> None:
        if key in self._ids:
            ids = np.array([key], dtype=np.int64)
            self._index.remove_ids(faiss.IDSelectorBatch(len(ids), faiss.swig_ptr(ids)))
            self._ids.discard(key)

    def contains(self, key: int) -> bool:
        return key in self._ids

    def keys(self) -> builtins.set[int]:
        return builtins.set(self._ids)

    def closest(self, key: int, k: int) -> tuple[int, ...]:
        if k <= 0 or self._index.ntotal == 0:
            return ()
        if key not in self._ids:
            raise KeyError(key)
        vec = self._index.reconstruct(key)[None]
        _, ids = self._index.search(vec, min(k + 1, self._index.ntotal))
        return tuple(int(fid) for fid in ids[0] if fid != -1 and int(fid) != key)[:k]

    def __len__(self) -> int:
        return self._index.ntotal

    def __contains__(self, key: int) -> bool:
        return self.contains(key)


@dataclass(slots=True)
class SparseEmbeddingIndex[K: Hashable]:
    def __init__(self) -> None:
        self._data = np.empty(0, dtype=np.float32)
        self._indices = np.empty(0, dtype=np.int32)
        self._indptr = np.zeros(1, dtype=np.int64)
        self._rows: list[K] = []
        self._row_of: dict[K, int] = {}
        self._vocab: dict[str, int] = {}
        self._tokens: list[str] = []

    def set(self, key: K, value: dict[str, float]) -> None:
        self.delete(key)
        clean = {t: w for t, w in value.items() if w}
        cols = [self._vocab.setdefault(t, len(self._vocab)) for t in clean]
        self._tokens.extend(t for t in clean if self._vocab[t] == len(self._tokens))
        self._data = np.concatenate(
            [self._data, np.fromiter(clean.values(), np.float32, len(clean))]
        )
        self._indices = np.concatenate(
            [self._indices, np.fromiter(cols, np.int32, len(clean))]
        )
        self._indptr = np.append(self._indptr, len(self._data))
        self._row_of[key] = len(self._rows)
        self._rows.append(key)

    def get(self, key: K) -> dict[str, float]:
        r = self._row_of[key]
        start, end = int(self._indptr[r]), int(self._indptr[r + 1])
        return {
            self._tokens[c]: float(w)
            for c, w in zip(
                self._indices[start:end], self._data[start:end], strict=True
            )
        }

    def delete(self, key: K) -> None:
        if (r := self._row_of.pop(key, None)) is None:
            return
        start, end = int(self._indptr[r]), int(self._indptr[r + 1])
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

    def contains(self, key: K) -> bool:
        return key in self._row_of

    def keys(self) -> builtins.set[K]:
        return builtins.set(self._row_of)

    def closest(self, key: K, k: int) -> tuple[K, ...]:
        if key not in self._row_of:
            raise KeyError(key)
        if k <= 0 or len(self._rows) <= 1:
            return ()
        r = self._row_of[key]
        start, end = int(self._indptr[r]), int(self._indptr[r + 1])
        q = np.zeros(len(self._vocab), dtype=np.float32)
        q[self._indices[start:end]] = self._data[start:end]
        scores = self._matrix @ q
        scores[r] = -np.inf
        scores[scores == 0.0] = -np.inf
        k_eff = min(k, len(self._rows) - 1)
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top], kind="stable")]
        return tuple(self._rows[i] for i in top if scores[i] != -np.inf)

    @property
    def _matrix(self) -> csr_matrix:
        return csr_matrix(
            (self._data, self._indices, self._indptr),
            shape=(len(self._rows), max(len(self._vocab), 1)),
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, key: K) -> bool:
        return self.contains(key)
