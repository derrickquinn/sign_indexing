"""High-level indexing classes used for similarity search.

This module exposes two primary classes:

``IndexSC``
    Builds an index for sign-concordance filtering (SCF) from input vectors and allows fast
    candidate filtering based on Hamming distance and thresholding.

``IndexRR``
    Re-ranks filtered candidates using full precision dot-product similarity.

Both classes are lightweight wrappers around Numba accelerated kernels defined
in :mod:`sign_indexing.sc_kernels`.
"""

import math
import pickle
import ctypes
import time
import code
import numpy as np
import numba
import h5py
from .sc_kernels import xor_blocked, count_bits, topk_over_ids

# Wrapper around numba kernel
class IndexRR:
    """Lightweight reranking index.

    The ``IndexRR`` class stores the full precision vectors and uses the
    :func:`~sign_indexing.sc_kernels.topk_over_ids` Numba kernel to compute the
    top-k most similar vectors for a set of filtered candidate IDs.
    """

    def __init__(self) -> None:
        """Create an empty reranking index."""
        pass

    def add(self, xb: np.ndarray):
        """Register the base vectors to be searched.

        Parameters
        ----------
        xb:
            2‑D array of shape ``(n_vectors, dim)`` containing the dataset in
            full precision.
        """
        self.xb = xb

    def search(self, xq: np.ndarray, k: int, ids: np.ndarray):
        """Re-rank candidates for each query vector.

        Parameters
        ----------
        xq:
            Query vectors of shape ``(n_queries, dim)``.
        k:
            Number of results to return.
        ids:
            Candidate indices produced by :class:`IndexSC`.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_queries, k)`` containing the top indices,
            sorted best-to-worst.
        """
        return topk_over_ids(xq, self.xb, ids, k)

# Sign-concordance index used for filtering
class IndexSC:
    """Sign-concordance filter used to preselect candidates.

    Parameters
    ----------
    file:
        Optional path to a previously saved index to load.
    transform:
        Optional matrix used to project input vectors before indexing.
    """

    def __init__(self, file=None, transform=False):
        self.transform = None
        self.v_signs = None
        self.threshold = 0
        self.ntotal = 0
        self.d = None
        self.block_size = 0
        self.blocks = None
        if file is not None:
            self.read_index(file, transform)

    def set_v_signs(self, v_signs: np.ndarray, d: int, num_blocks: int = 32) -> None:
        """Directly set the packed sign matrix for the index.

        This helper bypasses the normal ``add`` routine and is mainly used when
        loading a pre-computed index from disk.

        Parameters
        ----------
        v_signs:
            Packed bit matrix containing the signs of the base vectors.
        d:
            Dimensionality of the original vectors.
        num_blocks:
            Number of blocks to partition the sign matrix into for search.
        """
        self.v_signs = v_signs
        self.ntotal = v_signs.shape[0]
        self.d = d
        self._block(num_blocks)


    def _block(self, num_blocks: int = 16) -> None:
        """Partition the packed sign matrix into blocks for search."""
        n = self.v_signs.shape[0]
        packed_per_entry = self.v_signs.shape[1]

        self.block_size = math.ceil(n / num_blocks)

        remainder = n % self.block_size
        if remainder:
            pad = self.block_size - remainder
            packed = np.pad(self.v_signs, ((0, pad), (0, 0)), mode='constant', constant_values=False)

        else:
            packed = self.v_signs

        num_blocks = math.ceil(packed.shape[0] / self.block_size)

        self.blocks = np.reshape(packed, (num_blocks, self.block_size, packed_per_entry))


    def add(self, xb: np.ndarray, num_blocks: int = 16) -> None:
        """Add base vectors to the filter.

        Parameters
        ----------
        xb:
            Array of shape ``(n_vectors, dim)`` containing the dataset.
        num_blocks:
            Number of blocks to partition the index into.
        """

        signs = np.where(xb > 0, 1, 0).astype(np.int8)
        signs = signs.reshape(-1, signs.shape[-1]).astype(bool)
        self.ntotal = signs.shape[0]

        self.d = signs.shape[1]
        packed_int8 = np.packbits(signs, axis=1)

        n, d = signs.shape

        # Convert to 64-bit integers
        bits_per_packed = 64
        scaling_factor = bits_per_packed // 8
        packed_per_entry = math.ceil(d / bits_per_packed) 
        ints_needed = packed_per_entry * scaling_factor

        # Pad to correct number of entries for int64
        packed_int8 = np.pad(packed_int8, ((0, 0), (0, ints_needed - packed_int8.shape[1])),
                             mode='constant', constant_values=0)

        # Reshape to correct format
        packed_int8 = np.reshape(packed_int8, (n, packed_per_entry, scaling_factor))
        self.v_signs = packed_int8.view(np.int64).reshape(n, packed_per_entry)

        self._block(num_blocks)


    def search(
        self,
        xq: np.ndarray,
        threshold: int | None = None,
    ) -> np.ndarray:
        """Return IDs of vectors passing the sign-concordance filter.

        Parameters
        ----------
        xq:
            Query vectors of shape ``(n_queries, dim)``.
        threshold:
            Minimum number of matching sign bits required. If ``None`` the
            instance's ``threshold`` attribute is used.

        Returns
        -------
        np.ndarray
            1‑D array of candidate IDs.
        """

        d = self.d

        nq = xq.shape[0]

        q_array = np.array(xq)
        q_signs = np.where(q_array > 0, 1, 0).astype(np.int64)

        q_packed = np.packbits(q_signs, axis=1)

        bits_per_packed = 64
        scaling_factor = bits_per_packed // 8

        packed_per_entry = math.ceil(d / bits_per_packed)
        ints_needed = packed_per_entry * scaling_factor


        q_packed = np.pad(q_packed, ((0, 0), (0, ints_needed - q_packed.shape[1])),
                          mode='constant', constant_values=0)

        q_packed = np.reshape(q_packed, (nq, packed_per_entry, scaling_factor))

        q_packed = q_packed.view(np.int64).reshape(nq, packed_per_entry)

        if threshold is None:
            threshold = self.threshold

        hamming = xor_blocked(self.blocks, q_packed, np.int16(d-threshold))
        mask = hamming

        mask = mask[mask < self.ntotal]

        return mask



    def read_index(self, filename: str) -> None:
        """Load a previously saved index from ``filename``."""
        path = f"{filename}"

        path += ".pkl"
        with open(path, "rb") as pkl_file:
            read = pickle.load(pkl_file)
            self.v_signs = read["v_signs"]
            self.d = read["d"]
            self.ntotal = read["ntotal"]
            self._block()
        print("Read index")


    def write_index(self, filename: str) -> None:
        """Serialize the current index to ``filename``."""
        path = f"{filename}"

        path += ".pkl"

        with open(path, "wb") as pkl_file:
            out = {"v_signs": self.v_signs, "d": self.d, "ntotal": self.ntotal}
            pickle.dump(out, pkl_file)
