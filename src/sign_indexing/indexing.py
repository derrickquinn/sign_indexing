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
    def __init__(self):
        pass
    
    def add(self, xb: np.ndarray):
        self.xb = xb
    
    def search(self, xq: np.ndarray, k: int, ids: np.ndarray):
        return topk_over_ids(xq, self.xb, ids, k)

# Sign-concordance index used for filtering
class IndexSC:
    def __init__(self, file = None, transform=False, dim_out=None, zero_positive=False):
        self.transform = None
        self.v_signs = None
        self.threshold = 0
        self.ntotal = 0
        self.d = None
        self.dim_out = dim_out
        self.block_size = 0
        self.blocks = None
        if file is not None:
            self.read_index(file, transform, zero_positive)

    def set_v_signs(self, v_signs, d, num_blocks = 32):
        # print(v_signs.shape)
        self.v_signs = v_signs
        self.ntotal = v_signs.shape[0]
        self.d = d
        self._block(num_blocks)


    def _block(self, num_blocks = 16):
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


    def add(self, xb, zero_positive = False, num_blocks = 16):
        if self.dim_out is not None and self.dim_out > xb.shape[1]:
            xb = np.pad(xb, ((0, 0), (0, self.dim_out - xb.shape[1])),
                        mode='constant', constant_values=0)
        if self.transform is not None:
            xb = xb @ self.transform

        if zero_positive:
            signs = np.where(xb >= 0, 1, 0).astype(np.int8)
        else:
            signs = np.where(xb > 0, 1, 0).astype(np.int8)
        signs = signs.reshape(-1, signs.shape[-1]).astype(bool)
        self.ntotal = signs.shape[0]

        self.d = signs.shape[1]
        packed_int8 = np.packbits(signs, axis=1)

        n, d = signs.shape

        # Convert to 64-bit integers
        # Setup
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


    def search(self, xq, zero_positive = False, threshold = None):
        if self.dim_out is not None and self.dim_out > xq.shape[1]:
            padding = self.dim_out - xq.shape[1]
            xq = np.pad(xq, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        if self.transform is not None:

            xq = xq @ self.transform

        d = self.d

        nq = xq.shape[0]

        q_array = np.array(xq)
        if zero_positive:
            q_signs = np.where(q_array >= 0, 1, 0).astype(np.int64)
        else:
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



    def read_index(self, filename):
        path = f"{filename}"

        if self.dim_out is not None:
            path += f"_dim{self.dim_out}"

        path += ".pkl"
        with open(path, "rb") as pkl_file:
            read = pickle.load(pkl_file)
            self.v_signs = read["v_signs"]
            self.d = read["d"]
            self.ntotal = read["ntotal"]
            self._block()
        print("Read index")


    def write_index(self, filename):
        path = f"{filename}"

        path += ".pkl"

        with open(path, "wb") as pkl_file:
            out = {"v_signs": self.v_signs, "d": self.d, "ntotal": self.ntotal}
            pickle.dump(out, pkl_file)
