"""Low level Numba kernels used by :mod:`sign_indexing`.

The functions defined here implement the performance critical pieces of the
sign‑concordance filter and reranking stages.  They are written with Numba so
that they JIT compile to efficient native code when first executed.
"""

import numba
from numba.extending import intrinsic
from numba import njit, prange, get_num_threads, get_thread_id

import numpy as np


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@intrinsic
def popcnt64(typingctx, x):
    """Return the number of set bits in ``x`` using LLVM's ``ctpop`` intrinsic."""
    sig = x(x)
    def codegen(context, builder, signature, args):
        [a] = args

        ctpop = builder.module.declare_intrinsic("llvm.ctpop", [a.type])

        res = builder.call(ctpop, [a])

        return res
    return sig, codegen

@numba.njit(inline="always", fastmath=True, cache=True)
def count_bits(val):
    """Return the population count of ``val`` as an int8."""
    return np.int8(popcnt64(val))

@numba.njit(parallel=True, fastmath=True, cache=True)
def xor_blocked(block, q_signs, dot_threshold):
    """Filter candidates by Hamming distance.

    Parameters
    ----------
    block:
        3‑D array of packed sign bits representing the indexed vectors,
        partitioned into blocks.
    q_signs:
        Packed sign representation of the query vectors.
    dot_threshold:
        Maximum Hamming distance allowed (``d - threshold`` in the
        high‑level API).

    Returns
    -------
    np.ndarray
        1‑D array containing the IDs of vectors that satisfy the constraint.
    """
    b, m, k = block.shape
    n = q_signs.shape[0]
    joined = numba.typed.List.empty_list(np.int32[:])
    for i in range(b):
        joined.append(np.empty(0, dtype=np.int32))

    for b in numba.prange(b):
        local_list = numba.typed.List.empty_list(np.int32)

        sub = block[b]
        for i in range(m):
            all_xor = sub[i] ^ q_signs
            for j in range(n):
                s = np.int16(0)
                q_xor = all_xor[j]
                for p in range(k):
                    s += count_bits(q_xor[p])


                if s <= dot_threshold:
                    local_list.append(np.int32(i + b*m))
                    break

        local_arr = np.empty(len(local_list), dtype=np.int32)

        for i, c in enumerate(local_list):
            local_arr[i] = c

        joined[b] = local_arr

    total = 0
    for x in joined:
        total += x.shape[0]

    out = np.empty(total, dtype=np.int32)

    pos = 0

    for i, j in enumerate(joined):
        x = j
        l = x.shape[0]
        out[pos:pos+l] = x

        pos += l
    return out


### This method is used for reranking filtered results
@njit(parallel=True, fastmath=True)
def topk_over_ids(xq, xb, ids, k):
    """Compute top‑k dot products over a subset of IDs."""
    nq, d       = xq.shape
    n_threads   = get_num_threads()

    # thread-local top-k buffers: (threads, nq, k)
    topv_tls = np.full((n_threads, nq, k), -1e38, np.float32)
    topi_tls = np.full((n_threads, nq, k), -1,     np.int64)

    id_blocks = np.array_split(ids, n_threads)

    # -------- pass 1: each thread scans its chunk of ids --------
    for b in prange(len(id_blocks)):                     # parallelised over IDs
        tid = get_thread_id()

        idxs = id_blocks[b]
        for idx in idxs:
            for qi in range(nq):
                s = 0.0
                for dim in range(d):
                    s += xq[qi, dim] * xb[idx, dim]

                # insert into this thread’s top-k for query qi
                if s > topv_tls[tid, qi, -1]:
                    j = k - 1
                    while j > 0 and s > topv_tls[tid, qi, j-1]:
                        topv_tls[tid, qi, j] = topv_tls[tid, qi, j-1]
                        topi_tls[tid, qi, j] = topi_tls[tid, qi, j-1]
                        j -= 1
                    topv_tls[tid, qi, j] = s
                    topi_tls[tid, qi, j] = idx

    # -------- pass 2: merge per-thread buffers --------
    topv = np.full((nq, k), -1e38, np.float32)
    topi = np.full((nq, k), -1,     np.int64)

    for tid in range(n_threads):
        for qi in range(nq):
            for j in range(k):
                s   = topv_tls[tid, qi, j]
                idx = topi_tls[tid, qi, j]
                if s > topv[qi, -1]:
                    t = k - 1
                    while t > 0 and s > topv[qi, t-1]:
                        topv[qi, t] = topv[qi, t-1]
                        topi[qi, t] = topi[qi, t-1]
                        t -= 1
                    topv[qi, t] = s
                    topi[qi, t] = idx

    return topi                # shape (nq, k), best-to-worst