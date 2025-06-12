import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from sign_indexing import IndexSC, IndexRR
import h5py


def calculate_recall(ground_truth: list, results: list, k=32):
    correct = 0
    total = len(ground_truth) * k
    incl = set(results)

    for gt_row in ground_truth:
        correct += len(set(gt_row[:k]) & incl)
    return correct / total

    
def load_from_hdf5(path, dataset, ntotal=-1):
    with h5py.File(path, "r") as f:
        return f[dataset][:ntotal]

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def run_config(**kwargs):

    threshold = kwargs.get("threshold", 62)
    batch_size = kwargs.get("batch_size", 1)
    num_documents = kwargs.get("num_documents", 10)
    hdf5_path = kwargs.get("hdf5_path", "glove-100-angular.hdf5")
    num_queries = kwargs.get("num_queries", -1) # All queries

    queries_arr = load_from_hdf5(hdf5_path, "test", ntotal=num_queries)
    truth_arr = load_from_hdf5(hdf5_path, "neighbors", ntotal=num_queries)

    documents_arr = load_from_hdf5(hdf5_path, "train")
    
    # Need to normalize the documents for cosine similarity. Do this offline instead.
    documents_arr = normalize(documents_arr)

    index_rr = IndexRR()
    index_sc = IndexSC()

    index_rr.add(documents_arr)
    index_sc.add(documents_arr, num_blocks=16)

    index_sc.threshold = threshold


    query_loader = DataLoader(queries_arr, batch_size=batch_size)
    truth_loader = DataLoader(truth_arr, batch_size=batch_size)


    scf_times = []
    rr_times = []

    recalls = []
    indices_lens = []

    for i, (batch,t) in tqdm(enumerate(zip(query_loader, truth_loader)), total=len(query_loader)):
        s = time.time()
        indices = index_sc.search(batch.numpy())
        scf_times.append(time.time() - s)

        s = time.time()
        final_indices = index_rr.search(batch.numpy(), num_documents, ids = indices)
        rr_times.append(time.time() - s)

        # Used for computing filter ratio
        indices_lens.append(len(indices))

        # Flatten top IDs across all queries
        # This only works because we re-rank in full precision, and every query sees every ID.
        flattened_indices = final_indices.flatten()

        # Compute recall
        recall = calculate_recall(t.numpy(), flattened_indices, num_documents)
        recalls.append(recall)

    # Throw away warmups
    mean_scf = np.mean(scf_times[1:]) 
    mean_rr = np.mean(rr_times[1:])

    recall = np.mean(recalls)

    print(f"SCF: {mean_scf*1000:.2f} ms")
    print(f"RR: {mean_rr*1000:.2f} ms")
    print(f"Total: {(mean_scf + mean_rr)*1000:.2f} ms")
    print(f"QPS: {batch_size / (mean_scf + mean_rr):.2f}")
    print(f"Recall@{num_documents}: {recall:.3f}")
    print(f"Filter ratio: {index_sc.ntotal / np.mean(indices_lens):.2f}")

if __name__ == "__main__":

    common_params = {
        "hdf5_path": "glove-100-angular.hdf5",
        "num_documents": 10,
    }

    run_config(threshold=62, batch_size=1, **common_params)
    run_config(threshold=63, batch_size=16, **common_params)