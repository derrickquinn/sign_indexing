# Sign Indexing for Efficient Similarity Search

This repository implements an efficient similarity search system using Sign Concordance Filtering (SCF). It's designed for high-dimensional vector similarity search. Documentation is WIP.

## Components

- **IndexSC**: Performs fast, sign-based filtering of candidate vectors
- **IndexRR**: Performs filtered reranking of candidate vectors

Both IndexSC and IndexRR are implemented in numba kernels to improve performance. Futher improvements are almost certainly possible, but the current implementation is dramatically faster than naïve methods. 

These components have been validated on x86 CPUs.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  
   ```

2. Install the required packages:
   ```bash
   pip install .
   ```

### Running the Example

An example script is provided in `run_sc.py` that demonstrates the full pipeline. `run_sc.py` loads from a (hardcoded) hdf5 file and runs the pipeline. 

```bash
wget http://ann-benchmarks.com/glove-100-angular.hdf5
python run_sc.py

# Output on Intel Xeon Max 9462:
SCF: 1.27 ms
RR: 0.41 ms
Total: 1.69 ms
QPS: 592.30
Recall@10: 0.974
Filter ratio: 15.09

SCF: 9.61 ms
RR: 6.64 ms
Total: 16.26 ms
QPS: 984.31
Recall@10: 0.978
Filter ratio: 2.12
```



## Configuration

The system can be configured with the following parameters:

- `threshold`: The number of sign bits that must match (lower values are looser)
- `batch_size`: Number of queries to process in parallel


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

