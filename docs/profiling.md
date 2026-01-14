## Profiling 

Training was profiled using Python cProfile on the PyTorch Lightning pipeline.

Key observations:
- Most runtime is spent inside the Lightning training epoch loop.
- Data loading via torch.utils.data.DataLoader dominates execution time.
- Optimization and backward passes contribute less relative overhead.

This indicates the training pipeline is data-bound rather than compute-bound.
Future optimization directions include increasing DataLoader workers,
caching preprocessed data, and tuning batch size.
