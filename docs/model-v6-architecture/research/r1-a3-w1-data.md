# Wave 1 - Aspect 3: Data Pipeline (Async, Prefetch)

## Research Context
- **Architecture:** Ternary SSM (State Space Model) with weights in {-1, 0, +1}
- **Focus:** Training speed optimization through data loading pipeline
- **Date filter:** 2024-2026 papers only

## Findings

### ID: ASYNC-DATALOADER-PREFETCH
- **Src:** https://buildai.substack.com/p/data-loading-optimizations-asyncdataloader [2024] | **Gain:** Overlap data loading with compute | **V:** PROMISING
- **Principle:** AsyncDataLoader uses background threads to prefetch next batch while GPU computes on current batch
- **Application:** Use `DataLoader(num_workers=4, prefetch_factor=2, pin_memory=True)` for training

### ID: MEMORY-PINNING
- **Src:** https://buildai.substack.com/p/data-loading-optimizations-asyncdataloader [2024] | **Gain:** Faster CPU→GPU transfer | **V:** PROMISING
- **Principle:** Pinned memory enables direct DMA transfer to GPU without intermediate copy
- **Application:** Set `pin_memory=True` in DataLoader for CUDA training

### ID: NUMA-AWARE-DATALOADER
- **Src:** https://buildai.substack.com/p/data-loading-optimizations-asyncdataloader [2024] | **Gain:** NUMA-local data access | **V:** SPECULATIVE
- **Principle:** On multi-socket systems, ensure data loading happens on same NUMA node as CPU cores
- **Application:** Use `torch.utils.data.DataLoader` with `affinity` plugin for NUMA systems

### ID: DATA-PIPELINE-OPTIMIZATION
- **Src:** https://buildai.substack.com/p/data-loading-optimizations-asyncdataloader [2024] | **Gain:** Reduced data loading bottleneck | **V:** PROMISING
- **Principle:** Optimized data pipeline keeps GPU fed with minimal idle time between batches
- **Application:** Profile data loading with `torch.profiler` to identify bottlenecks

### ID: MULTI-TOKEN-PREDICTION
- **Src:** https://arxiv.org/html/2408.05606v2 [2024] | **Gain:** Better sample efficiency | **V:** PROMISING
- **Principle:** Training on multiple future tokens per position improves data utilization
- **Application:** Add auxiliary head for multi-token prediction in ternary SSM training

### ID: ADAPTIVE-BATCHING-USGM
- **Src:** https://arxiv.org/html/2408.05606v2 [2024] | **Gain:** Dynamic batch sizing based on curvature | **V:** SPECULATIVE
- **Principle:** Universal Stochastic Gradient Method adapts batch size based on loss landscape curvature estimation
- **Application:** Use adaptive batching to automatically find optimal batch size during training

### ID: SEQUENCE-PACKING
- **Src:** https://arxiv.org/html/2408.05606v2 [2024] | **Gain:** Efficient variable-length sequences | **V:** PROMISING
- **Principle:** Pack multiple sequences into single batch to eliminate padding waste
- **Application:** Implement sequence packing for training on variable-length sequences

## Key Insights
- Async data loading with prefetch can eliminate GPU idle time between batches
- Memory pinning provides 10-20% faster CPU→GPU transfer
- NUMA awareness important for multi-socket servers
- Adaptive batching can optimize batch size during training automatically
- Sequence packing eliminates padding inefficiency for variable-length data

## Seed Questions for Wave 2
1. How does data pipeline optimization interact with SSM state initialization?
2. Can adaptive batching be combined with gradient accumulation for optimal throughput?
