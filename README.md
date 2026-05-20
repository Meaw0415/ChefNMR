# ChefNMR Offline Export

Standalone scripts for running ChefNMR in a dedicated `nmr3d` environment and
exporting NMRAgnet-compatible offline cache files.

This repo is intentionally lightweight:
- It does not vendor the full ChefNMR codebase
- It expects an existing local clone of `chefnmr`
- It writes output in the cache format consumed by NMRAgnet's
  `nmr_chefnmr_offline_denovo_tool`

## Output Format

The exporter writes a pickle cache with records like:

```python
cache["uspto:0"] = {
    "dataset": "uspto",
    "sample_idx": 0,
    "split": "test",
    "candidates": [
        {
            "smiles": "...",
            "score": 0.72,
            "source": "chefnmr_denovo",
            "rank": 1,
            "target_smiles": "...",
        },
    ],
    "metadata": {
        "dataset": "uspto",
        "sample_idx": 0,
        "actual_test_index": "524289",
        "ckpt_path": "...",
        "diffusion_samples": 10,
        "num_sampling_steps": 50,
        "gpu": "0",
        "multiplicity": 10,
    },
}
```

This format is directly readable by NMRAgnet's offline ChefNMR tool.

## Prerequisites

1. A working local `chefnmr` clone
2. The `nmr3d` conda environment from ChefNMR
3. Downloaded ChefNMR datasets and checkpoints

Expected environment variables:

```bash
export CHEFNMR_DIR=/absolute/path/to/chefnmr
export CHEFNMR_CKPT_DIR=/absolute/path/to/checkpoints
export CHEFNMR_CUDA_VISIBLE_DEVICES=0
```

## Quick Start

Run one `uspto` chunk and produce an NMRAgnet-compatible cache:

```bash
cd /path/to/chefnmr-offline-export
PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
  --datasets uspto \
  --split test \
  --start-idx 0 \
  --max-samples 64 \
  --chunk-samples 64 \
  --top-k 10 \
  --test-batch-size 64 \
  --batch-size 64 \
  --output /path/to/checkpoints/chefnmr_offline_cache_uspto.pkl \
  --overwrite
```

## Full Run Example

```bash
cd /path/to/chefnmr-offline-export
PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
  --datasets uspto \
  --split test \
  --start-idx 0 \
  --max-samples -1 \
  --chunk-samples 1024 \
  --top-k 10 \
  --test-batch-size 64 \
  --batch-size 64 \
  --output /path/to/checkpoints/chefnmr_offline_cache_uspto.pkl \
  --overwrite
```

## 4-GPU Sharded Run

This is the recommended scale-out pattern. Run one shard per GPU and merge
afterward.

Example split for `uspto` with current ChefNMR test size `75567`:

```bash
CHEFNMR_CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
  --datasets uspto --split test --start-idx 0 --max-samples 19000 \
  --chunk-samples 1024 --top-k 10 --test-batch-size 64 --batch-size 64 \
  --output /path/to/checkpoints/uspto_shard0.pkl --overwrite
```

```bash
CHEFNMR_CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
  --datasets uspto --split test --start-idx 19000 --max-samples 19000 \
  --chunk-samples 1024 --top-k 10 --test-batch-size 64 --batch-size 64 \
  --output /path/to/checkpoints/uspto_shard1.pkl --overwrite
```

```bash
CHEFNMR_CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
  --datasets uspto --split test --start-idx 38000 --max-samples 19000 \
  --chunk-samples 1024 --top-k 10 --test-batch-size 64 --batch-size 64 \
  --output /path/to/checkpoints/uspto_shard2.pkl --overwrite
```

```bash
CHEFNMR_CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
  --datasets uspto --split test --start-idx 57000 --max-samples 8567 \
  --chunk-samples 1024 --top-k 10 --test-batch-size 64 --batch-size 64 \
  --output /path/to/checkpoints/uspto_shard3.pkl --overwrite
```

Merge after all shards finish:

```bash
python merge_shards.py \
  --inputs /path/to/checkpoints/uspto_shard0.pkl \
           /path/to/checkpoints/uspto_shard1.pkl \
           /path/to/checkpoints/uspto_shard2.pkl \
           /path/to/checkpoints/uspto_shard3.pkl \
  --output /path/to/checkpoints/chefnmr_offline_cache_uspto.pkl
```

## Notes

- The exporter disables unnecessary chain visualization storage:
  - `test_args.visualize_samples=0`
  - `test_args.visualize_chains=0`
  - `visualization_args.n_chain_frames=1`
- That reduces memory overhead compared with the stock ChefNMR sampling path.
- ChefNMR's config includes `trainer.precision: bf16-mixed`, but this exporter
  uses a custom direct inference loop, so Lightning mixed precision is not
  automatically applied.
- Included helper files:
  - `run_uspto_example.sh`
  - `run_uspto_4gpu_example.sh`
  - `merge_shards.py`
