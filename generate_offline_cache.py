#!/usr/bin/env python3
"""
Standalone ChefNMR offline cache exporter.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning.fabric.utilities.seed import seed_everything


CHEFNMR_DIR = Path(os.environ.get("CHEFNMR_DIR", ""))
if not CHEFNMR_DIR:
    raise EnvironmentError("Please set CHEFNMR_DIR to your local ChefNMR repo path.")
if str(CHEFNMR_DIR) not in sys.path:
    sys.path.insert(0, str(CHEFNMR_DIR))

CHEFNMR_CKPT_DIR = Path(os.environ.get("CHEFNMR_CKPT_DIR", ""))
if not CHEFNMR_CKPT_DIR:
    raise EnvironmentError("Please set CHEFNMR_CKPT_DIR to your checkpoint/output directory.")

CHEFNMR_GPU = os.environ.get("CHEFNMR_CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", CHEFNMR_GPU)
os.environ.setdefault("CHEFNMR_DIR", str(CHEFNMR_DIR))

DEFAULT_BENCH_ROOT = Path(
    os.environ.get("NMRBENCH_ROOT", "/hpc2ssd/JH_DATA/spooler/zfang723/NMR/NMRBench")
)

DEFAULT_DATA_CONFIG = {
    "uspto": "uspto",
    "spectrabase": "spectrabase",
    "spectranp": "spectranp",
}

DEFAULT_CONDITION = {
    "uspto": "h1c13nmr-10k-80",
    "spectrabase": "h1c13nmr-10k-80",
    "spectranp": "h1c13nmr-10k-10k",
}

DEFAULT_MODEL = {
    "uspto": "dit-l",
    "spectrabase": "dit-l",
    "spectranp": "dit-l",
}

DEFAULT_CKPT = {
    "uspto": CHEFNMR_CKPT_DIR / "US-H10kC80-L128-epoch3099.ckpt",
    "spectrabase": CHEFNMR_CKPT_DIR / "SB-H10kC80-L128-epoch5249.ckpt",
    "spectranp": CHEFNMR_CKPT_DIR / "NP-H10kC10k-L64-epoch18149.ckpt",
}


def _import_chefnmr():
    from src.utils import update_ckpt_config
    from src.data.datamodule import NMRDataModule
    from src.model.model import NMRTo3DStructureElucidation

    return update_ckpt_config, NMRDataModule, NMRTo3DStructureElucidation


class _FakeTrainer:
    def __init__(self):
        self.world_size = 1
        self.is_global_zero = True
        self.global_rank = 0
        self.local_rank = 0
        self.num_devices = 1


def canonical_dataset_name(dataset: str) -> str:
    d = (dataset or "").strip().lower()
    aliases = {
        "uspto": "uspto",
        "uspoto": "uspto",
        "spectrabase": "spectrabase",
        "sb": "spectrabase",
        "spectranp": "spectranp",
        "np": "spectranp",
    }
    if d not in aliases:
        raise ValueError(f"Unsupported ChefNMR dataset: {dataset}")
    return aliases[d]


def resolve_ckpt(dataset: str, explicit_ckpt: str) -> str:
    if explicit_ckpt:
        return str(Path(explicit_ckpt))
    path = DEFAULT_CKPT[dataset]
    if not path.exists():
        raise FileNotFoundError(f"ChefNMR checkpoint not found: {path}")
    return str(path)


def dataset_json_path(dataset: str, split: str) -> Path:
    subdir = "NMRGym" if dataset == "nmrgym" else dataset
    return DEFAULT_BENCH_ROOT / subdir / f"{split}.json"


def load_split_size(dataset: str, split: str) -> int:
    path = dataset_json_path(dataset, split)
    with open(path) as f:
        data = json.load(f)
    return len(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ChefNMR offline cache")
    parser.add_argument("--datasets", nargs="+", default=["uspto"], help="Datasets to process")
    parser.add_argument("--split", default="test", help="Benchmark split to use")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting ordinal sample index")
    parser.add_argument("--max-samples", type=int, default=128, help="Number of ordinal test samples to process")
    parser.add_argument("--chunk-samples", type=int, default=1024, help="Samples per chunk")
    parser.add_argument("--top-k", type=int, default=10, help="Candidates to store per sample")
    parser.add_argument("--diffusion-samples", type=int, default=10, help="ChefNMR multiplicity")
    parser.add_argument("--num-sampling-steps", type=int, default=50, help="ChefNMR reverse-diffusion steps")
    parser.add_argument("--test-batch-size", type=int, default=64, help="ChefNMR test batch size")
    parser.add_argument("--batch-size", type=int, default=64, help="ChefNMR dataset_args.batch_size")
    parser.add_argument(
        "--output",
        default=str(CHEFNMR_CKPT_DIR / "chefnmr_offline_cache.pkl"),
        help="Output pickle path",
    )
    parser.add_argument("--ckpt-path", default="", help="Optional explicit checkpoint path")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild cache from scratch")
    parser.add_argument("--save-every", type=int, default=1, help="Persist after every N chunks")
    return parser.parse_args()


def load_existing_cache(path: Path, overwrite: bool) -> dict:
    if overwrite or not path.exists():
        return {}
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Existing cache at {path} is not a dict")
    return obj


def persist_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f)
    tmp_path.replace(path)


def _compose_cfg(
    dataset: str,
    ckpt: Path,
    batch_size: int,
    test_batch_size: int,
    test_samples: int,
    diffusion_samples: int,
    num_sampling_steps: int,
):
    config_dir = str(CHEFNMR_DIR / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+data={DEFAULT_DATA_CONFIG[dataset]}",
                f"+condition={DEFAULT_CONDITION[dataset]}",
                f"+model={DEFAULT_MODEL[dataset]}",
                "+embedder=hybrid-baseline",
                "+lddt=threshold5124",
                "+training_transform=center_rot_trans",
                "+diffusion=edm-train_af3-sample_edm_sde",
                "+aug_conf=conf3",
                "+guidance=cfg15",
                "+exp=eval_ckpt",
                "general.experiment_name=chefnmr_offline_cache",
                f"dataset_args.batch_size={batch_size}",
                f"dataset_args.test_args.test_batch_size={test_batch_size}",
                f"dataset_args.test_args.test_samples={test_samples}",
                "dataset_args.test_args.test_index=null",
                f"test_args.diffusion_samples={diffusion_samples}",
                f"test_args.num_sampling_steps={num_sampling_steps}",
                "test_args.visualize_samples=0",
                "test_args.visualize_chains=0",
                "visualization_args.n_chain_frames=1",
                f"general.ckpt_abs_path={ckpt}",
                "general.seed=42",
            ],
        )
    update_ckpt_config, _, _ = _import_chefnmr()
    return update_ckpt_config(new_cfg=cfg)


@lru_cache(maxsize=8)
def _prepare_base_cfg(dataset: str, ckpt_path: str):
    _, NMRDataModule, _ = _import_chefnmr()
    ckpt = Path(ckpt_path)
    cfg = _compose_cfg(
        dataset=dataset,
        ckpt=ckpt,
        batch_size=64,
        test_batch_size=64,
        test_samples=1,
        diffusion_samples=10,
        num_sampling_steps=50,
    )
    dm = NMRDataModule(cfg.dataset_args)
    dm.prepare_data()
    if cfg.diffusion_process_args.edm_args.sigma_data is None:
        cfg.diffusion_process_args.edm_args.sigma_data = dm.sigma_data
    cfg.dataset_args.max_n_atoms = dm.max_n_atoms
    return cfg


@lru_cache(maxsize=8)
def load_chefnmr_test_size(dataset: str, ckpt_path: str) -> int:
    _, NMRDataModule, _ = _import_chefnmr()
    base_cfg = copy.deepcopy(_prepare_base_cfg(dataset, ckpt_path))
    dm = NMRDataModule(base_cfg.dataset_args)
    dm.prepare_data()
    if dm.test_indices is None:
        raise RuntimeError("ChefNMR datamodule test_indices is None after prepare_data().")
    return len(dm.test_indices)


@lru_cache(maxsize=8)
def _load_model_for_ckpt(dataset: str, ckpt_path: str):
    _, _, NMRTo3DStructureElucidation = _import_chefnmr()
    cfg = copy.deepcopy(_prepare_base_cfg(dataset, ckpt_path))
    model = NMRTo3DStructureElucidation.load_from_checkpoint(str(ckpt_path), cfg=cfg)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model._trainer = _FakeTrainer()
    model.log = lambda *args, **kwargs: None
    return model


def _build_test_loader(
    dataset: str,
    ckpt_path: str,
    sample_indices: list[int],
    batch_size: int,
    test_batch_size: int,
    diffusion_samples: int,
    num_sampling_steps: int,
):
    _, NMRDataModule, _ = _import_chefnmr()
    base_cfg = copy.deepcopy(_prepare_base_cfg(dataset, ckpt_path))
    base_cfg.dataset_args.batch_size = batch_size
    base_cfg.dataset_args.test_args.test_batch_size = test_batch_size
    base_cfg.dataset_args.test_args.test_samples = len(sample_indices)
    base_cfg.dataset_args.test_args.test_index = None
    base_cfg.test_args.diffusion_samples = diffusion_samples
    base_cfg.test_args.num_sampling_steps = num_sampling_steps

    dm = NMRDataModule(base_cfg.dataset_args)
    dm.prepare_data()
    if dm.test_indices is None:
        raise RuntimeError("ChefNMR datamodule test_indices is None after prepare_data().")

    total_test = len(dm.test_indices)
    if min(sample_indices) < 0 or max(sample_indices) >= total_test:
        raise IndexError(f"Requested ordinal range is out of bounds for dataset={dataset} with {total_test} test indices")

    ordered_actual = np.array([int(dm.test_indices[i]) for i in sample_indices], dtype=np.int64)
    dm.test_indices = ordered_actual
    dm.setup("test")
    return dm.test_dataloader(), ordered_actual.tolist()


def _extract_records_from_epoch_predictions(
    model,
    dataset: str,
    ordinal_indices: list[int],
    actual_indices: list[int],
    top_k: int,
    diffusion_samples: int,
    num_sampling_steps: int,
    ckpt_path: str,
) -> dict[str, dict[str, Any]]:
    preds = model.epoch_predictions
    target_smiles = preds.get("target_smiles", [])
    predicted_smiles = preds.get("predicted_smiles", [])
    similarity_list = preds.get("similarity_list", [])
    num_targets = len(target_smiles)
    multiplicity = max(1, len(predicted_smiles) // max(1, num_targets))

    out = {}
    for i, (ordinal_idx, actual_idx) in enumerate(zip(ordinal_indices, actual_indices)):
        start = i * multiplicity
        end = min((i + 1) * multiplicity, len(predicted_smiles))
        rows = []
        for pred, sim in zip(predicted_smiles[start:end], similarity_list[start:end]):
            rows.append(
                {
                    "target_smiles": target_smiles[i] if i < len(target_smiles) else "",
                    "predicted_smiles": pred,
                    "similarity": float(sim),
                }
            )
        rows.sort(key=lambda r: r["similarity"], reverse=True)

        candidates = []
        seen = set()
        for row in rows:
            smi = row.get("predicted_smiles")
            if not smi or smi == "None" or smi in seen:
                continue
            seen.add(smi)
            candidates.append(
                {
                    "smiles": smi,
                    "score": float(row["similarity"]),
                    "source": "chefnmr_denovo",
                    "rank": len(candidates) + 1,
                    "target_smiles": row.get("target_smiles", ""),
                }
            )
            if len(candidates) >= top_k:
                break

        key = f"{dataset}:{ordinal_idx}"
        out[key] = {
            "dataset": dataset,
            "sample_idx": ordinal_idx,
            "split": "test",
            "candidates": candidates,
            "metadata": {
                "dataset": dataset,
                "sample_idx": ordinal_idx,
                "actual_test_index": str(actual_idx),
                "ckpt_path": ckpt_path,
                "diffusion_samples": diffusion_samples,
                "num_sampling_steps": num_sampling_steps,
                "gpu": CHEFNMR_GPU,
                "multiplicity": multiplicity,
            },
        }
    return out


def run_batch_range(
    dataset: str,
    ckpt_path: str,
    start_idx: int,
    max_samples: int,
    top_k: int,
    diffusion_samples: int,
    num_sampling_steps: int,
    batch_size: int,
    test_batch_size: int,
) -> dict[str, dict[str, Any]]:
    model = _load_model_for_ckpt(dataset, ckpt_path)
    ordinal_indices = list(range(start_idx, start_idx + max_samples))
    loader, actual_indices = _build_test_loader(
        dataset=dataset,
        ckpt_path=ckpt_path,
        sample_indices=ordinal_indices,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        diffusion_samples=diffusion_samples,
        num_sampling_steps=num_sampling_steps,
    )

    seed_everything(42)
    model.test_args.diffusion_samples = diffusion_samples
    model.test_args.num_sampling_steps = num_sampling_steps
    model.epoch_predictions = model._init_epoch_predictions()

    for batch in loader:
        if torch.cuda.is_available():
            model = model.cuda()
            (model_inputs, target_smiles), model_outputs = batch
            model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
            model_outputs = {k: v.cuda() for k, v in model_outputs.items()}
            batch = ((model_inputs, target_smiles), model_outputs)
        model.sample_batch(
            batch,
            diffusion_samples=diffusion_samples,
            num_sampling_steps=num_sampling_steps,
        )

    return _extract_records_from_epoch_predictions(
        model=model,
        dataset=dataset,
        ordinal_indices=ordinal_indices,
        actual_indices=actual_indices,
        top_k=top_k,
        diffusion_samples=diffusion_samples,
        num_sampling_steps=num_sampling_steps,
        ckpt_path=ckpt_path,
    )


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    cache = load_existing_cache(output_path, overwrite=args.overwrite)

    for raw_dataset in args.datasets:
        dataset = canonical_dataset_name(raw_dataset)
        ckpt_path = resolve_ckpt(dataset, args.ckpt_path)
        bench_split_size = load_split_size(dataset, args.split)
        chefnmr_test_size = load_chefnmr_test_size(dataset, ckpt_path)
        max_samples = chefnmr_test_size - args.start_idx if args.max_samples < 0 else args.max_samples
        max_samples = min(max_samples, chefnmr_test_size - args.start_idx)

        print(f"=== dataset={dataset} split={args.split} ckpt={ckpt_path}")
        print(
            f"range: [{args.start_idx}, {args.start_idx + max_samples}) "
            f"out of ChefNMR test size {chefnmr_test_size} "
            f"(NMRBench {args.split}.json size={bench_split_size})"
        )

        chunk_size = max(1, int(args.chunk_samples))
        dataset_start = time.time()
        total_written = 0
        chunk_count = 0

        for chunk_start in range(args.start_idx, args.start_idx + max_samples, chunk_size):
            current_chunk = min(chunk_size, args.start_idx + max_samples - chunk_start)
            chunk_end = chunk_start + current_chunk
            chunk_count += 1
            print(f"[chunk {chunk_count}] dataset={dataset} range=[{chunk_start}, {chunk_end})")
            t0 = time.time()
            try:
                records = run_batch_range(
                    dataset=dataset,
                    ckpt_path=ckpt_path,
                    start_idx=chunk_start,
                    max_samples=current_chunk,
                    top_k=args.top_k,
                    diffusion_samples=args.diffusion_samples,
                    num_sampling_steps=args.num_sampling_steps,
                    batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size,
                )
                cache.update(records)
                total_written += len(records)
                elapsed = time.time() - t0
                print(
                    f"[ok] dataset={dataset} chunk=[{chunk_start},{chunk_end}) "
                    f"wrote={len(records)} chunk_time={elapsed:.1f}s total_written={total_written}"
                )
                if args.save_every > 0 and chunk_count % args.save_every == 0:
                    persist_cache(output_path, cache)
                    print(f"[save] {output_path} total_cache_records={len(cache)}")
            except Exception as exc:
                elapsed = time.time() - t0
                print(
                    f"[err] dataset={dataset} chunk=[{chunk_start},{chunk_end}) "
                    f"elapsed={elapsed:.1f}s error={exc}"
                )
                print(traceback.format_exc())
                persist_cache(output_path, cache)
                return 1

        persist_cache(output_path, cache)
        dataset_elapsed = time.time() - dataset_start
        print(
            f"[done-dataset] dataset={dataset} total_written={total_written} "
            f"elapsed={dataset_elapsed:.1f}s output={output_path}"
        )

    print(f"[done] wrote {len(cache)} total records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
