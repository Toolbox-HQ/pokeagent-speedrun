#!/usr/bin/env python3
import json
import sys
import torch
from pathlib import Path

CACHE_DIR = Path(".cache/lz/dinov2")
EXPECTED_DIM = 768
JSON_REQUIRED_KEYS = {"video_path", "video_fps", "video_total_frames", "sampled_frame_index"}

def validate_pair(stem: str) -> list[str]:
    errors = []
    pt_path = CACHE_DIR / f"{stem}.pt"
    json_path = CACHE_DIR / f"{stem}.json"

    try:
        tensor = torch.load(pt_path, weights_only=True)
    except Exception as e:
        errors.append(f"{pt_path}: failed to load: {e}")
        tensor = None

    if tensor is not None:
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{pt_path}: expected Tensor, got {type(tensor)}")
        elif tensor.ndim != 2 or tensor.shape[1] != EXPECTED_DIM:
            errors.append(f"{pt_path}: bad shape {tuple(tensor.shape)}, expected (N, {EXPECTED_DIM})")
        elif torch.isnan(tensor).any():
            errors.append(f"{pt_path}: contains NaN")
        elif torch.isinf(tensor).any():
            errors.append(f"{pt_path}: contains Inf")

    try:
        with open(json_path) as f:
            metadata = json.load(f)
    except Exception as e:
        errors.append(f"{json_path}: failed to load: {e}")
        metadata = None

    if metadata is not None:
        if not isinstance(metadata, list):
            errors.append(f"{json_path}: expected list, got {type(metadata)}")
        else:
            for i, entry in enumerate(metadata):
                missing = JSON_REQUIRED_KEYS - entry.keys()
                if missing:
                    errors.append(f"{json_path}[{i}]: missing keys {missing}")
                    break
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                if len(metadata) != tensor.shape[0]:
                    errors.append(
                        f"{stem}: JSON has {len(metadata)} entries but tensor has {tensor.shape[0]} rows"
                    )

    return errors


def main():
    pt_stems = {p.stem for p in CACHE_DIR.glob("*.pt")}
    json_stems = {p.stem for p in CACHE_DIR.glob("*.json")}

    all_stems = pt_stems | json_stems
    print(f"Found {len(all_stems)} unique IDs ({len(pt_stems)} .pt, {len(json_stems)} .json)")

    pt_only = pt_stems - json_stems
    json_only = json_stems - pt_stems
    for stem in sorted(pt_only):
        print(f"MISSING JSON: {stem}.json")
    for stem in sorted(json_only):
        print(f"MISSING PT:   {stem}.pt")

    paired = pt_stems & json_stems
    all_errors = []
    for i, stem in enumerate(sorted(paired), 1):
        if i % 500 == 0:
            print(f"  [{i}/{len(paired)}] checked...")
        errs = validate_pair(stem)
        all_errors.extend(errs)

    if all_errors:
        print(f"\nFOUND {len(all_errors)} ERROR(S):")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        missing = len(pt_only) + len(json_only)
        print(f"\nAll {len(paired)} pairs valid." + (f" ({missing} unpaired files)" if missing else ""))


if __name__ == "__main__":
    main()
