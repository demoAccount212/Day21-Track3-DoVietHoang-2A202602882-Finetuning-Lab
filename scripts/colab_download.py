#!/usr/bin/env python3
"""
Run this IN COLAB (not locally) to package and download all artifacts.
Copy-paste into a Colab cell and run.
"""
import os
import zipfile
import subprocess
from pathlib import Path

# Find the repo root in Colab
possible_paths = [
    "/content/Day21-Track3-Finetuning-Lab",
    "/content/drive/MyDrive/Day21-Track3-Finetuning-Lab",
    "/home/user/Day21-Track3-Finetuning-Lab",
]
repo_root = None
for p in possible_paths:
    if Path(p).exists():
        repo_root = Path(p)
        break

if repo_root is None:
    # Try to find it via git
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if result.returncode == 0:
        repo_root = Path(result.stdout.strip())

if repo_root is None:
    raise RuntimeError("Could not find repo root. Run this in Colab after cloning the repo.")

print(f"Repo root: {repo_root}")

# Files to package
files_to_pack = [
    # Results
    "results/baselines_frozen.json",
    "results/runs.csv",
    "results/verdict.json",
    "results/autopsy.json",
    "results/qualitative.json",
    # Adapters
    "adapters/correct/adapter_model.safetensors",
    "adapters/correct/adapter_config.json",
    "adapters/attn_only/adapter_model.safetensors",
    "adapters/attn_only/adapter_config.json",
    "adapters/wrong_lr/adapter_model.safetensors",
    "adapters/wrong_lr/adapter_config.json",
    "adapters/qlora/adapter_model.safetensors",
    "adapters/qlora/adapter_config.json",
]

existing_files = []
missing_files = []
for rel in files_to_pack:
    p = repo_root / rel
    if p.exists():
        existing_files.append(p)
    else:
        missing_files.append(rel)

if missing_files:
    print(f"⚠️ Missing (not generated yet): {missing_files}")

if not existing_files:
    raise RuntimeError("No artifact files found. Run NB2-NB5 first.")

# Create zip
zip_path = repo_root / "lab21_artifacts.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in existing_files:
        arcname = p.relative_to(repo_root)
        zf.write(p, arcname)
        print(f"  Added: {arcname}")

print(f"\n✅ Created: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

# Auto-download in Colab
try:
    from google.colab import files
    files.download(str(zip_path))
    print("📥 Download started in browser...")
except ImportError:
    print("Not in Colab environment. Zip saved at:", zip_path)