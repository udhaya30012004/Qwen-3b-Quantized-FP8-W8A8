"""Inspect and upload the Qwen2.5-3B FP8 W8A8 Per-Channel checkpoint + Model Card + Plots to Hugging Face Hub.

Runs on a lightweight CPU container in Modal (zero GPU cost) with fast cloud-to-cloud transfer.

Usage (from E:\\quantt\\fp_8\\hf_upload in PowerShell):
    # 1. Dry-run first to inspect every file on the Modal Volume (no upload, no token needed):
    modal run upload_to_hf.py --dry-run

    # 2. Upload to Hugging Face (set $env:HF_TOKEN first, or pass --hf-token):
    $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxx"
    modal run upload_to_hf.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
_CFG_PATH = HERE / "hf_config.json"
MODAL_CFG = (
    json.loads(_CFG_PATH.read_text(encoding="utf-8"))["modal"]
    if _CFG_PATH.exists()
    else {"app_name": "quantt-fp8-hf-upload", "volume_name": "quant-models", "volume_mount": "/vol"}
)

app = modal.App(MODAL_CFG["app_name"])
volume = modal.Volume.from_name(MODAL_CFG["volume_name"], create_if_missing=False)
VOLUMES = {MODAL_CFG["volume_mount"]: volume}

upload_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.25.0", "hf_transfer>=0.1.8")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=upload_image, volumes=VOLUMES, timeout=1800)
def inspect_and_upload(
    cfg: dict,
    readme_text: str,
    asset_files: dict[str, bytes],
    hf_token: str = "",
    dry_run: bool = True,
) -> dict:
    import fnmatch
    import os
    import struct

    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    volume.reload()
    src = cfg["modal"]["remote_model_dir"]
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Checkpoint directory not found on Modal Volume: {src}")

    all_files = sorted(
        f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))
    )
    allow = cfg["allow_patterns"]
    ignore = cfg["ignore_patterns"]

    def _matches(name: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    to_upload, skipped = [], []
    for name in all_files:
        if _matches(name, allow) and not _matches(name, ignore):
            to_upload.append(name)
        else:
            skipped.append(name)

    print(f"\n=== Modal Volume Checkpoint Audit: {src} ===")
    total_bytes = 0
    print("\n[CHECKPOINT FILES TO UPLOAD]")
    for name in to_upload:
        sz = os.path.getsize(os.path.join(src, name))
        total_bytes += sz
        print(f"  + {name:<36} {sz / 1e6:>9.2f} MB")

    print("\n[MODEL CARD & PLOTS TO UPLOAD]")
    readme_bytes = readme_text.encode("utf-8")
    total_bytes += len(readme_bytes)
    print(f"  + {'README.md':<36} {len(readme_bytes) / 1e6:>9.2f} MB")
    for rel_path, content in sorted(asset_files.items()):
        total_bytes += len(content)
        print(f"  + {rel_path:<36} {len(content) / 1e6:>9.2f} MB")

    total_count = len(to_upload) + 1 + len(asset_files)
    print(f"\n  Total upload size: {total_bytes / 1e9:.2f} GB ({total_count} files)")

    if skipped:
        print("\n[EXCLUDED / SKIPPED]")
        for name in skipped:
            sz = os.path.getsize(os.path.join(src, name))
            print(f"  - {name:<36} {sz / 1e6:>9.2f} MB")

    # ---- Sanity checks on required files and quantization_config
    missing = [req for req in cfg["required_files"] if req not in to_upload]
    has_weights = any(f.endswith(".safetensors") for f in to_upload)
    if missing:
        raise RuntimeError(f"Missing required Hugging Face files: {missing}")
    if not has_weights:
        raise RuntimeError("No .safetensors weight files found in checkpoint directory!")

    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        model_cfg = json.load(f)
    qcfg = model_cfg.get("quantization_config")
    if not qcfg:
        raise RuntimeError("config.json is missing 'quantization_config'!")

    print("\n[CONFIG.JSON VERIFICATION]")
    print(f"  architectures : {model_cfg.get('architectures')}")
    print(f"  quant_method  : {qcfg.get('quant_method')}")
    print(f"  format        : {qcfg.get('format')}")
    print(f"  ignore        : {qcfg.get('ignore')}")
    print("  status        : OK (ready for vLLM & transformers)")

    # ---- Verify safetensors binary header integrity (pure Python, instant)
    print("\n[SAFETENSORS HEADER VERIFICATION]")
    for sf in [f for f in to_upload if f.endswith(".safetensors")]:
        sf_path = os.path.join(src, sf)
        with open(sf_path, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            if hdr_len == 0 or hdr_len > 100_000_000:
                raise RuntimeError(f"Corrupted safetensors header length ({hdr_len}) in {sf}")
            hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        n_tensors = sum(1 for k in hdr if k != "__metadata__")
        print(f"  {sf:<36} header=OK, tensors={n_tensors}")

    repo_id = cfg["repo_id"]
    if dry_run:
        print(f"\n[DRY RUN COMPLETE] Target repo: https://huggingface.co/{repo_id}")
        print("Re-run without --dry-run (and with HF_TOKEN set) to publish.")
        return {
            "dry_run": True,
            "files": to_upload + ["README.md"] + sorted(asset_files.keys()),
            "skipped": skipped,
            "total_bytes": total_bytes,
        }

    if "YOUR_HF_USERNAME" in repo_id:
        raise ValueError("Please edit hf_config.json and replace 'YOUR_HF_USERNAME' with your real Hugging Face username.")
    if not hf_token:
        raise ValueError("Missing Hugging Face token. Set $env:HF_TOKEN or pass --hf-token 'hf_...'")

    api = HfApi(token=hf_token)
    whoami = api.whoami()
    print(f"\nAuthenticated to Hugging Face as: {whoami.get('name')}")

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        repo_type="model",
        private=cfg.get("private", False),
        exist_ok=True,
    )

    print(f"Uploading {len(to_upload)} checkpoint files to https://huggingface.co/{repo_id} ...")
    api.upload_folder(
        folder_path=src,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=allow,
        ignore_patterns=ignore,
        commit_message=cfg.get("commit_message", "Upload quantized FP8 model"),
    )

    # Upload README.md + all benchmark plots in one atomic commit
    final_readme = readme_text.replace("UchihaUdhaya/Qwen2.5-3B-FP8-W8A8-Dynamic", repo_id)
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=final_readme.encode("utf-8")),
    ]
    for rel_path, content in sorted(asset_files.items()):
        operations.append(CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=content))

    print(f"Uploading README.md + {len(asset_files)} benchmark plots ...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Add model card (README.md) with benchmark results and evaluation plots",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nSUCCESS! Model published at: {url}")
    return {
        "dry_run": False,
        "url": url,
        "files": to_upload + ["README.md"] + sorted(asset_files.keys()),
        "total_bytes": total_bytes,
    }


@app.local_entrypoint()
def main(
    repo_id: str = "",
    hf_token: str = "",
    dry_run: bool = False,
) -> None:
    import os

    cfg = json.loads((HERE / "hf_config.json").read_text(encoding="utf-8"))
    if repo_id:
        cfg["repo_id"] = repo_id
    token = hf_token or os.environ.get("HF_TOKEN", "")
    readme_text = (HERE / "README.md").read_text(encoding="utf-8")

    assets_dir = HERE / "assets"
    asset_files: dict[str, bytes] = {}
    if assets_dir.is_dir():
        for p in sorted(assets_dir.glob("*.png")):
            asset_files[f"assets/{p.name}"] = p.read_bytes()

    inspect_and_upload.remote(
        cfg=cfg,
        readme_text=readme_text,
        asset_files=asset_files,
        hf_token=token,
        dry_run=dry_run,
    )
