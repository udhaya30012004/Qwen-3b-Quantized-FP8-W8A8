# How to Upload `Qwen2.5-3B-FP8-Dynamic` (`w8a8_pc`) to Hugging Face

This folder (`fp_8/hf_upload/`) is a self-contained publishing package for uploading your verified FP8 W8A8 Per-Channel checkpoint directly from your Modal Volume (`quant-models`) to Hugging Face Hub.

## Files in This Folder

1. **`hf_config.json`** — Upload configuration (target `repo_id`, `private` flag, file allowlist/ignore list).
2. **`README.md`** — Production Hugging Face Model Card with YAML metadata, verified benchmark tables, and `vLLM` / `transformers` code examples.
3. **`upload_to_hf.py`** — Standalone Modal CPU script (`hf_transfer` enabled) that audits the checkpoint and uploads it cloud-to-cloud in ~15 seconds.

---

## Step-by-Step Instructions (PowerShell)

### Step 1: Set Your Hugging Face Repo Name
Open `hf_config.json` and change `"YOUR_HF_USERNAME/Qwen2.5-3B-FP8-Dynamic"` to your actual Hugging Face username:
```json
"repo_id": "your-username/Qwen2.5-3B-FP8-Dynamic"
```
*(Optional: set `"private": true` if you want to inspect the repo privately before making it public.)*

### Step 2: Run a Dry-Run Audit First (No Token Needed)
Verify every file on the Modal Volume and confirm `config.json` has `quantization_config`:
```powershell
cd e:\quantt\fp_8\hf_upload
modal run upload_to_hf.py --dry-run
```

### Step 3: Set Your Hugging Face Write Token & Publish
Create a **Write** token at https://huggingface.co/settings/tokens, then run:
```powershell
$env:HF_TOKEN = "hf_your_write_token_here"
modal run upload_to_hf.py
```

*(You can also override the repo ID from the CLI without editing `hf_config.json`:)*
```powershell
modal run upload_to_hf.py --repo-id "your-username/Qwen2.5-3B-FP8-Dynamic"
```
