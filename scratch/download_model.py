"""
download_model.py — GGUF Edition
---------------------------------
Downloads Qwen2.5-3B-Instruct Q4_K_M GGUF from HuggingFace Hub.

Q4_K_M is the recommended quantization level:
  - Q4_K_M: ~1.9 GB  | Best quality/size tradeoff for 4-bit
  - Q4_K_S: ~1.7 GB  | Slightly smaller, slightly lower quality
  - Q5_K_M: ~2.2 GB  | Better quality if you have RAM headroom
  - Q8_0:   ~3.2 GB  | Near-lossless, but 3B bfloat16 RAM levels

Run once before starting the server (from repo root):
  python scratch/download_model.py

INSTALL dependency:
  pip install huggingface_hub
"""

import os
from huggingface_hub import hf_hub_download

# ── Config ─────────────────────────────────────────────────────────────────
REPO_ID   = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME  = "qwen2.5-3b-instruct-q4_k_m.gguf"
SAVE_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
# ───────────────────────────────────────────────────────────────────────────


def download():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dest = os.path.abspath(os.path.join(SAVE_DIR, FILENAME))

    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 ** 2)
        print(f"Model already exists ({size_mb:.0f} MB): {dest}")
        print("Delete the file and re-run to force re-download.")
        return dest

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    print("Expected size: ~1.9 GB. This will take a few minutes.\n")

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=SAVE_DIR,
            # resume_download removed: deprecated in huggingface_hub >= 0.24
            # downloads always resume automatically now
        )
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"\nDownload complete! ({size_mb:.0f} MB)")
        print(f"Saved to: {os.path.abspath(path)}")
        return path
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Check your internet connection or HuggingFace Hub access.")
        raise


if __name__ == "__main__":
    download()