"""
models.py — GGUF Edition
------------------------
Switched from HuggingFace Transformers to llama-cpp-python.

WHY GGUF OVER bfloat16 TRANSFORMERS:
  - Q4_K_M quantization: ~1.9 GB on disk / ~2.1 GB RAM at runtime
    vs bfloat16 Transformers: ~6 GB RAM. Fits on any laptop now.
  - llama.cpp is written in C++ and is 2-4x faster on CPU than
    PyTorch for autoregressive generation.
  - Native stop-sequence support: no more post-processing string splits.
  - n_ctx gives direct control over context window budget.

INSTALL (run once):
  pip install llama-cpp-python
  # Windows CPU pre-compiled wheel (faster install):
  pip install llama-cpp-python --extra-index-url \
      https://abetlen.github.io/llama-cpp-python/whl/cpu

GGUF FILE:
  Run download_model.py first. It saves to:
    <repo_root>/models/qwen2.5-3b-instruct-q4_k_m.gguf
  Or set env var:  GGUF_MODEL_PATH=/your/path/model.gguf
"""

import os
import logging
import multiprocessing
from typing import List, Optional
from llama_cpp import Llama

logger = logging.getLogger(__name__)

_DEFAULT_GGUF = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "models",
        "qwen2.5-3b-instruct-q4_k_m.gguf"
    )
)


class LocalGenerator:
    """
    Singleton LLM wrapper around llama-cpp-python.
    Drop-in replacement for the Transformers-based class —
    the generate() signature is identical so no other files need changes.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = 0,       # 0 = auto-detect CPU cores
        n_gpu_layers: int = 0,    # 0 = full CPU; set e.g. 35 for full GPU offload
    ):
        if self._initialized:
            if model_path and model_path != self.model_id:
                logger.warning(
                    f"Singleton already loaded '{self.model_id}'. "
                    f"Ignoring request for '{model_path}'. Restart process to change."
                )
            return

        # Path resolution priority: env var > explicit arg > default
        resolved = (
            os.environ.get("GGUF_MODEL_PATH")
            or model_path
            or _DEFAULT_GGUF
        )
        resolved = os.path.abspath(resolved)

        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"GGUF model not found at: {resolved}\n"
                "Run:  python download_model.py"
            )

        threads = n_threads if n_threads > 0 else max(1, multiprocessing.cpu_count() - 1)

        logger.info(f"Loading GGUF: {resolved}")
        logger.info(f"n_ctx={n_ctx} | threads={threads} | gpu_layers={n_gpu_layers}")

        self._llm = Llama(
            model_path=resolved,
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=n_gpu_layers,
            chat_format="chatml",   # Qwen2.5 uses ChatML
            verbose=False,
        )

        self.model_id = os.path.basename(resolved)
        self.device   = "cpu" if n_gpu_layers == 0 else "gpu"
        self._initialized = True
        logger.info(f"Model ready. device={self.device}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        repetition_penalty: float = 1.05,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response. Interface is identical to the old Transformers
        version — all other files (service.py, generator.py, extractor.py)
        work without any changes.

        llama.cpp handles stop sequences natively, so no string-split hacks.
        """
        stops = stop_sequences or ["User:", "Assistant:", "Human:", "---"]

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=max(temperature, 1e-7),  # llama.cpp needs > 0
            repeat_penalty=repetition_penalty,
            stop=stops,
        )

        return response["choices"][0]["message"]["content"].strip()