# merge_lora.py
# Merge a LoRA/PEFT adapter into a base model and save a standalone merged model.
# Tested with: transformers >= 4.45, peft >= 0.13, accelerate >= 0.26, torch >= 2.1

from __future__ import annotations
import argparse
import gc
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _as_local_or_repo(s: str) -> str:
    """If s is a local path, normalize to POSIX so HF won't treat Windows paths as repo IDs."""
    p = Path(s)
    if p.exists():
        return p.resolve().as_posix()
    return s


def _ensure_adapter_folder(adapter_path: str) -> None:
    ap = Path(adapter_path)
    if not ap.exists():
        sys.exit(f"[error] adapter folder not found: {adapter_path}")
    cfg = ap / "adapter_config.json"
    if not cfg.exists():
        sys.exit(f"[error] missing adapter_config.json in: {adapter_path}")


def merge_lora_into_base(
    base_path: str,
    adapter_path: str,
    out_dir: str,
    offload_dir: str | None = None,
    cpu_only: bool = False,
    load_in_8bit: bool = False,
    local_only: bool = False,
    shard_size: str = "1GB",
) -> str:
    """
    Merge a PEFT/LoRA adapter into a base causal LM and save a standalone merged model.
    Returns output directory path as string.
    """
    base_path = _as_local_or_repo(base_path)
    adapter_path = _as_local_or_repo(adapter_path)
    _ensure_adapter_folder(adapter_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    offload = Path(offload_dir).resolve() if offload_dir else None
    if offload and not cpu_only:
        offload.mkdir(parents=True, exist_ok=True)

    # dtype selection
    if cpu_only:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[i] Loading base: {base_path}")
    base_kwargs = {
        "dtype": dtype,               # replaces deprecated torch_dtype
        "low_cpu_mem_usage": True,
    }
    if cpu_only:
        base_kwargs["device_map"] = None
    else:
        base_kwargs["device_map"] = "auto"
        if offload:
            base_kwargs["offload_folder"] = str(offload)
        if load_in_8bit:
            # Keeps it simple; you can switch to BitsAndBytesConfig later
            base_kwargs["load_in_8bit"] = True

    # If base_path is local or user forces local, avoid hub lookup
    if local_only or Path(base_path).exists():
        base_kwargs["local_files_only"] = True

    base = AutoModelForCausalLM.from_pretrained(base_path, **base_kwargs)

    print(f"[i] Loading adapter: {adapter_path}")
    peft_kwargs = {}
    if cpu_only:
        peft_kwargs["device_map"] = None
    else:
        peft_kwargs["device_map"] = "auto"
        if offload:
            # IMPORTANT: only offload_dir here (PEFT forbids offload_folder in this call)
            peft_kwargs["offload_dir"] = str(offload)

    model = PeftModel.from_pretrained(base, adapter_path, **peft_kwargs)

    print("[i] Merging adapter into base …")
    merged = model.merge_and_unload()  # returns a plain transformers model

    # Free memory from PEFT wrapper + base (GPU/offload)
    del model
    del base
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    print("[i] Finalizing on CPU for save …")
    merged = merged.to("cpu")
    gc.collect()

    print(f"[i] Saving merged model to: {out.as_posix()} (max_shard_size={shard_size})")
    merged.save_pretrained(
        out,
        safe_serialization=True,   # writes .safetensors
        max_shard_size=shard_size  # keep RAM peaks reasonable; use "512MB" if needed
    )

    tok = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    tok.save_pretrained(out)

    print("[ok] Saved merged model.")
    return str(out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge a LoRA/PEFT adapter into a base model.")
    ap.add_argument("--base", required=True, help="Base model repo id or local path")
    ap.add_argument("--adapter", required=True, help="PEFT adapter folder (with adapter_config.json)")
    ap.add_argument("--out", required=True, help="Directory to write merged model")
    ap.add_argument("--offload", help="Folder for Accelerate CPU offload (device_map='auto')")
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU-only merge (slow but robust)")
    ap.add_argument("--load_in_8bit", action="store_true", help="Load base in 8-bit (bitsandbytes required)")
    ap.add_argument("--local_only", action="store_true", help="Force local files only for base load")
    ap.add_argument("--shard_size", default="1GB", help="Max shard size for saving (e.g., 1GB, 512MB)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_path = merge_lora_into_base(
        base_path=args.base,
        adapter_path=args.adapter,
        out_dir=args.out,
        offload_dir=args.offload,
        cpu_only=args.cpu_only,
        load_in_8bit=args.load_in_8bit,
        local_only=args.local_only,
        shard_size=args.shard_size,
    )
    print("     Load like:")
    print(f'     tok = AutoTokenizer.from_pretrained(r"{out_path}")')
    print(f'     mdl = AutoModelForCausalLM.from_pretrained(r"{out_path}", device_map="auto")')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
