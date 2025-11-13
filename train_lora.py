#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_geopolitics_lora.py
---------------------------------------------------------
A clean, CLI-ready LoRA/QLoRA trainer for Mistral/Llama.

Features
- QLoRA (4-bit) by default; toggle off with --no-qlora
- Robust dataset loader: HF hub name OR local json/jsonl
- Auto-map "Question" + preferred "* Answer" columns OR Alpaca-style "instruction"/"output"
- Simple prompt template: "### Instruction:\n...\n\n### Response:\n..."
- Trainer (no TRL dependency) for stable APIs
- Works on Windows + Python 3.11 (recommended for training)

Install (in your training venv, e.g., train311):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.57.1 datasets==4.4.1 peft==0.17.1 accelerate==1.11.0 bitsandbytes==0.48.2
    pip install sentencepiece protobuf==4.25.3

Example (HF dataset):
    python train_geopolitics_lora.py ^
        --base_model mistralai/Mistral-7B-Instruct-v0.3 ^
        --dataset enkryptai/deepseek-geopolitical-bias-dataset ^
        --out lora_out_geo_bias ^
        --epochs 1 --max_len 1024 --bsz 1 --grad_acc 16

Example (local jsonl, Alpaca-style):
    python train_geopolitics_lora.py ^
        --base_model mistralai/Mistral-7B-Instruct-v0.3 ^
        --dataset data/geo_train.jsonl ^
        --out lora_out_geo ^
        --epochs 1 --max_len 1024 --bsz 1 --grad_acc 16
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Union

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)


# ----------------------------- Data mapping -----------------------------

DEFAULT_QUESTION_CANDS = ["Question", "question", "Prompt", "prompt", "Instruction", "instruction"]
DEFAULT_ANSWER_PREFS = [
    "O1 Answer",
    "DeepSeek-R1 Answer",
    "DeepSeek Chat Answer",
    "DeepSeek Distilled Llama 8B Answer",
    "Sonnet Answer",
    "Opus Answer",
    "answer",  # fallback
    "output",  # if already in alpaca style
]


def detect_columns(cols: List[str], question_col: Optional[str], answer_cols: Optional[List[str]]):
    # Question column
    qc = question_col
    if not qc:
        for c in DEFAULT_QUESTION_CANDS:
            if c in cols:
                qc = c
                break
        if not qc:
            # last fallback: first column
            qc = cols[0]

    # Answer columns
    ac = answer_cols or [c for c in DEFAULT_ANSWER_PREFS if c in cols]
    return qc, ac


def map_hf_to_pairs(ds: Dataset, question_col: str, answer_cols: List[str]) -> Dataset:
    def pick_answer(row: Dict) -> Optional[str]:
        for c in answer_cols:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def to_pair(row: Dict) -> Optional[Dict]:
        if not isinstance(row, dict):
            return None
        q = str(row.get(question_col, "")).strip()
        a = pick_answer(row)
        if not q or not a:
            return None
        return {"instruction": q, "output": a}

    pairs = []
    for r in ds:
        ex = to_pair(r)
        if ex:
            pairs.append(ex)

    if not pairs:
        raise RuntimeError(
            f"No usable samples found. Available cols={ds.column_names}; "
            f"question_col='{question_col}', answer_cols={answer_cols}"
        )
    return Dataset.from_list(pairs)


def load_any_dataset(path_or_name: str) -> Dataset:
    """
    Supports:
    - HuggingFace dataset name: e.g., "enkryptai/deepseek-geopolitical-bias-dataset"
    - Local json/jsonl with a list of {instruction, output} or similar
    """
    # Local file?
    lower = path_or_name.lower()
    if os.path.isfile(path_or_name) and (lower.endswith(".jsonl") or lower.endswith(".json")):
        if lower.endswith(".jsonl"):
            rows = []
            with open(path_or_name, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
        else:
            with open(path_or_name, "r", encoding="utf-8") as f:
                rows = json.load(f)

        # Normalize to instruction/output if possible
        normalized = []
        for r in rows:
            if isinstance(r, dict) and ("instruction" in r and "output" in r):
                normalized.append({"instruction": str(r["instruction"]), "output": str(r["output"])})
            else:
                # Try common alternates:
                q = None
                for c in DEFAULT_QUESTION_CANDS + ["input", "question_text", "query"]:
                    if c in r and isinstance(r[c], str):
                        q = r[c]
                        break
                a = None
                for c in DEFAULT_ANSWER_PREFS:
                    if c in r and isinstance(r[c], str):
                        a = r[c]
                        break
                if q and a:
                    normalized.append({"instruction": str(q), "output": str(a)})

        if not normalized:
            raise RuntimeError("No usable records found in local file; expected {'instruction','output'} or compatible columns.")
        return Dataset.from_list(normalized)

    # Otherwise try HF hub
    maybe = load_dataset(path_or_name)
    if isinstance(maybe, DatasetDict):
        return maybe["train"]
    return maybe


# ----------------------------- Training -----------------------------

PROMPT_INSTR = "### Instruction:\n"
PROMPT_RESP = "\n\n### Response:\n"


def build_prompt(instruction: str, output: str) -> str:
    return f"{PROMPT_INSTR}{instruction.strip()}{PROMPT_RESP}{output.strip()}"


def main():
    ap = argparse.ArgumentParser(description="LoRA/QLoRA trainer for Mistral/Llama on geopolitics data")
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dataset", required=True, help="HF dataset name or path to local json/jsonl")
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_holdout", type=float, default=0.05)
    ap.add_argument("--no_qlora", action="store_true", help="Disable QLoRA (use full precision)")
    ap.add_argument("--question_col", default=None, help="Override question column name for HF datasets")
    ap.add_argument("--answer_cols", nargs="*", default=None, help="Override answer column names (space separated)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load dataset
    raw = load_any_dataset(args.dataset)
    cols = raw.column_names

    # If it already has instruction/output, great; else map from HF style
    if "instruction" in cols and "output" in cols:
        ds_pairs = raw
    else:
        qcol, acols = detect_columns(cols, args.question_col, args.answer_cols)
        ds_pairs = map_hf_to_pairs(raw, qcol, acols)

    # Split
    ds_splits = ds_pairs.train_test_split(test_size=args.eval_holdout, seed=args.seed)
    train_ds, eval_ds = ds_splits["train"], ds_splits["test"]
    print(f"[info] samples: train={len(train_ds)}, eval={len(eval_ds)}")

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA config
    bnb_cfg = None
    if not args.no_qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype="auto",
    )

    # Prepare model & attach LoRA
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_cfg)

    # Build + tokenize
    def tok(example: Dict[str, str]):
        text = build_prompt(example["instruction"], example["output"])
        toks = tokenizer(
            text,
            truncation=True,
            max_length=args.max_len,
            padding=False,
            return_tensors=None,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    train_tok = train_ds.map(tok, remove_columns=train_ds.column_names, desc="Tokenizing train")
    eval_tok = eval_ds.map(tok, remove_columns=eval_ds.column_names, desc="Tokenizing eval")

    # Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,  # your 2070 supports fp16
        bf16=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=data_collator,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
    )

    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"✅ LoRA adapter saved to {args.out}")


if __name__ == "__main__":
    main()
