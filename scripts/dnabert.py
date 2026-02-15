"""
DNABERT Genome Scanning Script (Base + Fine-tuned)

Runs genome-wide promoter scanning on an E. coli genome using DNABERT.
This script supports two workflows:

1) Base model scanning (no fine-tuning):
   Uses the pretrained DNABERT model directly.

2) Fine-tuned scanning:
   Uses a locally fine-tuned checkpoint produced by `fine_tuned_dnabert.py`,
   then scans the whole genome with that fine-tuned model.

Example usage (Windows):

A) WITHOUT fine-tuning (pretrained base model):
    python dnabert.py ^
        --fasta "sequence.fasta" ^
        --model "zhihan1996/DNA_bert_6" ^
        --outdir "\\scan_out" ^
        --window 100 ^
        --step 10 ^
        --known_tss_csv "promoter.tsv" ^
        --known_tss_col "posTSS"

B) WITH fine-tuning (use local fine-tuned model folder):
   1) First run fine-tuning (example as in fine_tuned_dnabert):
        python fine_tuned_dnabert.py ^
            --fasta "sequence.fasta" ^
            --promoter_tsv "promoter.tsv" ^
            --outdir "\\out_dnabert" ^
            --window 100 ^
            --epochs 3

   2) Important (only if your Transformers setup needs it):
      Copy `dnabert_layer.py` from the HuggingFace modules cache into the
      fine-tuned model folder (so the local checkpoint can be loaded).

      Typical cache location (for Windows):
        %USERPROFILE%\\.cache\\huggingface\\modules\\transformers_modules\\
            zhihan1996\\DNA_bert_6\\...\\dnabert_layer.py

      Copy it into:
        \\out_dnabert\\dnabert_finetuned_w100\\dnabert_layer.py

   3) Run genome scanning using the fine-tuned model:
        python dnabert.py ^
            --fasta "sequence.fasta" ^
            --model "\\out_dnabert\\dnabert_finetuned_w100" ^
            --outdir "\\scan_out_fine_tuned" ^
            --window 100 ^
            --step 10 ^

"""


import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# DNABERT k-mer helper - convert a DNA sequence into space-separated k-mers (DNABERT expects this)
def seq_to_kmers(seq: str, k: int = 6) -> str:
    seq = seq.upper()
    if len(seq) < k:
        return seq
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


# DNABERT models expect A/C/G/T only, skipping windows with ambiguous letters (N, etc.)
def has_only_atcg(seq: str) -> bool:
    for ch in seq:
        if ch not in ("A", "C", "G", "T"):
            return False
    return True


# FASTA load
def load_genome_fasta(fasta_path: str) -> str:
    record = SeqIO.read(fasta_path, "fasta")
    genome = str(record.seq).upper()
    return genome


# sliding window (centered)
def generate_centers(genome_len: int, window: int, step: int) -> List[int]:
    half = window // 2
    return list(range(half, genome_len - half, step))


def get_window_centered(genome: str, center: int, window: int) -> str:
    half = window // 2
    start = center - half
    end = start + window
    return genome[start:end]


# chunked scanning with resume
@dataclass
class ScanConfig:
    window: int = 100
    step: int = 10
    kmer: int = 6
    batch_size: int = 64
    chunk_size: int = 50000
    use_fp16: bool = True
    num_workers_tokenize: int = 0


def dnabert_scan_genome(
        model,
        tokenizer,
        genome: str,
        cfg: ScanConfig,
        out_prefix: str,
        device: torch.device,
        resume: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    genome_len = len(genome)
    centers = generate_centers(genome_len, cfg.window, cfg.step)
    total = len(centers)

    meta_path = f"{out_prefix}_meta.json"
    centers_path = f"{out_prefix}_centers.npy"
    probs_path = f"{out_prefix}_probs.npy"

    start_index = 0
    probs_all: List[float] = []
    centers_all: List[int] = []

    if resume and os.path.exists(meta_path) and os.path.exists(centers_path) and os.path.exists(probs_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        start_index = int(meta.get("next_index", 0))

        centers_all = np.load(centers_path).tolist()
        probs_all = np.load(probs_path).tolist()

        if len(centers_all) != len(probs_all):
            start_index = 0
            centers_all, probs_all = [], []
    model.eval()

    # we use sigmoid if model outputs 1 logit, else softmax for 2 logits
    def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(-1) == 1:
            # shape: (B,1)
            return torch.sigmoid(logits).squeeze(-1)
        elif logits.dim() == 2 and logits.size(-1) >= 2:
            # shape: (B,2) expected, take class 1 prob
            return torch.softmax(logits, dim=-1)[:, 1]
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    autocast_ctx = torch.cuda.amp.autocast if (cfg.use_fp16 and device.type == "cuda") else torch.cpu.amp.autocast

    pbar = tqdm(range(start_index, total, cfg.chunk_size), desc="Scanning chunks", unit="chunk")

    for chunk_start in pbar:
        chunk_end = min(chunk_start + cfg.chunk_size, total)
        chunk_centers = centers[chunk_start:chunk_end]

        # building window strings
        windows = []
        valid_centers = []
        for c in chunk_centers:
            w = get_window_centered(genome, c, cfg.window)
            # skipping ambiguous windows
            if not has_only_atcg(w):
                continue
            windows.append(seq_to_kmers(w, cfg.kmer))
            valid_centers.append(c)

        if not windows:
            save_meta(meta_path, cfg, chunk_end, total)
            continue

        # batch inference
        chunk_probs: List[float] = []
        for i in tqdm(range(0, len(windows), cfg.batch_size), leave=False, desc="Batches", unit="batch"):
            batch_text = windows[i:i + cfg.batch_size]
            enc = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                with autocast_ctx():
                    out = model(**enc)
                    logits = out.logits
                    prob = logits_to_prob(logits)

            chunk_probs.extend(prob.detach().float().cpu().numpy().tolist())

        # append and persist
        centers_all.extend(valid_centers)
        probs_all.extend(chunk_probs)

        np.save(centers_path, np.asarray(centers_all, dtype=np.int64))
        np.save(probs_path, np.asarray(probs_all, dtype=np.float32))
        save_meta(meta_path, cfg, chunk_end, total)

        pbar.set_postfix({"saved": len(probs_all)})

    centers_np = np.asarray(centers_all, dtype=np.int64)
    probs_np = np.asarray(probs_all, dtype=np.float32)
    return centers_np, probs_np


def save_meta(meta_path: str, cfg: ScanConfig, next_index: int, total: int):
    meta = {
        "next_index": next_index,
        "total_centers": total,
        "window": cfg.window,
        "step": cfg.step,
        "kmer": cfg.kmer,
        "batch_size": cfg.batch_size,
        "chunk_size": cfg.chunk_size,
        "use_fp16": cfg.use_fp16
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# peak merge
def merge_peaks(positions: List[int], merge_gap: int = 200) -> List[int]:
    if not positions:
        return []
    positions = sorted(positions)
    merged = [positions[0]]
    last = positions[0]
    for p in positions[1:]:
        if p - last > merge_gap:
            merged.append(p)
            last = p
    return merged


def evaluate_peaks(
        merged_peaks: List[int],
        known_positions: List[int],
        tol: int = 100
) -> Tuple[int, int, int, float, float, float]:
    known_positions = list(known_positions)

    tp = 0
    for k in known_positions:
        if any(abs(k - p) <= tol for p in merged_peaks):
            tp += 1
    fp = len(merged_peaks) - tp
    fn = len(known_positions) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return tp, fp, fn, precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="Path to genome FASTA (e.g., sequence.fasta)")
    ap.add_argument("--model", required=True,
                    help="Path or HF model id. Use your fine-tuned DNABERT checkpoint folder or e.g. zhihan1996/DNA_bert_6")
    ap.add_argument("--outdir", default="scan_out", help="Output directory for scan arrays + csv")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--no_fp16", action="store_true", help="Disable fp16 autocast")
    ap.add_argument("--thresholds", default="0.4,0.6,0.8,0.9", help="Comma thresholds for peaks")
    ap.add_argument("--merge_gap", type=int, default=200)
    ap.add_argument("--tol", type=int, default=100)

    # optional evaluation
    ap.add_argument("--known_tss_csv", default=None,
                    help="Optional CSV file containing a column posTSS (RegulonDB positions).")
    ap.add_argument("--known_tss_col", default="posTSS", help="Column name for TSS positions")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genome = load_genome_fasta(args.fasta)
    genome_len = len(genome)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_safetensors=True
    )

    model.to(device)

    cfg = ScanConfig(
        window=args.window,
        step=args.step,
        kmer=6,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        use_fp16=(not args.no_fp16)
    )

    out_prefix = os.path.join(args.outdir, f"dnabert_w{args.window}_s{args.step}")

    centers, probs = dnabert_scan_genome(
        model=model,
        tokenizer=tokenizer,
        genome=genome,
        cfg=cfg,
        out_prefix=out_prefix,
        device=device,
        resume=True
    )

    raw_csv = os.path.join(args.outdir, f"dnabert_scan_raw_w{args.window}_s{args.step}.csv")
    pd.DataFrame({"center": centers, "prob": probs}).to_csv(raw_csv, index=False)

    known_positions = None
    if args.known_tss_csv is not None:
        df_known = pd.read_csv(args.known_tss_csv, sep="\t", skiprows=33)
        known_positions = df_known[args.known_tss_col].dropna().astype(int).tolist()
        known_positions = [p for p in known_positions if 0 <= p < genome_len]

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    eval_rows = []

    for thr in thresholds:
        predicted = centers[probs >= thr].tolist()
        merged = merge_peaks(predicted, merge_gap=args.merge_gap)

        row = {
            "Window": args.window,
            "Step": args.step,
            "Threshold": thr,
            "MergedPeaks": len(merged),
        }

        if known_positions is not None:
            tp, fp, fn, prec, rec, f1 = evaluate_peaks(
                merged_peaks=merged,
                known_positions=known_positions,
                tol=args.tol
            )
            row.update({
                "TP": tp, "FP": fp, "FN": fn,
                "Precision": prec, "Recall": rec, "F1": f1
            })

        eval_rows.append(row)

    eval_df = pd.DataFrame(eval_rows).sort_values(["Threshold"])
    eval_csv = os.path.join(args.outdir, f"dnabert_scan_eval_w{args.window}_s{args.step}.csv")
    eval_df.to_csv(eval_csv, index=False)


if __name__ == "__main__":
    main()




