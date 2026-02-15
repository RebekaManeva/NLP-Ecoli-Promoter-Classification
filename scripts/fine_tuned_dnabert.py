"""
DNABERT Fine-tuning and Genome Scanning Pipeline

Fine-tunes DNABERT on E. coli promoter data and performs genome-wide scanning.

Example usage:
    python dnabert_finetuning.py \
        --fasta data/sequence.fasta \
        --promoter_tsv data/promoter.tsv \
        --outdir out_dnabert \
        --window 100 \
        --epochs 3

"""

import os
import re
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

DEFAULT_MODEL_ID = "zhihan1996/DNA_bert_6"

# sequence processing
def clean_seq(seq: str) -> str:
    return re.sub(r"[^ATCG]", "", str(seq).upper())


def has_only_atcg(seq: str) -> bool:
    return set(seq) <= {"A", "C", "G", "T"}


def extract_centered_window(genome: str, center_pos: int, window: int) -> Optional[str]:
    half = window // 2
    start = center_pos - half
    end = start + window
    if start < 0 or end > len(genome):
        return None
    seq = genome[start:end]
    if not has_only_atcg(seq):
        return None
    return seq


def seq_to_kmers(seq: str, k: int = 6) -> str:
    seq = clean_seq(seq)
    if len(seq) < k:
        return seq
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_prauc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def eval_probs(y_true, probs, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= thr).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "ROC_AUC": safe_auc(y_true, probs),
        "PR_AUC": safe_prauc(y_true, probs),
    }


# data loading
def load_genome_fasta(fasta_path: str) -> str:
    record = SeqIO.read(fasta_path, "fasta")
    genome = clean_seq(str(record.seq))
    return genome


def load_promoter_tsv_for_tss(promoter_path: str, skiprows: int = 33) -> pd.DataFrame:
    df = pd.read_csv(promoter_path, sep="\t", skiprows=skiprows)

    seq_cols = [c for c in df.columns if "seq" in c.lower()]
    pos_cols = [c for c in df.columns if "pos" in c.lower()]

    if not seq_cols:
        raise ValueError("Could not find a sequence column containing 'seq' in promoter.tsv.")
    if not pos_cols:
        raise ValueError("Could not find a position column containing 'pos' in promoter.tsv.")

    seq_col = seq_cols[0]
    pos_col = pos_cols[0]

    df = df[[seq_col, pos_col]].dropna()
    df[seq_col] = df[seq_col].apply(clean_seq)
    df[pos_col] = df[pos_col].astype(int)

    df = df.rename(columns={seq_col: "sequence", pos_col: "posTSS"})
    df["label"] = 1
    return df


# negative sampling
def sample_genome_negatives(
        genome: str,
        known_centers: List[int],
        n_samples: int,
        window: int,
        min_distance: int,
        seed: int = 42
) -> Tuple[List[int], List[str]]:
    rng = random.Random(seed)
    centers = np.array(sorted(set(known_centers)), dtype=np.int64)

    def far_enough(pos: int) -> bool:
        idx = np.searchsorted(centers, pos)
        candidates = []
        if idx > 0:
            candidates.append(abs(pos - centers[idx - 1]))
        if idx < len(centers):
            candidates.append(abs(pos - centers[idx]))
        if not candidates:
            return True
        return min(candidates) >= min_distance

    neg_centers, neg_seqs = [], []
    attempts = 0
    max_attempts = n_samples * 400

    half = window // 2
    while len(neg_seqs) < n_samples and attempts < max_attempts:
        attempts += 1
        pos = rng.randint(half, len(genome) - half - 1)
        if not far_enough(pos):
            continue
        seq = extract_centered_window(genome, pos, window)
        if seq is None:
            continue
        neg_centers.append(pos)
        neg_seqs.append(seq)

    return neg_centers, neg_seqs


def build_training_df_from_genome(
        genome: str,
        known_tss: List[int],
        window: int,
        neg_per_pos: float,
        min_distance: int,
        seed: int
) -> pd.DataFrame:
    # positives: centered at TSS
    pos_centers, pos_seqs = [], []
    for tss in known_tss:
        seq = extract_centered_window(genome, tss, window)
        if seq is not None:
            pos_centers.append(tss)
            pos_seqs.append(seq)

    pos_df = pd.DataFrame({"sequence": pos_seqs, "label": 1, "center_pos": pos_centers})

    n_neg = int(len(pos_df) * neg_per_pos)
    neg_centers, neg_seqs = sample_genome_negatives(
        genome, pos_df["center_pos"].tolist(), n_neg, window, min_distance, seed=seed
    )
    neg_df = pd.DataFrame({"sequence": neg_seqs, "label": 0, "center_pos": neg_centers})

    df = pd.concat([pos_df, neg_df], ignore_index=True).drop_duplicates(subset=["sequence"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# DNABERT fine-tuning
def make_hf_dataset(df: pd.DataFrame, window: int, kmer: int, tokenizer) -> Dataset:
    df_local = df[["sequence", "label"]].copy()

    def norm(s: str) -> str:
        s = clean_seq(s)
        if len(s) >= window:
            return s[:window]
        return s + ("A" * (window - len(s)))

    df_local["sequence"] = df_local["sequence"].apply(norm)
    df_local["text"] = df_local["sequence"].apply(lambda s: seq_to_kmers(s, kmer))

    ds = Dataset.from_pandas(df_local[["text", "label"]]).rename_column("label", "labels")

    max_len = window - kmer + 1

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len)

    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")
    return ds


def finetune_dnabert(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_id_or_path: str,
        out_dir: str,
        window: int,
        kmer: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int
) -> Tuple[str, dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id_or_path, num_labels=2, trust_remote_code=True
    )

    train_ds = make_hf_dataset(train_df, window, kmer, tokenizer)
    val_ds = make_hf_dataset(val_df, window, kmer, tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (exps / exps.sum(axis=1, keepdims=True))[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "roc_auc": safe_auc(labels, probs),
            "pr_auc": safe_prauc(labels, probs),
        }

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    pred = trainer.predict(val_ds)
    logits = pred.predictions
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = (exps / exps.sum(axis=1, keepdims=True))[:, 1]
    labels = pred.label_ids.astype(int)

    metrics = eval_probs(labels, probs, thr=0.5)
    return out_dir, metrics


# enome scanning
@dataclass
class ScanConfig:
    window: int = 100
    step: int = 10
    kmer: int = 6
    batch_size: int = 32
    chunk_size: int = 50000
    use_fp16: bool = True


def generate_centers(genome_len: int, window: int, step: int) -> List[int]:
    half = window // 2
    return list(range(half, genome_len - half, step))


def get_window_centered(genome: str, center: int, window: int) -> str:
    half = window // 2
    start = center - half
    end = start + window
    return genome[start:end]


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
            print("[WARN] Resume files mismatch; restarting.")
            start_index = 0
            centers_all, probs_all = [], []

        print(f"[RESUME] Loaded {len(probs_all)} predictions. Continue from {start_index}/{total}...")

    model.eval()

    def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(-1) == 1:
            return torch.sigmoid(logits).squeeze(-1)
        elif logits.dim() == 2 and logits.size(-1) >= 2:
            return torch.softmax(logits, dim=-1)[:, 1]
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    use_amp = cfg.use_fp16 and device.type == "cuda"
    pbar = tqdm(range(start_index, total, cfg.chunk_size), desc="Scanning chunks", unit="chunk")

    for chunk_start in pbar:
        chunk_end = min(chunk_start + cfg.chunk_size, total)
        chunk_centers = centers[chunk_start:chunk_end]

        windows = []
        valid_centers = []
        for c in chunk_centers:
            w = get_window_centered(genome, c, cfg.window)
            if not has_only_atcg(w):
                continue
            windows.append(seq_to_kmers(w, cfg.kmer))
            valid_centers.append(c)

        if not windows:
            save_meta(meta_path, cfg, chunk_end, total)
            continue

        chunk_probs: List[float] = []
        for i in tqdm(range(0, len(windows), cfg.batch_size), leave=False, desc="Batches", unit="batch"):
            batch_text = windows[i:i + cfg.batch_size]
            enc = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        out = model(**enc)
                        prob = logits_to_prob(out.logits)
                else:
                    out = model(**enc)
                    prob = logits_to_prob(out.logits)

            chunk_probs.extend(prob.detach().float().cpu().numpy().tolist())

        centers_all.extend(valid_centers)
        probs_all.extend(chunk_probs)

        np.save(centers_path, np.asarray(centers_all, dtype=np.int64))
        np.save(probs_path, np.asarray(probs_all, dtype=np.float32))
        save_meta(meta_path, cfg, chunk_end, total)

        pbar.set_postfix({"saved": len(probs_all)})

    return np.asarray(centers_all, dtype=np.int64), np.asarray(probs_all, dtype=np.float32)


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


def evaluate_peaks(merged_peaks: List[int], known_positions: List[int], tol: int = 100):
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
    ap.add_argument("--fasta", required=True, help="Genome FASTA (single record)")
    ap.add_argument("--promoter_tsv", required=True, help="promoter.tsv (RegulonDB) containing TSS positions")
    ap.add_argument("--skiprows", type=int, default=33, help="skiprows for promoter.tsv")
    ap.add_argument("--base_model", default=DEFAULT_MODEL_ID, help="HF model id or local path (base)")
    ap.add_argument("--outdir", default="out_dnabert", help="Output directory")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--kmer", type=int, default=6)

    # dataset construction
    ap.add_argument("--neg_per_pos", type=float, default=1.0)
    ap.add_argument("--min_distance", type=int, default=1000)

    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)

    # scanning
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--scan_batch", type=int, default=32)
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--no_fp16", action="store_true")

    # evaluation thresholds
    ap.add_argument("--thresholds", default="0.5,0.7,0.9")
    ap.add_argument("--merge_gap", type=int, default=200)
    ap.add_argument("--tol", type=int, default=100)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genome = load_genome_fasta(args.fasta)
    genome_len = len(genome)

    prom_df = load_promoter_tsv_for_tss(args.promoter_tsv, skiprows=args.skiprows)
    known_tss = prom_df["posTSS"].dropna().astype(int).tolist()
    known_tss = [p for p in known_tss if 0 <= p < genome_len]

    full_df = build_training_df_from_genome(
        genome=genome,
        known_tss=known_tss,
        window=args.window,
        neg_per_pos=args.neg_per_pos,
        min_distance=args.min_distance,
        seed=args.seed
    )

    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        stratify=full_df["label"].values,
        random_state=args.seed
    )

    finetuned_dir = os.path.join(args.outdir, f"dnabert_finetuned_w{args.window}")
    print("Fine-tuning DNABERT ->", finetuned_dir)

    finetuned_path, val_metrics = finetune_dnabert(
        train_df=train_df,
        val_df=val_df,
        model_id_or_path=args.base_model,
        out_dir=finetuned_dir,
        window=args.window,
        kmer=args.kmer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(finetuned_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_path, trust_remote_code=True)
    model.to(device)

    scan_out_prefix = os.path.join(args.outdir, f"scan_w{args.window}_s{args.step}")
    cfg = ScanConfig(
        window=args.window,
        step=args.step,
        kmer=args.kmer,
        batch_size=args.scan_batch,
        chunk_size=args.chunk_size,
        use_fp16=(not args.no_fp16)
    )

    centers, probs = dnabert_scan_genome(
        model=model,
        tokenizer=tokenizer,
        genome=genome,
        cfg=cfg,
        out_prefix=scan_out_prefix,
        device=device,
        resume=True
    )

    raw_csv = os.path.join(args.outdir, f"scan_raw_w{args.window}_s{args.step}.csv")
    pd.DataFrame({"center": centers, "prob": probs}).to_csv(raw_csv, index=False)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    rows = []
    for thr in thresholds:
        predicted = centers[probs >= thr].tolist()
        merged = merge_peaks(predicted, merge_gap=args.merge_gap)
        tp, fp, fn, prec, rec, f1 = evaluate_peaks(merged, known_tss, tol=args.tol)
        rows.append({
            "Threshold": thr,
            "MergedPeaks": len(merged),
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec, "F1": f1
        })

    eval_df = pd.DataFrame(rows).sort_values("Threshold")
    eval_csv = os.path.join(args.outdir, f"scan_eval_w{args.window}_s{args.step}.csv")
    eval_df.to_csv(eval_csv, index=False)


if __name__ == "__main__":
    main()


