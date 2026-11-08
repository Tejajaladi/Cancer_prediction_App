#!/usr/bin/env python
# coding: utf-8

# Deep Learning-Based Lung Cancer Classification from CT Images Using Fine-Tuned EfficientNet-B0 with Patient-Level LOOCV Validation
# 
# This project presents an automated deep learning framework for multi-class lung cancer classification from computed tomography (CT) images using a fine-tuned EfficientNet-B0 convolutional neural network. The model is trained to classify lung cancer subtypes into four categories: Adenocarcinoma, Small Cell Carcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma.
# 
# The proposed approach uses a two-stage transfer learning strategy, where the EfficientNet-B0 backbone is initially frozen for classifier warm-up and subsequently fine-tuned end-to-end using medical CT data. The training pipeline incorporates CT-specific image normalization, advanced data augmentation, MixUp regularization, class-balanced loss weighting, label smoothing, and adaptive learning-rate scheduling to improve generalization and handle class imbalance.
# 
# To ensure reliable evaluation, the framework performs conventional train-validation-test evaluation along with Leave-One-Patient-Out Cross-Validation (LOOCV) on patient-specific CT samples, reducing data leakage and providing a more clinically realistic performance estimate. The system evaluates performance using accuracy, macro F1-score, ROC-AUC, precision, recall, and confusion matrix analysis.
# 
# The final model provides a robust AI-assisted solution for lung cancer subtype identification from CT images and demonstrates the potential of deep learning techniques in computer-aided diagnosis and medical image analysis.


import os, re, gc, json, math, time, random, warnings, sys, subprocess, importlib.util
from pathlib import Path
warnings.filterwarnings("ignore")

# HuggingFace token──
HF_TOKEN = "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" 
os.environ["HF_TOKEN"]             = HF_TOKEN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_XET"]   = "1"
os.environ["HF_HOME"]              = "/kaggle/working/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"]= "/kaggle/working/hf_cache"

# Auto-install missing packages 
def ensure(name, pkg=None):
    if importlib.util.find_spec(name) is None:
        subprocess.run([sys.executable,"-m","pip","install","-q",pkg or name],
                       check=False)

ensure("timm",            "timm>=1.0.3")
ensure("huggingface_hub", "huggingface_hub>=0.23.0")
ensure("safetensors",     "safetensors")

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix, roc_auc_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from huggingface_hub import HfApi, upload_folder, snapshot_download
from huggingface_hub import login as hf_login

try:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ HF login OK")
except Exception as e:
    print(f"⚠️  HF login: {e}")


# CONFIGURATION

DATA_ROOT = Path("/kaggle/input/datasets/tejajaladi/"
                 "cancer-image-dataset-part-5-preprocessed/part5_preprocessed")
WORK_DIR  = Path("/kaggle/working/effb0_v61")
WORK_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO_MAIN  = "aimlteam5/lung-cancer-ct-efficientnet-b0"
HF_REPO_LOOCV = "aimlteam5/lung-cancer-ct-efficientnet-b0-loocv"

ALL_CLASS_NAMES = ["Adenocarcinoma", "SmallCell", "LargeCell", "Squamous"]
C2I             = {n: i for i, n in enumerate(ALL_CLASS_NAMES)}
NUM_CLASSES     = 4
SEED            = 42
IMG_SIZE        = 224

# Hyperparameters (research-backed) 
PHASE1_EPOCHS  = 5       # head-only warmup (PMC12459604)
PHASE1_LR      = 1e-3    # safe for frozen backbone
PHASE1_BS      = 32      # CT papers use 16-32 (PMC12459604 = 32)

PHASE2_EPOCHS  = 25      # full fine-tune (PMC12210548)
PHASE2_LR      = 2e-5    # backbone (PMC12210548)
PHASE2_HEAD_LR = 2e-4    # head runs 10x faster
PHASE2_BS      = 32

LOOCV_EPOCHS   = 8
LOOCV_LR       = 5e-5
LOOCV_BS       = 16      # smaller: LargeCell folds have few slices

NUM_WORKERS    = 2
USE_AMP        = True
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1     # PMC12210548 recommends for medical CT
DROPOUT        = 0.45    # between PMC12459604(0.5) and PMC12210548(0.4)
EARLY_STOP_PAT = 7
LOOCV_ES_PAT   = 4
HF_PUSH_EVERY  = 3
LOCAL_SAVE_STEP= 300
RESUME         = True
MONITOR        = "auc_macro_ovr"
MIXUP_ALPHA    = 0.2     # MixUp for minority class generalization

# CT-specific normalization
# Derived from lung HU window [-1000,+400] → 0-255
# Source: LIDC-IDRI preprocessing paper
CT_MEAN   = [0.282, 0.282, 0.282]
CT_STD    = [0.197, 0.197, 0.197]
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD  = [0.229, 0.224, 0.225]

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

api = HfApi(token=HF_TOKEN)
for repo_id in [HF_REPO_MAIN, HF_REPO_LOOCV]:
    try:
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=True, exist_ok=True)
        print(f"✅ HF repo → https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  HF repo {repo_id}: {e}")


# PATH RESOLUTION  (THE KEY FIX)
# manifest stores: /kaggle/working/part5_preprocessed/images/train/CT/Class/f.png
# actual location: DATA_ROOT / images/train/CT/Class/f.png

def resolve(p: str) -> str:
    p = str(p)
    if os.path.exists(p):
        return p
    # Strip everything up to and including 'part5_preprocessed/'
    suffix    = re.sub(r'^.*/part5_preprocessed/', '', p)
    candidate = str(DATA_ROOT / suffix)
    return candidate


# LOAD MANIFEST + VALIDATE PATHS

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

PNG_COL     = "png_path"
STATUS_COL  = "png_status"
SPLIT_COL   = "split"
CLASS_COL   = "cancer_type"
PATIENT_COL = "short_id"

df_raw   = pd.read_csv(DATA_ROOT / "manifest_CT.csv")
folds_df = pd.read_csv(DATA_ROOT / "loocv_folds.csv")

df = df_raw[df_raw[STATUS_COL].astype(str).str.lower() == "ok"].copy()
df = df.dropna(subset=[PNG_COL, CLASS_COL, SPLIT_COL, PATIENT_COL]).copy()
df[PNG_COL] = df[PNG_COL].astype(str).map(resolve)
df["label"] = df[CLASS_COL].map(C2I)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# Path existence check──
_probe = df.sample(min(500, len(df)), random_state=0)
_found = _probe[PNG_COL].map(os.path.exists).sum()
_pct   = 100 * _found / len(_probe)
print(f"Path check: {_found}/{len(_probe)} found ({_pct:.1f}%)")
if _found == 0:
    raise RuntimeError(
        "❌ ZERO PNG files found. Check DATA_ROOT or re-run preprocessing.")
elif _pct < 80:
    print(f"⚠️  Only {_pct:.1f}% PNGs found. Filtering to accessible only.")
    df = df[df[PNG_COL].map(os.path.exists)].copy()
else:
    print("✅ Paths OK")

# Pixel quality + auto-detect normalization─
_accessible = df[df[PNG_COL].map(os.path.exists)]
_stds, _means = [], []
for _, row in _accessible.sample(min(30, len(_accessible)), random_state=1).iterrows():
    try:
        arr = np.array(Image.open(row[PNG_COL]).convert("L"), dtype=np.float32)
        _stds.append(arr.std()); _means.append(arr.mean())
    except: pass

_avg_std  = float(np.mean(_stds))  if _stds  else 0.0
_avg_mean = float(np.mean(_means)) if _means else 128.0

print(f"Pixel stats: mean={_avg_mean:.1f}  std={_avg_std:.1f}")
if _avg_std < 5:
    raise RuntimeError(
        f"❌ avg pixel std={_avg_std:.1f} — images are blank. "
        "HU windowing was not applied. Re-run preprocessing.")

USE_CT_NORM = (_avg_mean < 170 and _avg_std < 65)
NORM_MEAN   = CT_MEAN   if USE_CT_NORM else IMGNET_MEAN
NORM_STD    = CT_STD    if USE_CT_NORM else IMGNET_STD
print(f"Normalization: {'CT-specific (μ=0.282)' if USE_CT_NORM else 'ImageNet'}")

# Splits─
_sp      = df[SPLIT_COL].str.lower()
train_df = df[_sp=="train"].copy().reset_index(drop=True)
val_df   = df[_sp=="val"  ].copy().reset_index(drop=True)
test_df  = df[_sp=="test" ].copy().reset_index(drop=True)
loocv_df = df[_sp=="loocv"].copy().reset_index(drop=True)

print(f"\nTrain : {len(train_df):,}  {train_df[CLASS_COL].value_counts().to_dict()}")
print(f"Val   : {len(val_df):,}  {val_df[CLASS_COL].value_counts().to_dict()}")
print(f"Test  : {len(test_df):,}  {test_df[CLASS_COL].value_counts().to_dict()}")
print(f"LOOCV : {len(loocv_df):,}  {loocv_df[CLASS_COL].value_counts().to_dict()}")

# LOOCV fold info─
# loocv_folds.csv columns: fold_id, test_patient, n_test_slices
FOLD_ID_COL  = "fold_id"
FOLD_PAT_COL = "test_patient"
FOLD_N_COL   = "n_test_slices"
print(f"\nLOOCV folds:\n{folds_df.to_string(index=False)}")

# LargeCell loocv_source paths ─
# loocv_source/CT/LargeCell/ contains the 808 LargeCell slices
# Their png_path in the manifest has split=loocv
# We need these for each fold's test set
LOOCV_SOURCE = DATA_ROOT / "loocv_source" / "CT" / "LargeCell"
print(f"\nloocv_source exists: {LOOCV_SOURCE.exists()}")
if LOOCV_SOURCE.exists():
    lc_pngs = list(LOOCV_SOURCE.rglob("*.png"))
    print(f"LargeCell PNGs in source: {len(lc_pngs)}")


# TRANSFORMS Image Augmentation

train_tfm = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.3),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.15, contrast=0.20),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

eval_tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])


# DATASET

class LungCTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df  = df.reset_index(drop=True)
        self.tfm = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[PNG_COL])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return mid-gray image (not black) to avoid all-zero inputs
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=128)
        if self.tfm:
            img = self.tfm(img)
        return img, int(row["label"])


# MIXUP (improves generalization for minority classes)

def mixup_batch(imgs, lbls, alpha=MIXUP_ALPHA):
    """MixUp augmentation — blends two random images and their labels."""
    if alpha <= 0:
        return imgs, F.one_hot(lbls, NUM_CLASSES).float()
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)          # keep dominant sample dominant
    bs  = imgs.size(0)
    idx = torch.randperm(bs, device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    y_a   = F.one_hot(lbls, NUM_CLASSES).float()
    y_b   = F.one_hot(lbls[idx], NUM_CLASSES).float()
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed, mixed_y

def mixup_loss(logits, mixed_y, weight=None):
    """Soft cross-entropy for MixUp targets."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    if weight is not None:
        w = weight.to(logits.device)
        log_probs = log_probs * w.unsqueeze(0)
    return -(mixed_y * log_probs).sum(dim=-1).mean()


# CLASS WEIGHTS — Effective Number (Cui et al. 2019)

def make_class_weights(label_series: pd.Series) -> torch.Tensor:
    counts = label_series.value_counts().to_dict()
    beta   = 0.9999
    w      = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    present = [i for i in range(NUM_CLASSES) if counts.get(i, 0) > 0]
    for i in present:
        en   = (1.0 - beta ** counts[i]) / (1.0 - beta)
        w[i] = 1.0 / (en + 1e-9)
    # Normalize: mean of present weights = 1.0
    w_mean = w[present].mean()
    if w_mean > 0:
        w = w / w_mean
    print(f"  class_weights: { {ALL_CLASS_NAMES[i]: round(float(w[i]),4) for i in range(NUM_CLASSES)} }")
    return w


# DATALOADERS

def make_loader(ds, bs, shuffle=False, pin=True):
    kw = dict(dataset=ds, batch_size=bs, shuffle=shuffle,
              num_workers=NUM_WORKERS, pin_memory=pin,
              drop_last=shuffle, persistent_workers=False)
    if NUM_WORKERS > 0:
        kw["prefetch_factor"] = 2
    return DataLoader(**kw)

def build_loaders(tr_df, va_df, te_df, bs):
    cw  = make_class_weights(tr_df["label"])
    tr  = make_loader(LungCTDataset(tr_df, train_tfm), bs, shuffle=True)
    va  = make_loader(LungCTDataset(va_df, eval_tfm),  bs, shuffle=False)
    te  = make_loader(LungCTDataset(te_df, eval_tfm),  bs, shuffle=False)
    return tr, va, te, cw


# MODEL

def build_model(device, pretrained=True):
    m = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
        drop_rate=DROPOUT,
    )
    return m.to(device)

def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone FROZEN | trainable: {n:,} params")

def unfreeze_all(model):
    """Unfreeze all layers for end-to-end fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All layers UNFROZEN | trainable: {n:,} params")


# LOSS

def build_criterion(cw, device):
    return nn.CrossEntropyLoss(
        weight=cw.to(device),
        label_smoothing=LABEL_SMOOTH,
        reduction="mean",
    )


# OPTIMIZER + SCHEDULER

def build_opt_sched(model, phase, n_epochs, steps_per_epoch):
    if phase == 1:
        params  = [p for p in model.parameters() if p.requires_grad]
        opt     = torch.optim.AdamW(params, lr=PHASE1_LR,
                                    weight_decay=WEIGHT_DECAY)
        base_lr = PHASE1_LR
    elif phase == 2:
        head_p  = [p for n,p in model.named_parameters() if "classifier" in n]
        back_p  = [p for n,p in model.named_parameters() if "classifier" not in n]
        opt     = torch.optim.AdamW([
            {"params": back_p, "lr": PHASE2_LR},
            {"params": head_p, "lr": PHASE2_HEAD_LR},
        ], weight_decay=WEIGHT_DECAY)
        base_lr = PHASE2_LR
    else:  # LOOCV
        opt     = torch.optim.AdamW(model.parameters(), lr=LOOCV_LR,
                                    weight_decay=WEIGHT_DECAY)
        base_lr = LOOCV_LR

    total    = n_epochs * steps_per_epoch
    warmup   = max(50, int(0.05 * total))
    min_lr_r = 1e-7 / base_lr

    def lr_lambda(step):
        if step < warmup:
            return max(step / max(1, warmup), 0.01)
        prog = (step - warmup) / max(1, total - warmup)
        cos  = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_r + (1.0 - min_lr_r) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


# METRICS

def safe_softmax(x: np.ndarray) -> np.ndarray:
    x   = np.nan_to_num(np.clip(x, -60, 60), nan=0.0)
    e   = np.exp(x - x.max(1, keepdims=True))
    p   = e / (e.sum(1, keepdims=True) + 1e-9)
    return p

def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    probs   = safe_softmax(logits)
    preds   = probs.argmax(1)
    present = sorted(set(int(v) for v in y_true))

    acc  = float(accuracy_score(y_true, preds))
    _,_,f1m,_ = precision_recall_fscore_support(
        y_true, preds, average="macro",    zero_division=0)
    _,_,f1w,_ = precision_recall_fscore_support(
        y_true, preds, average="weighted", zero_division=0)

    auc_mac = auc_wt = float("nan")
    if len(present) >= 2:
        try:
            auc_mac = roc_auc_score(y_true, probs, multi_class="ovr",
                                    average="macro",
                                    labels=list(range(NUM_CLASSES)))
            auc_wt  = roc_auc_score(y_true, probs, multi_class="ovr",
                                    average="weighted",
                                    labels=list(range(NUM_CLASSES)))
        except Exception as ex:
            print(f"  AUC warn: {ex}")

    report = classification_report(
        y_true, preds, labels=present,
        target_names=[ALL_CLASS_NAMES[i] for i in present],
        output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=present)

    return dict(
        accuracy=acc, f1_macro=float(f1m), f1_weighted=float(f1w),
        auc_macro_ovr=auc_mac, auc_weighted_ovr=auc_wt,
        per_class=report, confusion_matrix=cm.tolist(),
        present_labels=present, preds=preds, probs=probs,
    )

def safe_json(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if k in {"preds", "probs"}:
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = None if math.isnan(float(v)) else float(v)
        elif isinstance(v, float) and math.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


# CHECKPOINT HELPERS

def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)

def load_ckpt(path: Path):
    if not Path(path).exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ⚠️  load_ckpt {path}: {e}")
        return None

def hf_push(local_dir: Path, repo_id: str, msg: str):
    try:
        upload_folder(repo_id=repo_id, folder_path=str(local_dir),
                      repo_type="model", commit_message=msg,
                      token=HF_TOKEN, ignore_patterns=["*.tmp"])
        print(f"  ↑ HF pushed: {msg}")
    except Exception as e:
        print(f"  ⚠️  HF push: {e}")

def hf_restore(repo_id: str, local_dir: Path) -> bool:
    try:
        snapshot_download(repo_id=repo_id, repo_type="model",
                          local_dir=str(local_dir), token=HF_TOKEN,
                          ignore_patterns=["*.tmp"])
        print(f"  ↓ HF restored: {repo_id}")
        return True
    except Exception as e:
        print(f"  ℹ️  HF restore skipped ({e})")
        return False


# TRAIN ONE EPOCH (with MixUp)

def train_epoch(model, loader, opt, sched, crit, cw,
                device, scaler, ep, run_dir, gstep, meta,
                use_mixup=True):
    model.train()
    opt.zero_grad(set_to_none=True)
    losses, skip = [], 0

    for bi, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            if use_mixup and random.random() < 0.5:
                mixed_imgs, mixed_y = mixup_batch(imgs, lbls)
                logits = model(mixed_imgs)
                loss   = mixup_loss(logits, mixed_y, weight=cw)
            else:
                logits = model(imgs)
                loss   = crit(logits, lbls)

        if torch.isnan(loss) or torch.isinf(loss):
            opt.zero_grad(set_to_none=True); skip += 1; continue

        scaler.scale(loss).backward()
        losses.append(float(loss.item()))

        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step(); gstep += 1

        # Periodic local save (safety net for Kaggle 9-hour limit)
        if (bi + 1) % LOCAL_SAVE_STEP == 0:
            atomic_save({**meta, "epoch": ep, "batch": bi,
                         "epoch_completed": False,
                         "model": model.state_dict(),
                         "optimizer": opt.state_dict(),
                         "scheduler": sched.state_dict(),
                         "scaler": scaler.state_dict(),
                         "global_step": gstep},
                        run_dir / "latest" / "checkpoint.pt")

    if skip:
        print(f"  ⚠️  {skip} NaN-loss batches skipped")
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, gstep


# EVAL ONE EPOCH

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    losses, logits_all, y_all = [], [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            lg   = model(imgs)
            loss = crit(lg, lbls)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            losses.append(float(loss.item()))
        logits_all.append(lg.float().cpu().numpy())
        y_all.append(lbls.cpu().numpy())

    m = compute_metrics(np.concatenate(y_all), np.concatenate(logits_all))
    m["loss"] = float(np.mean(losses)) if losses else float("nan")
    return m


# PHASE RUNNER (shared by phase1, phase2, LOOCV)

def run_phase(model, tr_ld, va_ld, crit, cw, device,
              phase_name, phase_num, n_epochs, run_dir,
              repo_id=None, history=None, gstep=0,
              start_ep=0, best_metric=-1.0, best_epoch=-1, no_imp=0,
              es_patience=EARLY_STOP_PAT,
              saved_opt=None, saved_sched=None, saved_scaler=None):

    if history is None:
        history = []

    opt, sched = build_opt_sched(model, phase_num, n_epochs, len(tr_ld))
    scaler     = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for state, obj in [(saved_opt, opt), (saved_sched, sched),
                       (saved_scaler, scaler)]:
        if state:
            try: obj.load_state_dict(state)
            except: pass

    use_mixup = (phase_num == 2)   # MixUp only during full fine-tune

    for ep in range(start_ep, n_epochs):
        t0  = time.time()
        meta = dict(phase=phase_name, best_metric=best_metric,
                    best_epoch=best_epoch, history=history)

        tr_loss, gstep = train_epoch(
            model, tr_ld, opt, sched, crit, cw,
            device, scaler, ep, run_dir, gstep, meta, use_mixup)

        val_m = eval_epoch(model, va_ld, crit, device)
        cur   = val_m[MONITOR]
        imp   = (not math.isnan(cur)) and (cur > best_metric)

        # Log row─
        row = dict(
            phase=phase_name, epoch=ep,
            tr_loss=None if math.isnan(tr_loss) else round(tr_loss, 5),
            val_loss=None if math.isnan(val_m["loss"]) else round(val_m["loss"], 5),
            val_acc=round(val_m["accuracy"], 4),
            val_f1m=round(val_m["f1_macro"], 4),
            val_auc=None if math.isnan(cur) else round(cur, 4),
        )
        history.append(row)

        # Checkpoint─
        ckpt = dict(phase=phase_name, epoch=ep, epoch_completed=True,
                    model=model.state_dict(), optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                    global_step=gstep, best_metric=best_metric,
                    best_epoch=best_epoch, history=history)
        atomic_save(ckpt, run_dir / "latest" / "checkpoint.pt")

        if imp:
            best_metric = cur; best_epoch = ep; no_imp = 0
            bdir = run_dir / "best"; bdir.mkdir(exist_ok=True)
            atomic_save({**ckpt, "best_metric": best_metric,
                         "best_epoch": best_epoch},
                        bdir / "checkpoint.pt")
            torch.save(model.state_dict(), bdir / "pytorch_model.bin")
            with open(bdir / "val_metrics.json", "w") as f:
                json.dump(safe_json(val_m), f, indent=2, default=str)
            if repo_id:
                hf_push(bdir, repo_id,
                        f"best {phase_name} ep={ep} auc={best_metric:.4f}")
        else:
            no_imp += 1

        # Periodic HF push of latest
        if repo_id and (ep + 1) % HF_PUSH_EVERY == 0:
            hf_push(run_dir / "latest", repo_id,
                    f"latest {phase_name} ep={ep}")

        # Print
        elapsed  = (time.time() - t0) / 60
        flag_str = "✅ BEST" if imp else f"no_imp {no_imp}/{es_patience}"
        auc_s    = f"{cur:.4f}" if not math.isnan(cur) else "nan"
        acc_s    = f"{val_m['accuracy']:.4f}"
        print(f"  [{phase_name}] ep {ep:02d}/{n_epochs-1} | "
              f"tr={tr_loss:.4f} | "
              f"val_loss={val_m['loss']:.4f} | "
              f"acc={acc_s} | f1={val_m['f1_macro']:.4f} | "
              f"auc={auc_s} | {flag_str} | {elapsed:.1f}m")

        gc.collect(); torch.cuda.empty_cache()

        if no_imp >= es_patience:
            print(f"  ⏹  Early stop [{phase_name}] ep={ep}. "
                  f"Best ep={best_epoch}.")
            break

    return best_metric, best_epoch, history, gstep


# TEST EVALUATION (final)

def evaluate_test(model, te_ld, crit, device, out_dir: Path, tag="test"):
    m = eval_epoch(model, te_ld, crit, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(safe_json(m), f, indent=2, default=str)
    print(f"\n{'─'*60}")
    print(f"  {tag.upper()} RESULTS:")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  F1 Macro  : {m['f1_macro']:.4f}")
    print(f"  AUC (OvR) : {m['auc_macro_ovr']:.4f}")
    present = m["present_labels"]
    for i in present:
        cls  = ALL_CLASS_NAMES[i]
        info = m["per_class"].get(cls, {})
        print(f"    {cls:<22}: "
              f"prec={info.get('precision',0):.3f} "
              f"rec={info.get('recall',0):.3f} "
              f"f1={info.get('f1-score',0):.3f} "
              f"n={int(info.get('support',0))}")
    print(f"  CM:\n{np.array(m['confusion_matrix'])}")
    print(f"{'─'*60}")
    return m


# MAIN TRAINING

def run_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"MAIN TRAINING | device={device}")
    print(f"{'='*72}")

    run_dir = WORK_DIR / "main"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build loaders──
    print("\nBuilding dataloaders...")
    tr_ld, va_ld, te_ld, cw = build_loaders(
        train_df, val_df, test_df, PHASE1_BS)
    print(f"  train batches={len(tr_ld)}, val batches={len(va_ld)}, "
          f"test batches={len(te_ld)}")

    # Try to restore from HF ─
    if RESUME:
        hf_restore(HF_REPO_MAIN, run_dir / "latest")

    # Load or build model──
    print("\nBuilding model (EfficientNet-B0, pretrained=ImageNet)...")
    model = build_model(device)
    crit  = build_criterion(cw, device)

    # Determine start state
    phase_start = 1
    start_ep    = 0
    best_metric = -1.0
    best_epoch  = -1
    history     = []
    gstep       = 0
    no_imp      = 0
    saved_opt = saved_sched = saved_scaler = None

    ckpt_path = run_dir / "latest" / "checkpoint.pt"
    ckpt      = load_ckpt(ckpt_path)
    if ckpt:
        phase_start = 1 if ckpt.get("phase","phase1").startswith("phase1") else 2
        start_ep    = ckpt.get("epoch", 0) + (1 if ckpt.get("epoch_completed") else 0)
        best_metric = ckpt.get("best_metric", -1.0)
        best_epoch  = ckpt.get("best_epoch",  -1)
        history     = ckpt.get("history",     [])
        gstep       = ckpt.get("global_step", 0)
        saved_opt   = ckpt.get("optimizer")
        saved_sched = ckpt.get("scheduler")
        saved_scaler= ckpt.get("scaler")
        try:
            model.load_state_dict(ckpt["model"])
            print(f"  ✅ Resumed from {ckpt['phase']} ep={ckpt.get('epoch',0)}, "
                  f"best_auc={best_metric:.4f}")
        except Exception as e:
            print(f"  ⚠️  model state load failed ({e}), starting fresh")
            model = build_model(device)
            phase_start = 1; start_ep = 0
            best_metric = -1.0; best_epoch = -1; history = []
    else:
        print("  No checkpoint found — starting fresh")

    
    # PHASE 1 — Freeze backbone, train head only
    
    if phase_start == 1:
        print(f"\n{'─'*60}")
        print(f"PHASE 1 — Head-only warmup ({PHASE1_EPOCHS} epochs)")
        freeze_backbone(model)
        crit_p1 = build_criterion(cw, device)

        best_metric, best_epoch, history, gstep = run_phase(
            model=model, tr_ld=tr_ld, va_ld=va_ld,
            crit=crit_p1, cw=cw, device=device,
            phase_name="phase1", phase_num=1,
            n_epochs=PHASE1_EPOCHS, run_dir=run_dir,
            repo_id=HF_REPO_MAIN, history=history,
            gstep=gstep, start_ep=start_ep,
            best_metric=best_metric, best_epoch=best_epoch,
            no_imp=no_imp, es_patience=PHASE1_EPOCHS,  # no early stop in p1
            saved_opt=saved_opt, saved_sched=saved_sched,
            saved_scaler=saved_scaler,
        )
        # Reset for phase 2
        start_ep = 0; no_imp = 0
        saved_opt = saved_sched = saved_scaler = None

    
    # PHASE 2 — Unfreeze all, end-to-end fine-tune
    
    print(f"\n{'─'*60}")
    print(f"PHASE 2 — Full fine-tune ({PHASE2_EPOCHS} epochs, BS={PHASE2_BS})")

    # Reload best phase-1 weights before phase 2
    best_ckpt_path = run_dir / "best" / "checkpoint.pt"
    best_ckpt      = load_ckpt(best_ckpt_path)
    if best_ckpt and best_ckpt.get("phase","").startswith("phase1"):
        try:
            model.load_state_dict(best_ckpt["model"])
            print(f"  Loaded best phase1 weights (auc={best_metric:.4f})")
        except: pass

    # Re-build loaders with phase2 batch size (same, but rebuild to be safe)
    tr_ld2, va_ld2, te_ld2, cw2 = build_loaders(
        train_df, val_df, test_df, PHASE2_BS)
    crit_p2 = build_criterion(cw2, device)

    unfreeze_all(model)

    # Load phase2 checkpoint if resuming mid-phase2
    if phase_start == 2 and saved_opt is not None:
        pass  # already loaded above
    else:
        best_metric = -1.0; best_epoch = -1; no_imp = 0

    best_metric, best_epoch, history, gstep = run_phase(
        model=model, tr_ld=tr_ld2, va_ld=va_ld2,
        crit=crit_p2, cw=cw2, device=device,
        phase_name="phase2", phase_num=2,
        n_epochs=PHASE2_EPOCHS, run_dir=run_dir,
        repo_id=HF_REPO_MAIN, history=history,
        gstep=gstep, start_ep=start_ep if phase_start == 2 else 0,
        best_metric=best_metric, best_epoch=best_epoch,
        no_imp=no_imp, es_patience=EARLY_STOP_PAT,
        saved_opt=saved_opt if phase_start == 2 else None,
        saved_sched=saved_sched if phase_start == 2 else None,
        saved_scaler=saved_scaler if phase_start == 2 else None,
    )

    # Final test evaluation
    best_ckpt = load_ckpt(run_dir / "best" / "checkpoint.pt")
    if best_ckpt:
        try: model.load_state_dict(best_ckpt["model"])
        except: pass

    test_m = evaluate_test(model, te_ld2, crit_p2, device,
                           run_dir / "best", tag="test")

    # Save training history
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    print(f"\nTraining history → {run_dir/'training_history.csv'}")

    # Final HF push
    hf_push(run_dir / "best", HF_REPO_MAIN, "FINAL best model")
    print(f"\n✅ MAIN TRAINING COMPLETE | "
          f"Best AUC={best_metric:.4f} @ ep={best_epoch}")
    return model, crit_p2, cw2


# LOOCV — Leave-One-Patient-Out (LargeCell E-prefix patients)

def run_loocv(base_model, base_crit, base_cw):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loocv_dir = WORK_DIR / "loocv"
    loocv_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print("LOOCV — Leave-One-Patient-Out (LargeCell)")
    print(f"{'='*72}")
    print(f"Folds: {len(folds_df)}")

    fold_results = []

    for _, fold_row in folds_df.iterrows():
        fold_id  = int(fold_row[FOLD_ID_COL])
        test_pat = str(fold_row[FOLD_PAT_COL])
        n_test   = int(fold_row.get(FOLD_N_COL, 0))
        print(f"\n{'─'*60}")
        print(f"FOLD {fold_id} | test_patient={test_pat} | n_test_slices={n_test}")

        fold_run_dir = loocv_dir / f"fold_{fold_id}"
        fold_run_dir.mkdir(exist_ok=True)

        # Build fold train/val sets ─
        # Train: all loocv slices EXCEPT test patient + full train_df
        lc_train = loocv_df[loocv_df[PATIENT_COL] != test_pat]
        lc_test  = loocv_df[loocv_df[PATIENT_COL] == test_pat]

        # Also try to find the test patient in the images/ loocv folder
        fold_test_dir = (DATA_ROOT / "images" / "loocv"
                         / f"fold_{fold_id}" / "test")
        if fold_test_dir.exists():
            sub_dirs  = list(fold_test_dir.iterdir())
            fold_pngs = list(fold_test_dir.rglob("*.png"))
            print(f"  fold test dir PNGs: {len(fold_pngs)}")
            # Build a minimal dataframe from file paths if manifest is sparse
            if len(lc_test) < len(fold_pngs):
                rows = []
                for p in fold_pngs:
                    cls = p.parent.name   # .../CT/LargeCell/file.png
                    if cls not in C2I: cls = p.parts[-2]
                    rows.append({PNG_COL: str(p), CLASS_COL: cls,
                                 "label": C2I.get(cls, -1), PATIENT_COL: test_pat})
                extra = pd.DataFrame(rows)
                extra = extra[extra["label"] >= 0]
                lc_test = pd.concat([lc_test, extra]).drop_duplicates(
                    subset=[PNG_COL]).reset_index(drop=True)

        # Combine with full train set (LargeCell train slices from loocv)
        fold_train = pd.concat([train_df, lc_train]).reset_index(drop=True)
        fold_val   = val_df.copy()   # use global val for early stop signal

        print(f"  fold_train={len(fold_train):,}  "
              f"fold_val={len(fold_val):,}  "
              f"fold_test={len(lc_test):,}")

        if len(lc_test) == 0:
            print(f"  ⚠️  No test slices for fold {fold_id} — skipping")
            continue

        # DataLoaders
        fold_cw = make_class_weights(fold_train["label"])
        fold_crit = build_criterion(fold_cw, device)

        fold_tr_ld = make_loader(LungCTDataset(fold_train, train_tfm),
                                 LOOCV_BS, shuffle=True)
        fold_va_ld = make_loader(LungCTDataset(fold_val, eval_tfm), LOOCV_BS)
        fold_te_ld = make_loader(LungCTDataset(lc_test, eval_tfm), LOOCV_BS)

        # Init model from best main checkpoint─
        fold_model = build_model(device, pretrained=False)
        main_best  = WORK_DIR / "main" / "best" / "pytorch_model.bin"
        if main_best.exists():
            fold_model.load_state_dict(
                torch.load(main_best, map_location=device, weights_only=True))
            print("  Loaded main best weights for fold init")
        else:
            print("  ⚠️  Main best not found — using ImageNet init")
            fold_model = build_model(device, pretrained=True)

        # Try resume fold from HF 
        if RESUME:
            hf_restore(HF_REPO_LOOCV, fold_run_dir / "latest")

        fold_ckpt = load_ckpt(fold_run_dir / "latest" / "checkpoint.pt")
        f_start_ep = 0; f_best = -1.0; f_best_ep = -1
        f_history  = []; f_gstep = 0; f_no_imp = 0
        f_opt = f_sched = f_scal = None
        if fold_ckpt:
            try:
                fold_model.load_state_dict(fold_ckpt["model"])
                f_start_ep = fold_ckpt.get("epoch", 0) + 1
                f_best     = fold_ckpt.get("best_metric", -1.0)
                f_best_ep  = fold_ckpt.get("best_epoch",  -1)
                f_history  = fold_ckpt.get("history",     [])
                f_gstep    = fold_ckpt.get("global_step", 0)
                f_opt      = fold_ckpt.get("optimizer")
                f_sched    = fold_ckpt.get("scheduler")
                f_scal     = fold_ckpt.get("scaler")
                print(f"  Resumed fold {fold_id} from ep={fold_ckpt['epoch']}")
            except: pass

        # Run fold training─
        unfreeze_all(fold_model)
        fold_best, fold_best_ep, fold_hist, _ = run_phase(
            model=fold_model, tr_ld=fold_tr_ld, va_ld=fold_va_ld,
            crit=fold_crit, cw=fold_cw, device=device,
            phase_name=f"loocv_fold{fold_id}", phase_num=3,
            n_epochs=LOOCV_EPOCHS, run_dir=fold_run_dir,
            repo_id=HF_REPO_LOOCV, history=f_history,
            gstep=f_gstep, start_ep=f_start_ep,
            best_metric=f_best, best_epoch=f_best_ep,
            no_imp=f_no_imp, es_patience=LOOCV_ES_PAT,
            saved_opt=f_opt, saved_sched=f_sched, saved_scaler=f_scal,
        )

        # Evaluate on fold test set
        # Load best fold weights
        fold_best_path = fold_run_dir / "best" / "checkpoint.pt"
        fold_bc = load_ckpt(fold_best_path)
        if fold_bc:
            try: fold_model.load_state_dict(fold_bc["model"])
            except: pass

        fold_test_m = evaluate_test(
            fold_model, fold_te_ld, fold_crit, device,
            fold_run_dir / "best", tag=f"loocv_fold{fold_id}_test")

        fold_results.append(dict(
            fold=fold_id, patient=test_pat, n_test=n_test,
            accuracy=fold_test_m["accuracy"],
            f1_macro=fold_test_m["f1_macro"],
            auc=fold_test_m.get("auc_macro_ovr", float("nan")),
        ))

        hf_push(fold_run_dir / "best", HF_REPO_LOOCV,
                f"fold{fold_id} test acc={fold_test_m['accuracy']:.4f}")

        del fold_model; gc.collect(); torch.cuda.empty_cache()

    # LOOCV Summary──
    if fold_results:
        fr_df = pd.DataFrame(fold_results)
        fr_df.to_csv(loocv_dir / "loocv_summary.csv", index=False)
        print(f"\n{'='*60}")
        print("LOOCV SUMMARY")
        print(fr_df.to_string(index=False))
        valid_acc  = fr_df["accuracy"].dropna()
        valid_auc  = fr_df["auc"].dropna()
        print(f"\n  Mean accuracy : {valid_acc.mean():.4f} ± {valid_acc.std():.4f}")
        print(f"  Mean AUC      : {valid_auc.mean():.4f} ± {valid_auc.std():.4f}")
        hf_push(loocv_dir, HF_REPO_LOOCV, "FINAL LOOCV summary")
    print(f"\n✅ LOOCV COMPLETE")


# ENTRY POINT

if __name__ == "__main__":
    print(f"\n{'#'*72}")
    print("# EfficientNet-B0 | Lung Cancer CT | v6.1")
    print(f"# GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"# PyTorch: {torch.__version__} | timm: {timm.__version__}")
    print(f"{'#'*72}\n")

    trained_model, main_crit, main_cw = run_main()
    run_loocv(trained_model, main_crit, main_cw)


# # Overall Test Performance (Main Model)
# 
# | Metric | Value |
# |----------|----------|
# | Overall Accuracy | **64.92%** |
# | F1 Macro | **0.3689** |
# | AUC OvR | **N/A** *(nan — single positive class issue)* |
# 
# ## Class-wise Precision, Recall & F1
# 
# | Class | Precision | Recall | F1-Score | Test Slices |
# |---------|---------:|---------:|---------:|---------:|
# | Adenocarcinoma | 0.683 | 0.900 | 0.777 | 26,603 |
# | SmallCell | 0.725 | 0.073 | 0.133 | 3,056 |
# | Squamous | 0.342 | 0.138 | 0.197 | 9,635 |
# 
# ---
# 
# # Training Parameters
# 
# ## Model Architecture
# 
# | Parameter | Value |
# |------------|------------|
# | Backbone | EfficientNet-B0 |
# | Input Resolution | 224 × 224 × 3 |
# | Trainable Parameters | 4,012,672 |
# | Pretrained Initialization | ImageNet Weights |
# | Number of Classes | 3 (main) + LargeCell via LOOCV |
# 
# ---
# 
# ## Two-Phase Fine-Tuning Strategy
# 
# | Setting | Phase 1 (Frozen Backbone) | Phase 2 (Full Fine-Tune) |
# |----------|----------|----------|
# | Layers | Head only unfrozen | All layers unfrozen |
# | Optimizer | Adam (warm-up) | Adam (lower learning rate) |
# | Best Validation Accuracy | ~73.5% | ~75.8% |
# | Best Validation Loss | ~0.9956 | ~0.9208 |
# | Early Stopping | Epoch 12 (patience reached) | Epoch 12 |
# 
# ---
# 
# ## Training Split
# 
# | Split | Number of Slices |
# |----------|----------:|
# | Train | 136,762–136,909 *(varies per LOOCV fold)* |
# | Validation | 29,180 |
# | Test | 39,294 *(main test set)* |
# 
# ---
# 
# ## Class Weights Used During Training
# 
# Class weighting was applied to mitigate class imbalance.
# 
# | Class | Weight |
# |----------|----------:|
# | Adenocarcinoma | 0.188–0.224 |
# | SmallCell | 0.305–0.363 |
# | LargeCell | 3.170–3.303 |
# | Squamous | 0.205–0.244 |
# 
# ### Weighting Strategy
# 
# - **Adenocarcinoma:** down-weighted due to majority-class dominance.
# - **SmallCell:** moderately up-weighted to improve minority-class learning.
# - **LargeCell:** heavily up-weighted because of extreme underrepresentation.
# - **Squamous:** slightly down-weighted to balance class contributions.
# 
# ---
