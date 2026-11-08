#!/usr/bin/env python
# coding: utf-8

# Downloads and validates 7 trained checkpoints (M1, M2, M3).
# Builds a Streamlit application.
# Performs lung cancer histology classification using an ensemble of EfficientNetV2-L, ConvNeXt-Large, and Swin3D-S.
# Supports CT + PET inputs.
# Launches the app with a Cloudflare public URL.




import os
import sys
import shutil
import subprocess
from pathlib import Path

HF_TOKEN_PLACEHOLDER = "ValueHiddenForSecurityReason"

ROOT = Path("/kaggle/working/lung_ensemble_app")
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "hf_cache"
APP_FILE = ROOT / "app.py"

ROOT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("CELL 1: DOWNLOAD MODELS AND CREATE STREAMLIT APP")
print("=" * 90)


# INSTALL PYTHON PACKAGES

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "streamlit>=1.40",
        "huggingface_hub>=0.27",
        "timm==1.0.26",
        "pillow<12.0",
    ],
    check=True,
)

print("Python packages installed.")


# READ HF TOKEN

HF_TOKEN = HF_TOKEN_PLACEHOLDER.strip() or os.environ.get("HF_TOKEN", "").strip()

if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception:
        HF_TOKEN = ""

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF token: enabled")
else:
    print("HF token: not set; downloading public repositories anonymously")


# DOWNLOAD MODELS

from huggingface_hub import hf_hub_download

def complete(path, minimum_mb):
    path = Path(path)
    return path.exists() and path.stat().st_size >= minimum_mb * 1024 * 1024

def download(repo_id, filename, destination, revision="main", minimum_mb=50):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if complete(destination, minimum_mb):
        print(
            f"REUSE | {destination.name:18s} | "
            f"{destination.stat().st_size / 1024**2:.1f} MB"
        )
        return

    print(f"DOWNLOAD | {repo_id}")
    print(f"FILE     | {filename}")

    source = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=HF_TOKEN if HF_TOKEN else None,
        local_dir=str(CACHE_DIR),
    )

    shutil.copy2(source, destination)

    if not complete(destination, minimum_mb):
        raise RuntimeError(f"Incomplete checkpoint: {destination}")

    print(
        f"SAVED    | {destination.name:18s} | "
        f"{destination.stat().st_size / 1024**2:.1f} MB"
    )

# M1: Five EfficientNetV2-L E-LOOCV fold checkpoints
M1_REPO = "tejajaladi/lung-cancer-effnetv2l-m1-e-loocv-v8-hybrid"

M1_FOLDS = [
    ("E0", "778f504d9460"),
    ("E1", "f90773ee1086"),
    ("E2", "95d4eda99811"),
    ("E3", "aa9e624b9c82"),
    ("E4", "26c2c911cbca"),
]

for fold_name, commit_hash in M1_FOLDS:
    download(
        repo_id=M1_REPO,
        revision=commit_hash,
        filename="bestfinal.pt",
        destination=MODEL_DIR / "m1" / f"{fold_name}.pt",
        minimum_mb=300,
    )

# M2: ConvNeXt-Large Direct-4 checkpoint
download(
    repo_id="tejajaladi/lung-cancer-convnext-large-m2-e-loocv-v9-stable",
    revision="main",
    filename="fold_0/direct4/bestfinal.pt",
    destination=MODEL_DIR / "m2" / "direct4.pt",
    minimum_mb=500,
)

# M3: Correct public repository and Direct-4 checkpoint
download(
    repo_id="tejajaladi/lung-cancer-swin3ds-m3-MINT-LOOCV-Aggregate-v7",
    revision="main",
    filename="E0_D4_best.pt",
    destination=MODEL_DIR / "m3" / "E0_D4_best.pt",
    minimum_mb=150,
)

required = [
    *[MODEL_DIR / "m1" / f"E{i}.pt" for i in range(5)],
    MODEL_DIR / "m2" / "direct4.pt",
    MODEL_DIR / "m3" / "E0_D4_best.pt",
]

missing = [str(x) for x in required if not x.exists()]
if missing:
    raise RuntimeError("Missing checkpoint files:\n" + "\n".join(missing))

print("\nAll 7 checkpoints are ready.")


# CREATE STREAMLIT APP

APP_CODE = r'''
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import timm

from pathlib import Path
from PIL import Image
from torchvision.models.video import swin3d_s
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


st.set_page_config(
    page_title="Lung Cancer Ensemble",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path("/kaggle/working/lung_ensemble_app")
MODEL_DIR = ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Adenocarcinoma",
    "Small Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Large Cell Carcinoma",
]

CT_PATH = Path(
    "/kaggle/input/datasets/tejajaladi/"
    "cancer-image-dataset-part-5-preprocessed/"
    "part5/ct_patches/A0001.npy"
)

PET_PATH = Path(
    "/kaggle/input/datasets/tejajaladi/"
    "cancer-image-dataset-part-5-preprocessed/"
    "part5/pet_zeros/A0001.npy"
)



# SHORT SIDEBAR — CLOSED BY DEFAULT

with st.sidebar:
    st.title("🫁 Model details")

    st.markdown("**M1 — EfficientNetV2-L**")
    st.caption("Five 2D folds, top slices, flip TTA")

    st.markdown("**M2 — ConvNeXt-Large**")
    st.caption("2D slices with PET modality embedding")

    st.markdown("**M3 — Swin3D-S**")
    st.caption("3D CT/PET volume with four-view TTA")

    st.divider()
    st.caption("Final = weighted average of M1, M2 and M3")
    st.warning("Research use only. Not a clinical diagnosis.")



# M1 ARCHITECTURE

class M1Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(features + 1),
            nn.Dropout(0.35),
            nn.Linear(features + 1, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 4),
        )

    def forward(self, x, modality):
        z = self.backbone(x)
        modality = modality.float().unsqueeze(1).to(z.device)
        return self.head(torch.cat([z, modality], dim=1))



# M2 ARCHITECTURE

class M2Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_large.fb_in22k_ft_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        features = self.backbone.num_features
        self.mod_emb = nn.Embedding(3, 8)

        self.head = nn.Sequential(
            nn.LayerNorm(features + 8),
            nn.Dropout(0.35),
            nn.Linear(features + 8, 512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 4),
        )

    def forward(self, x, modality):
        z = self.backbone.forward_features(x)
        z = self.backbone.forward_head(z, pre_logits=True)

        if z.ndim > 2:
            z = z.flatten(1)

        modality = self.mod_emb(modality.long().to(z.device))
        return self.head(torch.cat([z, modality], dim=1))



# M3 ARCHITECTURE

class M3Model(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = swin3d_s(weights=None, progress=False)

        self.patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop
        self.features = backbone.features
        self.norm = backbone.norm
        self.avgpool = backbone.avgpool

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(backbone.num_features),
            nn.Dropout(0.35),
            nn.Linear(backbone.num_features, 512),
            nn.GELU(),
            nn.Dropout(0.175),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return self.head(self.avgpool(x))



# CHECKPOINT LOADING

def checkpoint_state(path):
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]

    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]

    elif isinstance(checkpoint, dict):
        state_dict = {
            key: value
            for key, value in checkpoint.items()
            if isinstance(value, torch.Tensor)
        }

    else:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    cleaned = {}

    for key, value in state_dict.items():
        for prefix in ("module.", "model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix):]

        cleaned[key] = value

    return cleaned


def load_strict(model, path, name):
    missing, unexpected = model.load_state_dict(
        checkpoint_state(path),
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint mismatch.\n"
            f"Missing: {missing[:10]}\n"
            f"Unexpected: {unexpected[:10]}"
        )

    return model.eval().to(DEVICE)


@st.cache_resource(show_spinner="Loading M1, M2 and M3 checkpoints...")
def load_models():
    m1_models = []

    for fold in range(5):
        model = load_strict(
            M1Model(),
            MODEL_DIR / "m1" / f"E{fold}.pt",
            f"M1 E{fold}",
        )
        m1_models.append(model)

    m2_model = load_strict(
        M2Model(),
        MODEL_DIR / "m2" / "direct4.pt",
        "M2 Direct-4",
    )

    m3_model = load_strict(
        M3Model(),
        MODEL_DIR / "m3" / "E0_D4_best.pt",
        "M3 E0 Direct-4",
    )

    return m1_models, m2_model, m3_model



# INPUT FUNCTIONS

def normalize_ct(x):
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def normalize_pet(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if x.size and float(x.max()) > 1.0:
        x = np.clip(x, 0.0, 10.0) / 10.0

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def validate_volume(ct, pet):
    if ct.ndim != 4 or ct.shape[0] != 2:
        raise ValueError(f"CT must be [2,128,128,64], received {ct.shape}")

    if pet.ndim != 4 or pet.shape[0] != 1:
        raise ValueError(f"PET must be [1,128,128,64], received {pet.shape}")

    if ct.shape[1:] != pet.shape[1:]:
        raise ValueError(f"CT/PET mismatch: CT={ct.shape}, PET={pet.shape}")

    if tuple(ct.shape[1:]) != (128, 128, 64):
        raise ValueError(
            "The trained models require volume shape [H,W,D]=[128,128,64]. "
            f"Received {ct.shape[1:]}"
        )


def load_numpy(uploaded_file):
    if uploaded_file.name.lower().endswith(".npz"):
        archive = np.load(uploaded_file)

        if not archive.files:
            raise ValueError("Uploaded NPZ contains no arrays.")

        return archive[archive.files[0]].astype(np.float32)

    return np.load(uploaded_file, allow_pickle=False).astype(np.float32)


def make_demo_volume(uploaded_file, channels):
    image = Image.open(uploaded_file).convert("L").resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0

    if channels == 2:
        volume = np.stack([array, array], axis=0)[:, :, :, None]
    else:
        volume = array[None, :, :, None]

    return np.repeat(volume, repeats=64, axis=3).astype(np.float32)



# 2D INPUT PREPROCESSING

def slice_image(ct, pet, depth):
    image = np.stack(
        [
            normalize_ct(ct[0, :, :, depth]),
            normalize_ct(ct[1, :, :, depth]),
            normalize_pet(pet[0, :, :, depth]),
        ],
        axis=0,
    )

    return TF.resize(
        torch.from_numpy(image),
        [224, 224],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def select_m1_depths(ct):
    values = [
        (float((ct[0, :, :, d] > 0.10).mean()), d)
        for d in range(ct.shape[-1])
    ]

    return [d for _, d in sorted(values, reverse=True)[:48]]


def select_m2_depths(ct, pet):
    values = []

    for d in range(ct.shape[-1]):
        occupancy = float((ct[0, :, :, d] > 0.10).mean())
        texture = float(np.clip(ct[1, :, :, d], 0, 1).mean())
        pet_signal = float(np.clip(pet[0, :, :, d], 0, None).mean())

        values.append(
            (0.70 * occupancy + 0.15 * texture + 0.15 * pet_signal, d)
        )

    return [d for _, d in sorted(values, reverse=True)[:32]]



# M1 PREDICTION

@torch.inference_mode()
def predict_m1(models, ct, pet, real_pet):
    images = [slice_image(ct, pet, d) for d in select_m1_depths(ct)]
    fold_probabilities = []

    for model in models:
        output = []

        for start in range(0, len(images), 16):
            x = torch.stack(images[start:start + 16]).to(DEVICE)

            modality = torch.full(
                (x.shape[0],),
                1.0 if real_pet else 0.0,
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = 0.5 * (
                model(x, modality)
                + model(torch.flip(x, dims=[3]), modality)
            )

            output.append(logits.float().cpu())

        logits = torch.cat(output, dim=0)

        confidence = torch.softmax(logits, dim=1).max(dim=1).values

        selected = torch.topk(
            confidence,
            k=min(12, len(logits)),
        ).indices

        fold_probabilities.append(
            torch.softmax(
                logits[selected].mean(dim=0),
                dim=0,
            ).numpy()
        )

    return np.mean(np.stack(fold_probabilities), axis=0)



# M2 PREDICTION

@torch.inference_mode()
def predict_m2(model, ct, pet, real_pet):
    images = [slice_image(ct, pet, d) for d in select_m2_depths(ct, pet)]
    output = []

    for start in range(0, len(images), 12):
        x = torch.stack(images[start:start + 12]).to(DEVICE)

        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)

        modality = torch.full(
            (x.shape[0],),
            1 if real_pet else 0,
            dtype=torch.long,
            device=DEVICE,
        )

        output.append(model(x, modality).float().cpu())

    logits = torch.cat(output, dim=0)

    confidence = torch.softmax(logits, dim=1).max(dim=1).values

    selected = torch.topk(
        confidence,
        k=min(8, len(logits)),
    ).indices

    patient_logits = (
        0.75 * logits[selected].mean(dim=0)
        + 0.25 * logits.mean(dim=0)
    )

    return torch.softmax(patient_logits, dim=0).numpy()



# M3 PREDICTION

def normalize_m3(volume):
    result = np.empty_like(volume, dtype=np.float32)

    for channel in range(3):
        x = volume[channel].astype(np.float32)

        result[channel] = np.clip(
            (x - x.mean()) / (x.std() + 1e-6),
            -3.0,
            3.0,
        ) / 6.0 + 0.5

    return result


@torch.inference_mode()
def predict_m3(model, ct, pet):
    volume = np.concatenate([ct, pet], axis=0).astype(np.float32)

    # Input transform: [C,H,W,D] -> [C,D,H,W]
    volume = np.ascontiguousarray(
        normalize_m3(volume).transpose(0, 3, 1, 2)
    )

    x = torch.from_numpy(volume).unsqueeze(0).to(DEVICE)

    views = [
        x,
        torch.flip(x, dims=[3]),
        torch.flip(x, dims=[4]),
        torch.flip(x, dims=[2]),
    ]

    native = torch.stack(
        [torch.softmax(model(view), dim=1) for view in views],
        dim=0,
    ).mean(dim=0)[0].cpu().numpy()

    # M3 native class order:
    # [Adenocarcinoma, Large Cell, Small Cell, Squamous]
    #
    # Common ensemble order:
    # [Adenocarcinoma, Small Cell, Squamous, Large Cell]
    return native[[0, 2, 3, 1]]



# STREAMLIT UI

st.title("Lung Cancer Histology Ensemble")
st.caption("M1 + M2 + M3 weighted-average prediction")

st.warning(
    "Research prototype only. It is not a medical device and must "
    "not be used as a clinical diagnosis."
)

with st.expander("Input format", expanded=False):
    st.markdown(
        """
- **Recommended CT:** `.npy` or `.npz`, shape `[2,128,128,64]`
- **Recommended PET:** `.npy` or `.npz`, shape `[1,128,128,64]`
- PET is optional; if absent, the app uses a zero-filled PET volume
- PNG/JPG mode is only a UI test; it creates a synthetic repeated volume
"""
    )

mode = st.radio(
    "Input source",
    [
        "Reference A0001 CT + zero PET",
        "Upload CT/PET .npy or .npz",
        "Upload PNG/JPG demo",
    ],
    horizontal=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    w1 = st.number_input("M1 weight", min_value=0.0, value=1.0, step=0.1)

with c2:
    w2 = st.number_input("M2 weight", min_value=0.0, value=1.0, step=0.1)

with c3:
    w3 = st.number_input("M3 weight", min_value=0.0, value=1.0, step=0.1)

ct = None
pet = None
real_pet = False

if mode == "Reference A0001 CT + zero PET":
    if not CT_PATH.exists() or not PET_PATH.exists():
        st.error("Reference A0001 input files were not found in this notebook.")

    else:
        ct = np.load(CT_PATH, allow_pickle=False).astype(np.float32)
        pet = np.load(PET_PATH, allow_pickle=False).astype(np.float32)
        real_pet = False

        st.info("Loaded A0001 CT with zero-filled PET.")

elif mode == "Upload CT/PET .npy or .npz":
    ct_upload = st.file_uploader("Upload CT volume", type=["npy", "npz"])
    pet_upload = st.file_uploader("Upload PET volume (optional)", type=["npy", "npz"])

    real_pet = st.toggle("Real PET supplied", value=True)

    if ct_upload is not None:
        try:
            ct = load_numpy(ct_upload)

            if pet_upload is not None:
                pet = load_numpy(pet_upload)
            else:
                pet = np.zeros((1,) + tuple(ct.shape[1:]), dtype=np.float32)
                real_pet = False
                st.info("PET was not supplied; using zero-filled PET.")

        except Exception as error:
            st.error(f"Could not load input: {error}")

else:
    ct_image = st.file_uploader(
        "Upload CT-like PNG/JPG",
        type=["png", "jpg", "jpeg"],
        key="ct_image",
    )

    pet_image = st.file_uploader(
        "Upload PET-like PNG/JPG (optional)",
        type=["png", "jpg", "jpeg"],
        key="pet_image",
    )

    real_pet = st.toggle("Treat PET image as real PET", value=False)

    if ct_image is not None:
        ct = make_demo_volume(ct_image, channels=2)

        if pet_image is not None:
            pet = make_demo_volume(pet_image, channels=1)
        else:
            pet = np.zeros((1, 128, 128, 64), dtype=np.float32)
            real_pet = False

        st.warning("PNG/JPG creates a synthetic volume and is only for UI testing.")

if ct is not None and pet is not None:
    st.caption(
        f"CT: `{ct.shape}` | PET: `{pet.shape}` | Real PET: `{real_pet}`"
    )

    st.image(
        ct[0, :, :, ct.shape[-1] // 2],
        caption="Middle CT slice preview",
        clamp=True,
        width=280,
    )

run = st.button(
    "Run M1 + M2 + M3 Ensemble",
    type="primary",
    disabled=(ct is None or pet is None),
)

if run:
    try:
        validate_volume(ct, pet)

        if w1 + w2 + w3 <= 0:
            raise ValueError("At least one model weight must be greater than zero.")

        with st.spinner("Loading models and running ensemble inference..."):
            m1_models, m2_model, m3_model = load_models()

            p1 = predict_m1(m1_models, ct, pet, real_pet)
            p2 = predict_m2(m2_model, ct, pet, real_pet)
            p3 = predict_m3(m3_model, ct, pet)

            weights = np.array([w1, w2, w3], dtype=np.float32)
            weights = weights / weights.sum()

            final = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
            final = final / final.sum()

        prediction = int(np.argmax(final))

        st.success(f"Final prediction: {CLASS_NAMES[prediction]}")
        st.metric("Final confidence", f"{100 * float(final[prediction]):.2f}%")

        results = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "M1 five-fold mean": np.round(p1, 6),
                "M2 Direct-4": np.round(p2, 6),
                "M3 Direct-4": np.round(p3, 6),
                "Final weighted average": np.round(final, 6),
            }
        )

        st.dataframe(results, hide_index=True, use_container_width=True)

        chart = pd.DataFrame(
            {
                "Class": CLASS_NAMES,
                "Probability": final,
            }
        ).set_index("Class")

        st.bar_chart(chart)

    except Exception as error:
        st.exception(error)
'''

APP_FILE.write_text(APP_CODE, encoding="utf-8")

print(f"\nCreated Streamlit app: {APP_FILE}")


import re
import time
import platform
import subprocess
from pathlib import Path
import numpy as np


# Replace with your selected final validation/test scores
M1_VALIDATION_SCORE = 1.00
M2_VALIDATION_SCORE = 1.00
M3_VALIDATION_SCORE = 1.00

ROOT = Path("/kaggle/working/lung_ensemble_app")
MODEL_DIR = ROOT / "models"
APP_FILE = ROOT / "app.py"

STREAMLIT_LOG = ROOT / "streamlit_two_reference_cases.log"
TUNNEL_LOG = ROOT / "cloudflared_two_reference_cases.log"

BIN_DIR = ROOT / "bin"
CLOUDFLARED_BIN = BIN_DIR / "cloudflared"

DATA_ROOT = Path(
    "/kaggle/input/datasets/tejajaladi/"
    "cancer-image-dataset-part-5-preprocessed/part5"
)

REFERENCE_CASES = {
    "Case 1 — A0001: CT + zero PET": {
        "patient_id": "A0001",
        "ct_path": DATA_ROOT / "ct_patches" / "A0001.npy",
        "pet_path": DATA_ROOT / "pet_zeros" / "A0001.npy",
        "real_pet": False,
        "description": "CT with zero-filled PET placeholder",
    },
    "Case 2 — A0164: CT + real PET": {
        "patient_id": "A0164",
        "ct_path": DATA_ROOT / "ct_patches" / "A0164.npy",
        "pet_path": DATA_ROOT / "pet_patches" / "A0164.npy",
        "real_pet": True,
        "description": "Matched CT and real PET patch",
    },
}

# VERIFY CHECKPOINT FILES
required_files = [
    MODEL_DIR / "m1/E0.pt",
    MODEL_DIR / "m1/E1.pt",
    MODEL_DIR / "m1/E2.pt",
    MODEL_DIR / "m1/E3.pt",
    MODEL_DIR / "m1/E4.pt",
    MODEL_DIR / "m2/direct4.pt",
    MODEL_DIR / "m3/E0_D4_best.pt",
]

missing = [str(path) for path in required_files if not path.exists()]

if missing:
    raise FileNotFoundError(
        "Missing checkpoint files:\n\n"
        + "\n".join(missing)
        + "\n\nRun Cell 1 once before this Cell 2."
    )

ROOT.mkdir(parents=True, exist_ok=True)
BIN_DIR.mkdir(parents=True, exist_ok=True)


# VALIDATE REFERENCE FILES BEFORE LAUNCH
def validate_reference_case(case_name, case):
    ct_path = case["ct_path"]
    pet_path = case["pet_path"]

    print(f"\nChecking {case_name}")
    print("CT :", ct_path)
    print("PET:", pet_path)

    if not ct_path.exists():
        raise FileNotFoundError(
            f"Missing CT file for {case_name}:\n{ct_path}"
        )

    if not pet_path.exists():
        raise FileNotFoundError(
            f"Missing PET file for {case_name}:\n{pet_path}"
        )

    ct = np.load(ct_path, mmap_mode="r", allow_pickle=False)
    pet = np.load(pet_path, mmap_mode="r", allow_pickle=False)

    print("CT shape :", ct.shape)
    print("PET shape:", pet.shape)

    if ct.shape != (2, 128, 128, 64):
        raise ValueError(
            f"{case_name}: invalid CT shape {ct.shape}. "
            "Expected (2, 128, 128, 64)."
        )

    if pet.shape != (1, 128, 128, 64):
        raise ValueError(
            f"{case_name}: invalid PET shape {pet.shape}. "
            "Expected (1, 128, 128, 64)."
        )

    print("Status: valid")


for case_name, case_info in REFERENCE_CASES.items():
    validate_reference_case(case_name, case_info)

print("\nBoth reference cases are valid.")


# COMPLETE STREAMLIT APP

APP_CODE = r'''
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import timm

from pathlib import Path
from PIL import Image
from torchvision.models.video import swin3d_s
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

st.set_page_config(
    page_title="Lung Cancer Ensemble",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path("/kaggle/working/lung_ensemble_app")
MODEL_DIR = ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Adenocarcinoma",
    "Small Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Large Cell Carcinoma",
]

M1_VALIDATION_SCORE = __M1_SCORE__
M2_VALIDATION_SCORE = __M2_SCORE__
M3_VALIDATION_SCORE = __M3_SCORE__

REFERENCE_CASES = {
    "Case 1 — A0001: CT + zero PET": {
        "patient_id": "A0001",
        "ct_path": Path(
            "/kaggle/input/datasets/tejajaladi/"
            "cancer-image-dataset-part-5-preprocessed/"
            "part5/ct_patches/A0001.npy"
        ),
        "pet_path": Path(
            "/kaggle/input/datasets/tejajaladi/"
            "cancer-image-dataset-part-5-preprocessed/"
            "part5/pet_zeros/A0001.npy"
        ),
        "real_pet": False,
        "description": "CT with zero-filled PET placeholder",
    },
    "Case 2 — A0164: CT + real PET": {
        "patient_id": "A0164",
        "ct_path": Path(
            "/kaggle/input/datasets/tejajaladi/"
            "cancer-image-dataset-part-5-preprocessed/"
            "part5/ct_patches/A0164.npy"
        ),
        "pet_path": Path(
            "/kaggle/input/datasets/tejajaladi/"
            "cancer-image-dataset-part-5-preprocessed/"
            "part5/pet_patches/A0164.npy"
        ),
        "real_pet": True,
        "description": "Matched CT and real PET patch",
    },
}

with st.sidebar:
    st.title("🫁 Lung Cancer Models")
    st.markdown("**M1 — EfficientNetV2-L**")
    st.markdown("**M2 — ConvNeXt-Large**")
    st.markdown("**M3 — Swin3D-S**")
    st.divider()
    st.markdown("**Final prediction**")
    st.caption("Weighted average of M1, M2 and M3")
    st.warning("Research use only. Not a clinical diagnosis.")

class M1Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(features + 1),
            nn.Dropout(0.35),
            nn.Linear(features + 1, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 4),
        )

    def forward(self, x, modality):
        z = self.backbone(x)
        modality = modality.float().unsqueeze(1).to(z.device)
        return self.head(torch.cat([z, modality], dim=1))


class M2Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_large.fb_in22k_ft_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        features = self.backbone.num_features
        self.mod_emb = nn.Embedding(3, 8)

        self.head = nn.Sequential(
            nn.LayerNorm(features + 8),
            nn.Dropout(0.35),
            nn.Linear(features + 8, 512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 4),
        )

    def forward(self, x, modality):
        z = self.backbone.forward_features(x)
        z = self.backbone.forward_head(z, pre_logits=True)

        if z.ndim > 2:
            z = z.flatten(1)

        modality = self.mod_emb(modality.long().to(z.device))
        return self.head(torch.cat([z, modality], dim=1))


class M3Model(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = swin3d_s(weights=None, progress=False)

        self.patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop
        self.features = backbone.features
        self.norm = backbone.norm
        self.avgpool = backbone.avgpool

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(backbone.num_features),
            nn.Dropout(0.35),
            nn.Linear(backbone.num_features, 512),
            nn.GELU(),
            nn.Dropout(0.175),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return self.head(self.avgpool(x))



# MODEL LOADING
def read_checkpoint(path):
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = {
            key: value
            for key, value in checkpoint.items()
            if isinstance(value, torch.Tensor)
        }
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    cleaned = {}

    for key, value in state_dict.items():
        for prefix in ("module.", "model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value

    return cleaned


def strict_load(model, path, label):
    missing, unexpected = model.load_state_dict(
        read_checkpoint(path),
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint mismatch.\n"
            f"Missing: {missing[:10]}\n"
            f"Unexpected: {unexpected[:10]}"
        )

    return model.eval().to(DEVICE)


@st.cache_resource(show_spinner="Loading M1, M2 and M3 checkpoints...")
def load_models():
    m1_models = [
        strict_load(
            M1Model(),
            MODEL_DIR / "m1" / f"E{i}.pt",
            f"M1 E{i}",
        )
        for i in range(5)
    ]

    m2_model = strict_load(
        M2Model(),
        MODEL_DIR / "m2/direct4.pt",
        "M2 Direct-4",
    )

    m3_model = strict_load(
        M3Model(),
        MODEL_DIR / "m3/E0_D4_best.pt",
        "M3 Direct-4",
    )

    return m1_models, m2_model, m3_model



# INPUT HELPERS

def normalize_ct(x):
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def normalize_pet(x):
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if x.size and float(x.max()) > 1.0:
        x = np.clip(x, 0.0, 10.0) / 10.0

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def validate_volume(ct, pet):
    if ct.ndim != 4 or ct.shape[0] != 2:
        raise ValueError(
            f"CT must be [2,128,128,64]. Received {ct.shape}"
        )

    if pet.ndim != 4 or pet.shape[0] != 1:
        raise ValueError(
            f"PET must be [1,128,128,64]. Received {pet.shape}"
        )

    if ct.shape[1:] != pet.shape[1:]:
        raise ValueError(
            f"CT/PET shape mismatch: {ct.shape} vs {pet.shape}"
        )

    if tuple(ct.shape[1:]) != (128, 128, 64):
        raise ValueError(
            "Expected spatial shape [128,128,64]. "
            f"Received {ct.shape[1:]}"
        )


def load_numpy(uploaded_file):
    if uploaded_file.name.lower().endswith(".npz"):
        archive = np.load(uploaded_file)

        if not archive.files:
            raise ValueError("The uploaded NPZ contains no arrays.")

        return archive[archive.files[0]].astype(np.float32)

    return np.load(uploaded_file, allow_pickle=False).astype(np.float32)


def make_demo_volume(uploaded_file, channels):
    image = Image.open(uploaded_file).convert("L").resize((128, 128))
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    if channels == 2:
        volume = np.stack(
            [image_array, image_array],
            axis=0,
        )[:, :, :, None]
    else:
        volume = image_array[None, :, :, None]

    return np.repeat(
        volume,
        repeats=64,
        axis=3,
    ).astype(np.float32)



# M1 AND M2 2D PREPROCESSING

def make_slice_image(ct, pet, depth):
    image = np.stack(
        [
            normalize_ct(ct[0, :, :, depth]),
            normalize_ct(ct[1, :, :, depth]),
            normalize_pet(pet[0, :, :, depth]),
        ],
        axis=0,
    )

    return TF.resize(
        torch.from_numpy(image),
        [224, 224],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def m1_depths(ct):
    scores = [
        (float((ct[0, :, :, depth] > 0.10).mean()), depth)
        for depth in range(ct.shape[-1])
    ]

    return [
        depth
        for _, depth in sorted(scores, reverse=True)[:48]
    ]


def m2_depths(ct, pet):
    scores = []

    for depth in range(ct.shape[-1]):
        occupancy = float((ct[0, :, :, depth] > 0.10).mean())
        texture = float(
            np.clip(ct[1, :, :, depth], 0.0, 1.0).mean()
        )
        pet_signal = float(
            np.clip(pet[0, :, :, depth], 0.0, None).mean()
        )

        scores.append(
            (
                0.70 * occupancy
                + 0.15 * texture
                + 0.15 * pet_signal,
                depth,
            )
        )

    return [
        depth
        for _, depth in sorted(scores, reverse=True)[:32]
    ]

# PREDICTION FUNCTIONS
@torch.inference_mode()
def predict_m1(models, ct, pet, real_pet):
    images = [
        make_slice_image(ct, pet, depth)
        for depth in m1_depths(ct)
    ]

    fold_results = []

    for model in models:
        logits_list = []

        for start in range(0, len(images), 16):
            x = torch.stack(
                images[start:start + 16]
            ).to(DEVICE)

            modality = torch.full(
                (x.shape[0],),
                1.0 if real_pet else 0.0,
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = 0.5 * (
                model(x, modality)
                + model(torch.flip(x, dims=[3]), modality)
            )

            logits_list.append(logits.float().cpu())

        logits = torch.cat(logits_list, dim=0)

        confidence = torch.softmax(
            logits,
            dim=1,
        ).max(dim=1).values

        selected = torch.topk(
            confidence,
            k=min(12, logits.shape[0]),
        ).indices

        fold_results.append(
            torch.softmax(
                logits[selected].mean(dim=0),
                dim=0,
            ).numpy()
        )

    return np.mean(np.stack(fold_results), axis=0)


@torch.inference_mode()
def predict_m2(model, ct, pet, real_pet):
    images = [
        make_slice_image(ct, pet, depth)
        for depth in m2_depths(ct, pet)
    ]

    logits_list = []

    for start in range(0, len(images), 12):
        x = torch.stack(
            images[start:start + 12]
        ).to(DEVICE)

        if DEVICE.type == "cuda":
            x = x.contiguous(
                memory_format=torch.channels_last
            )

        modality = torch.full(
            (x.shape[0],),
            1 if real_pet else 0,
            dtype=torch.long,
            device=DEVICE,
        )

        logits_list.append(
            model(x, modality).float().cpu()
        )

    logits = torch.cat(logits_list, dim=0)

    confidence = torch.softmax(
        logits,
        dim=1,
    ).max(dim=1).values

    selected = torch.topk(
        confidence,
        k=min(8, logits.shape[0]),
    ).indices

    combined_logits = (
        0.75 * logits[selected].mean(dim=0)
        + 0.25 * logits.mean(dim=0)
    )

    return torch.softmax(
        combined_logits,
        dim=0,
    ).numpy()


def normalize_m3(volume):
    result = np.empty_like(
        volume,
        dtype=np.float32,
    )

    for channel in range(3):
        x = volume[channel].astype(np.float32)

        result[channel] = np.clip(
            (x - x.mean()) / (x.std() + 1e-6),
            -3.0,
            3.0,
        ) / 6.0 + 0.5

    return result


@torch.inference_mode()
def predict_m3(model, ct, pet):
    volume = np.concatenate(
        [ct, pet],
        axis=0,
    ).astype(np.float32)

    volume = np.ascontiguousarray(
        normalize_m3(volume).transpose(0, 3, 1, 2)
    )

    x = torch.from_numpy(volume).unsqueeze(0).to(DEVICE)

    views = [
        x,
        torch.flip(x, dims=[3]),
        torch.flip(x, dims=[4]),
        torch.flip(x, dims=[2]),
    ]

    native = torch.stack(
        [
            torch.softmax(model(view), dim=1)
            for view in views
        ],
        dim=0,
    ).mean(dim=0)[0].cpu().numpy()

    # Native M3 order:
    # [Adenocarcinoma, Large Cell, Small Cell, Squamous]
    #
    # Shared ensemble order:
    # [Adenocarcinoma, Small Cell, Squamous, Large Cell]
    return native[[0, 2, 3, 1]]

# SESSION STATE
if "ct" not in st.session_state:
    st.session_state.ct = None

if "pet" not in st.session_state:
    st.session_state.pet = None

if "real_pet" not in st.session_state:
    st.session_state.real_pet = False

if "source" not in st.session_state:
    st.session_state.source = None

if "result" not in st.session_state:
    st.session_state.result = None

if "reference_error" not in st.session_state:
    st.session_state.reference_error = None

if "manual_weights" not in st.session_state:
    st.session_state.manual_weights = False

scores = np.asarray(
    [
        M1_VALIDATION_SCORE,
        M2_VALIDATION_SCORE,
        M3_VALIDATION_SCORE,
    ],
    dtype=np.float32,
)

if np.any(scores < 0) or float(scores.sum()) <= 0:
    scores = np.ones(3, dtype=np.float32)

default_weights = scores / scores.sum()

if "m1_weight" not in st.session_state:
    st.session_state.m1_weight = float(default_weights[0])

if "m2_weight" not in st.session_state:
    st.session_state.m2_weight = float(default_weights[1])

if "m3_weight" not in st.session_state:
    st.session_state.m3_weight = float(default_weights[2])


def clear_result():
    st.session_state.result = None


def select_reference_case(case_name):
    try:
        case = REFERENCE_CASES[case_name]

        ct_value = np.load(
            case["ct_path"],
            allow_pickle=False,
        ).astype(np.float32)

        pet_value = np.load(
            case["pet_path"],
            allow_pickle=False,
        ).astype(np.float32)

        validate_volume(ct_value, pet_value)

        st.session_state.ct = ct_value
        st.session_state.pet = pet_value
        st.session_state.real_pet = bool(case["real_pet"])
        st.session_state.source = (
            f"{case_name} | {case['description']}"
        )

        st.session_state.reference_error = None
        clear_result()

    except Exception as error:
        st.session_state.reference_error = str(error)


def clear_input():
    st.session_state.ct = None
    st.session_state.pet = None
    st.session_state.real_pet = False
    st.session_state.source = None
    st.session_state.reference_error = None
    clear_result()



# USER INTERFACE

st.title("Lung Cancer Histology Ensemble")
st.caption("M1 + M2 + M3 weighted-average prediction")

st.warning(
    "Research prototype only."
)

with st.expander("Advanced settings", expanded=False):
    st.toggle(
        "Change model weights manually",
        key="manual_weights",
    )

    if st.session_state.manual_weights:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input(
                "M1 weight",
                min_value=0.0,
                step=0.01,
                format="%.3f",
                key="m1_weight",
            )

        with col2:
            st.number_input(
                "M2 weight",
                min_value=0.0,
                step=0.01,
                format="%.3f",
                key="m2_weight",
            )

        with col3:
            st.number_input(
                "M3 weight",
                min_value=0.0,
                step=0.01,
                format="%.3f",
                key="m3_weight",
            )

        st.caption("Manual weights are normalized automatically.")

    else:
        st.caption(
            f"Default score-based weights: "
            f"M1 {default_weights[0]:.3f} | "
            f"M2 {default_weights[1]:.3f} | "
            f"M3 {default_weights[2]:.3f}"
        )

if st.session_state.manual_weights:
    raw_weights = np.asarray(
        [
            st.session_state.m1_weight,
            st.session_state.m2_weight,
            st.session_state.m3_weight,
        ],
        dtype=np.float32,
    )
else:
    raw_weights = default_weights.copy()

weights_valid = float(raw_weights.sum()) > 0.0

active_weights = (
    raw_weights / raw_weights.sum()
    if weights_valid
    else default_weights.copy()
)

tab_reference, tab_upload, tab_demo = st.tabs(
    [
        "Reference Cases",
        "Upload .npy / .npz",
        "PNG / JPG Demo",
    ]
)


# TAB 1 — TWO REFERENCE CASES

with tab_reference:
    st.subheader("Reference cases")

    selected_case_name = st.radio(
        "Choose a reference patient",
        options=list(REFERENCE_CASES.keys()),
        key="reference_case_selector",
    )

    selected_case = REFERENCE_CASES[selected_case_name]

    st.info(
        f"Patient ID: {selected_case['patient_id']}\n\n"
        f"Input type: {selected_case['description']}"
    )

    st.code(
        f"CT:  {selected_case['ct_path']}\n"
        f"PET: {selected_case['pet_path']}",
        language=None,
    )

    ct_available = selected_case["ct_path"].exists()
    pet_available = selected_case["pet_path"].exists()

    if ct_available and pet_available:
        st.success("CT and PET files are available.")

        st.button(
            "Use Selected Reference Case",
            type="primary",
            on_click=select_reference_case,
            args=(selected_case_name,),
            key="reference_button",
        )

    else:
        if not ct_available:
            st.error(
                f"CT file is unavailable:\n{selected_case['ct_path']}"
            )

        if not pet_available:
            st.error(
                f"PET file is unavailable:\n{selected_case['pet_path']}"
            )

    if st.session_state.reference_error:
        st.error(
            "Reference-case loading failed: "
            + st.session_state.reference_error
        )



# TAB 2 — UPLOAD NPY / NPZ

with tab_upload:
    st.subheader("Upload CT and PET volumes")
    st.caption("CT [2,128,128,64] | PET [1,128,128,64]")

    uploaded_ct = st.file_uploader(
        "Upload CT volume",
        type=["npy", "npz"],
        key="uploaded_ct",
    )

    uploaded_pet = st.file_uploader(
        "Upload PET volume (optional)",
        type=["npy", "npz"],
        key="uploaded_pet",
    )

    upload_real_pet = st.checkbox(
        "Uploaded PET is real PET data",
        value=True,
        key="upload_real_pet",
    )

    if st.button(
        "Use Uploaded Volumes",
        type="primary",
        key="upload_button",
    ):
        try:
            if uploaded_ct is None:
                raise ValueError("Upload a CT volume first.")

            ct_value = load_numpy(uploaded_ct)

            if uploaded_pet is None:
                pet_value = np.zeros(
                    (1,) + tuple(ct_value.shape[1:]),
                    dtype=np.float32,
                )
                real_pet_value = False
            else:
                pet_value = load_numpy(uploaded_pet)
                real_pet_value = upload_real_pet

            validate_volume(ct_value, pet_value)

            st.session_state.ct = ct_value
            st.session_state.pet = pet_value
            st.session_state.real_pet = real_pet_value
            st.session_state.source = "Uploaded CT/PET volume"
            st.session_state.reference_error = None
            clear_result()

            st.success("Volume selected successfully.")

        except Exception as error:
            st.error(f"Could not use uploaded volume: {error}")

# TAB 3 — PNG / JPG DEMO
with tab_demo:
    st.subheader("PNG / JPG demonstration mode")

    st.warning(
        "Demo mode creates a synthetic repeated 3D volume. "
        "It is for interface testing only."
    )

    uploaded_ct_image = st.file_uploader(
        "Upload CT-like PNG/JPG",
        type=["png", "jpg", "jpeg"],
        key="uploaded_ct_image",
    )

    uploaded_pet_image = st.file_uploader(
        "Upload PET-like PNG/JPG (optional)",
        type=["png", "jpg", "jpeg"],
        key="uploaded_pet_image",
    )

    demo_real_pet = st.checkbox(
        "Treat PET image as real PET",
        value=False,
        key="demo_real_pet",
    )

    if st.button(
        "Use Demo Images",
        type="primary",
        key="demo_button",
    ):
        try:
            if uploaded_ct_image is None:
                raise ValueError("Upload a CT-like image first.")

            ct_value = make_demo_volume(
                uploaded_ct_image,
                channels=2,
            )

            if uploaded_pet_image is None:
                pet_value = np.zeros(
                    (1, 128, 128, 64),
                    dtype=np.float32,
                )
                real_pet_value = False
            else:
                pet_value = make_demo_volume(
                    uploaded_pet_image,
                    channels=1,
                )
                real_pet_value = demo_real_pet

            validate_volume(ct_value, pet_value)

            st.session_state.ct = ct_value
            st.session_state.pet = pet_value
            st.session_state.real_pet = real_pet_value
            st.session_state.source = "PNG/JPG demonstration"
            st.session_state.reference_error = None
            clear_result()

            st.success("Demo input selected successfully.")

        except Exception as error:
            st.error(f"Could not use demonstration input: {error}")



# PREDICTION SECTION

st.divider()
st.subheader("Prediction")

ct = st.session_state.ct
pet = st.session_state.pet
real_pet = st.session_state.real_pet
source = st.session_state.source

if ct is None or pet is None:
    st.info("Choose an input in one of the three tabs above.")

else:
    st.success(f"Selected input: {source}")

    left, right = st.columns([1, 2])

    with left:
        st.image(
            ct[0, :, :, ct.shape[-1] // 2],
            caption="Middle CT slice",
            clamp=True,
            width=300,
        )

    with right:
        st.code(
            f"CT shape: {ct.shape}\n"
            f"PET shape: {pet.shape}\n"
            f"Real PET: {real_pet}",
            language=None,
        )

        st.caption(
            f"Active weights: "
            f"M1 {active_weights[0]:.3f} | "
            f"M2 {active_weights[1]:.3f} | "
            f"M3 {active_weights[2]:.3f}"
        )

        st.button(
            "Clear selected input",
            on_click=clear_input,
            key="clear_input_button",
        )

    if not weights_valid:
        st.error("At least one manual weight must be greater than zero.")

    if st.button(
        "Run M1 + M2 + M3 Ensemble",
        type="primary",
        disabled=not weights_valid,
        key="run_prediction",
    ):
        try:
            with st.spinner(
                "Running M1, M2, M3 and final prediction..."
            ):
                m1_models, m2_model, m3_model = load_models()

                p1 = predict_m1(
                    m1_models,
                    ct,
                    pet,
                    real_pet,
                )

                p2 = predict_m2(
                    m2_model,
                    ct,
                    pet,
                    real_pet,
                )

                p3 = predict_m3(
                    m3_model,
                    ct,
                    pet,
                )

                final = (
                    active_weights[0] * p1
                    + active_weights[1] * p2
                    + active_weights[2] * p3
                )

                final = final / final.sum()

            st.session_state.result = {
                "m1": p1.tolist(),
                "m2": p2.tolist(),
                "m3": p3.tolist(),
                "final": final.tolist(),
                "weights": active_weights.tolist(),
                "source": source,
            }

        except Exception as error:
            st.error(f"Prediction failed: {error}")



# PREDICTION RESULT

if st.session_state.result is not None:
    result = st.session_state.result

    p1 = np.asarray(result["m1"], dtype=np.float32)
    p2 = np.asarray(result["m2"], dtype=np.float32)
    p3 = np.asarray(result["m3"], dtype=np.float32)
    final = np.asarray(result["final"], dtype=np.float32)
    weights = np.asarray(result["weights"], dtype=np.float32)

    prediction_index = int(np.argmax(final))

    st.divider()
    st.subheader("Prediction result")

    st.success(
        f"Final prediction: {CLASS_NAMES[prediction_index]}"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Final confidence",
            f"{100 * float(final[prediction_index]):.2f}%",
        )

    with c2:
        st.metric(
            "Predicted class",
            CLASS_NAMES[prediction_index],
        )

    st.caption(
        f"Used weights: "
        f"M1 {weights[0]:.3f} | "
        f"M2 {weights[1]:.3f} | "
        f"M3 {weights[2]:.3f}"
    )

    results = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "M1 probability": np.round(p1, 6),
            "M2 probability": np.round(p2, 6),
            "M3 probability": np.round(p3, 6),
            "Final probability": np.round(final, 6),
        }
    )

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True,
    )

    chart = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Final probability": final,
        }
    ).set_index("Class")

    st.bar_chart(chart)
'''


# INSERT YOUR WEIGHTS INTO THE APP SOURCE

APP_CODE = (
    APP_CODE
    .replace("__M1_SCORE__", str(float(M1_VALIDATION_SCORE)))
    .replace("__M2_SCORE__", str(float(M2_VALIDATION_SCORE)))
    .replace("__M3_SCORE__", str(float(M3_VALIDATION_SCORE)))
)

APP_FILE.write_text(APP_CODE, encoding="utf-8")

syntax_check = subprocess.run(
    ["python", "-m", "py_compile", str(APP_FILE)],
    capture_output=True,
    text=True,
)

if syntax_check.returncode != 0:
    raise RuntimeError(
        "Generated app.py has a syntax error:\n\n"
        + syntax_check.stderr
    )

print("\nRebuilt app.py with two reference cases.")
print("Python syntax validation passed.")


# INSTALL CLOUDFLARED IF NEEDED
if platform.machine().lower() not in ("x86_64", "amd64"):
    raise RuntimeError(
        "This Cloudflare launcher expects an x86_64 runtime."
    )

if not CLOUDFLARED_BIN.exists():
    install = subprocess.run(
        [
            "bash",
            "-lc",
            "curl -L --fail --retry 3 "
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            f"cloudflared-linux-amd64 -o '{CLOUDFLARED_BIN}' "
            f"&& chmod +x '{CLOUDFLARED_BIN}'",
        ],
        capture_output=True,
        text=True,
    )

    if install.returncode != 0:
        raise RuntimeError(
            "cloudflared installation failed:\n\n"
            + install.stderr[-3000:]
        )


# CREATE STREAMLIT CONFIG
config_dir = ROOT / ".streamlit"
config_dir.mkdir(parents=True, exist_ok=True)

(config_dir / "config.toml").write_text(
    """
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false
fileWatcherType = "none"

[browser]
gatherUsageStats = false

[client]
toolbarMode = "minimal"

[runner]
fastReruns = false
""".strip(),
    encoding="utf-8",
)


# PROCESS HELPERS
def shell(command):
    return subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
    )


def streamlit_ok():
    response = shell(
        "curl -s --max-time 2 http://127.0.0.1:8501/_stcore/health"
    )

    return (
        response.returncode == 0
        and "ok" in response.stdout.lower()
    )


# STOP OLD APP SERVER AND TUNNEL
print("Stopping previous Streamlit server and tunnel...")

shell("fuser -k 8501/tcp >/dev/null 2>&1 || true")
shell("pkill -x cloudflared >/dev/null 2>&1 || true")

time.sleep(20)


# LAUNCH STREAMLIT
print("Starting Streamlit...")

streamlit_handle = open(STREAMLIT_LOG, "w")

streamlit_process = subprocess.Popen(
    [
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ],
    stdout=streamlit_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

started = False

for _ in range(45):
    if streamlit_ok():
        started = True
        break

    if streamlit_process.poll() is not None:
        break

    time.sleep(1)

if not started:
    logs = (
        STREAMLIT_LOG.read_text(errors="ignore")[-6000:]
        if STREAMLIT_LOG.exists()
        else "No Streamlit log was created."
    )

    raise RuntimeError(
        "Streamlit failed to start:\n\n"
        + logs
    )


# LAUNCH CLOUDFLARE QUICK TUNNEL

print("Creating Cloudflare public URL...")

tunnel_handle = open(TUNNEL_LOG, "w")

tunnel_process = subprocess.Popen(
    [
        str(CLOUDFLARED_BIN),
        "tunnel",
        "--no-autoupdate",
        "--protocol",
        "http2",
        "--url",
        "http://127.0.0.1:8501",
    ],
    stdout=tunnel_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

public_url = None

for _ in range(60):
    log_text = (
        TUNNEL_LOG.read_text(errors="ignore")
        if TUNNEL_LOG.exists()
        else ""
    )

    match = re.search(
        r"https://[-a-zA-Z0-9]+\.trycloudflare\.com",
        log_text,
    )

    if match:
        public_url = match.group(0)
        break

    if tunnel_process.poll() is not None:
        break

    time.sleep(1)

if public_url is None:
    logs = (
        TUNNEL_LOG.read_text(errors="ignore")[-5000:]
        if TUNNEL_LOG.exists()
        else "No Cloudflare tunnel log was created."
    )

    raise RuntimeError(
        "Cloudflare tunnel failed:\n\n"
        + logs
    )

print("\n" + "=" * 100)
print("APP IS LIVE")
print("=" * 100)
print(public_url)
print("=" * 100)


# KEEP THE KAGGLE SESSION ACTIVE

try:
    while True:
        if not streamlit_ok():
            print("\nStreamlit stopped. Check:", STREAMLIT_LOG)
            break

        if tunnel_process.poll() is not None:
            print("\nCloudflare tunnel stopped. Check:", TUNNEL_LOG)
            break

        time.sleep(20)

except KeyboardInterrupt:
    try:
        tunnel_process.terminate()
    except Exception:
        pass

    try:
        streamlit_process.terminate()
    except Exception:
        pass

    print("App stopped.")

