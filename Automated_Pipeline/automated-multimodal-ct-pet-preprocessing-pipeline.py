import subprocess
subprocess.run(["pip", "install", "lungmask", "SimpleITK", "pydicom",
                "scipy", "scikit-learn", "openpyxl", "-q"], check=True)

import os, glob, json, re, warnings
import numpy as np
import pandas as pd
import pydicom
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

BASE         = "/kaggle/input/datasets/tejajaladi"
PATH_PART1   = f"{BASE}/cancer-image-dataset-part1"
PATH_PART2   = f"{BASE}/cancer-image-dataset-part2"
PATH_XLSX    = f"{BASE}/cancer-image-dataset-part-3-statistics-clinical/statistics-clinical-20201221.xlsx"
PATH_ANN     = f"{BASE}/cancer-image-dataset-part-4-annotation-zip/Annotation"

OUT_ROOT     = "/kaggle/working/part5"
OUT_CT       = f"{OUT_ROOT}/ct_patches"
OUT_PET      = f"{OUT_ROOT}/pet_patches"
OUT_PET_ZERO = f"{OUT_ROOT}/pet_zeros"
OUT_AUG      = f"{OUT_ROOT}/augmented_e"
OUT_LOOCV    = f"{OUT_ROOT}/loocv_splits"
OUT_META     = f"{OUT_ROOT}/metadata"

for d in [OUT_CT, OUT_PET, OUT_PET_ZERO, OUT_AUG, OUT_LOOCV, OUT_META]:
    os.makedirs(d, exist_ok=True)

PATCH_SHAPE  = (128, 128, 64)   # (H, W, D)
SEP          = "=" * 70
VALID_PID_RE = re.compile(r'^[ABEG]\d{4}$', re.IGNORECASE)

CLASS_MAP = {
    "A": "Adenocarcinoma",
    "B": "Small Cell Carcinoma",
    "E": "Large Cell Carcinoma",
    "G": "Squamous Cell Carcinoma"
}

def get_class(pid):
    return CLASS_MAP.get(str(pid).strip().upper()[0], "Unknown")

def is_valid_pid(pid):
    return bool(VALID_PID_RE.match(str(pid).strip()))



# HELPER: Load DICOM series → (H, W, D) float32 HU volume
# FIXED: handles single-frame AND multiframe DICOMs

def load_dicom_volume(series_path):
    dcm_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    if not dcm_files:
        return None, None

    slices, positions = [], []

    for fpath in dcm_files:
        try:
            ds = pydicom.dcmread(fpath)
            arr = ds.pixel_array.astype(np.float32)
            slope     = float(getattr(ds, "RescaleSlope",     1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            arr = arr * slope + intercept

            # FIX BUG 1: Multiframe DICOM — pixel_array is (D, H, W)
            if arr.ndim == 3:
                for frame_i in range(arr.shape[0]):
                    slices.append(arr[frame_i])     # (H, W)
                    positions.append(float(frame_i))
            elif arr.ndim == 2:
                slices.append(arr)                  # (H, W)
                try:
                    pos = float(ds.ImagePositionPatient[2])
                except Exception:
                    try:
                        pos = float(ds.InstanceNumber)
                    except Exception:
                        pos = float(len(positions))
                positions.append(pos)
        except Exception:
            pass

    if not slices:
        return None, None

    # Sort slices by spatial position
    order = np.argsort(positions)
    slices = [slices[i] for i in order]

    # Ensure all slices same spatial shape — use most common shape
    shapes = [s.shape for s in slices]
    common_shape = max(set(shapes), key=shapes.count)
    slices = [s for s in slices if s.shape == common_shape]

    volume = np.stack(slices, axis=-1)  # (H, W, D)
    assert volume.ndim == 3, f"Expected 3D, got shape {volume.shape}"

    # Extract metadata from first file
    try:
        ds0 = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    except Exception:
        ds0 = None

    meta = {
        "modality"         : str(getattr(ds0, "Modality",         "UNK")) if ds0 else "UNK",
        "manufacturer"     : str(getattr(ds0, "Manufacturer",     "UNK")) if ds0 else "UNK",
        "slice_thickness"  : float(getattr(ds0, "SliceThickness", 1.0) or 1.0) if ds0 else 1.0,
        "pixel_spacing"    : [float(x) for x in getattr(ds0, "PixelSpacing", [1.0, 1.0])] if ds0 else [1.0, 1.0],
        "n_slices"         : volume.shape[2],
        "original_shape"   : list(volume.shape),
        "series_desc"      : str(getattr(ds0, "SeriesDescription", "")) if ds0 else "",
        "injected_dose_bq" : 0.0,
        "suv_fallback"     : True,
    }

    try:
        rpi = ds0.RadiopharmaceuticalInformationSequence[0]
        dose = getattr(rpi, "RadionuclideTotalDose", None)
        if dose is not None:
            meta["injected_dose_bq"] = float(dose)
            meta["suv_fallback"]     = False
    except Exception:
        pass

    return volume, meta


def is_scout(meta):
    desc = str(meta.get("series_desc", "")).lower()
    st   = float(meta.get("slice_thickness", 1.0))
    return any(k in desc for k in ["scout", "topogram", "localizer", "surview"]) or st > 20.0

def apply_ct_windows(volume):
    assert volume.ndim == 3, f"apply_ct_windows: expected 3D, got {volume.shape}"
    lung = (np.clip(volume, -1000.0, -200.0) - (-1000.0)) / 800.0
    medi = (np.clip(volume,  -175.0,  225.0) - (-175.0))  / 400.0
    return np.stack([lung, medi], axis=0).astype(np.float32)  # (2, H, W, D)

# HELPER: PET → SUV conversion
def convert_pet_suv(volume, weight_kg, injected_dose_bq):
    """Returns (suv_normalized [0,1], used_fallback bool)."""
    if injected_dose_bq > 0 and weight_kg > 0:
        suv = volume * (weight_kg * 1000.0) / injected_dose_bq
        used_fallback = False
    else:
        # Fallback: scale raw values proportionally to [0,20] range
        v_max = volume.max()
        suv = (volume / (v_max + 1e-8)) * 20.0 if v_max > 0 else volume
        used_fallback = True
    suv_norm = np.clip(suv, 0.0, 20.0) / 20.0
    return suv_norm.astype(np.float32), used_fallback

def extract_patch(volume, cx, cy, patch_shape=PATCH_SHAPE):
    # FIX BUG 2: strict 3D assertion
    assert volume.ndim == 3, f"extract_patch: expected 3D, got {volume.ndim}D shape={volume.shape}"
    H, W, D = volume.shape
    ph, pw, pd = patch_shape

    # FIX BUG 5: safe clipping with max guard for small volumes
    half_w = pw // 2
    half_h = ph // 2
    half_d = pd // 2

    cx_int = int(np.clip(round(cx), half_w, max(W - half_w, half_w + 1)))
    cy_int = int(np.clip(round(cy), half_h, max(H - half_h, half_h + 1)))
    cz_int = D // 2

    x1 = max(0, cx_int - half_w)
    y1 = max(0, cy_int - half_h)
    z1 = max(0, cz_int - half_d)

    x2 = min(W, x1 + pw)
    y2 = min(H, y1 + ph)
    z2 = min(D, z1 + pd)

    # Ensure we always get full patch width by sliding start back
    x1 = max(0, x2 - pw)
    y1 = max(0, y2 - ph)
    z1 = max(0, z2 - pd)

    patch = volume[y1:y2, x1:x2, z1:z2]

    # Guarantee exact output shape with zero-padding
    out = np.zeros(patch_shape, dtype=np.float32)
    sh = min(patch.shape[0], ph)
    sw = min(patch.shape[1], pw)
    sd = min(patch.shape[2], pd)
    out[:sh, :sw, :sd] = patch[:sh, :sw, :sd]
    return out

# HELPER: Augmentation for E-class (20x)
# Input ct_2ch: (2,H,W,D), pet_1ch: (1,H,W,D) or None
def augment_volume(ct_2ch, pet_1ch=None, seed=None):
    rng = np.random.default_rng(seed)
    c = ct_2ch.copy()   # (2, H, W, D)

    # Random flip along all 3 spatial axes
    for ax in [1, 2, 3]:
        if rng.random() > 0.5:
            c = np.flip(c, axis=ax).copy()

    # Gaussian noise
    c = np.clip(c + rng.normal(0, 0.02, c.shape).astype(np.float32), 0.0, 1.0)

    # Gamma shift on lung channel only
    gamma = rng.uniform(0.8, 1.2)
    c[0]  = np.power(np.clip(c[0], 1e-6, 1.0), gamma)

    pet_aug = None
    if pet_1ch is not None:
        pet_aug = pet_1ch.copy()
        for ax in [1, 2, 3]:
            if rng.random() > 0.5:
                pet_aug = np.flip(pet_aug, axis=ax).copy()
        # SUV intensity shift ±10%
        pet_aug = np.clip(pet_aug * rng.uniform(0.9, 1.1), 0.0, 1.0).astype(np.float32)

    return c.astype(np.float32), pet_aug

# STEP 1: Load Clinical XLSX
print(SEP)
print("STEP 1: Loading clinical metadata")
print(SEP)

df_cl = pd.read_excel(PATH_XLSX, sheet_name="Clinical Data")
df_cl.columns     = [str(c).strip() for c in df_cl.columns]
df_cl["NewPatientID"] = df_cl["NewPatientID"].astype(str).str.strip().str.upper()
df_cl["cancer_class"] = df_cl["NewPatientID"].apply(get_class)
df_cl["weight_kg"]    = pd.to_numeric(df_cl["weight (kg)"], errors="coerce")

weight_lookup = dict(zip(df_cl["NewPatientID"], df_cl["weight_kg"]))

print(f"  Loaded {len(df_cl)} patients")
print(f"  Classes: {df_cl['cancer_class'].value_counts().to_dict()}")


# STEP 2: Parse XML Annotations → 3D centroid per patient
# FIX BUG 6: filter valid PIDs only (A/B/E/G + 4 digits)
print(f"\n{SEP}")
print("STEP 2: Parsing XML annotations")
print(SEP)

pid_boxes    = defaultdict(list)   # pid → [(cx, cy), ...]
xml_files    = glob.glob(os.path.join(PATH_ANN, "**", "*.xml"), recursive=True)
xml_files   += glob.glob(os.path.join(PATH_ANN, "*.xml"))
skipped_xml  = 0

for xf in xml_files:
    try:
        # Extract PID from path: Annotation/{PID}/xxx.xml
        rel = xf.replace(PATH_ANN, "").strip("/")
        parts = rel.split("/")
        pid = parts[0].upper() if parts else "UNK"

        # FIX BUG 6: only accept valid PID patterns
        if not is_valid_pid(pid):
            skipped_xml += 1
            continue

        tree   = ET.parse(xf)
        root_el = tree.getroot()

        for obj in root_el.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                pid_boxes[pid].append(((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))
            except Exception:
                pass
    except Exception:
        pass

ann_centroids = {}
for pid, boxes in pid_boxes.items():
    ann_centroids[pid] = {
        "cx"      : float(np.mean([b[0] for b in boxes])),
        "cy"      : float(np.mean([b[1] for b in boxes])),
        "n_boxes" : len(boxes)
    }

print(f"  Valid patient annotations parsed: {len(ann_centroids)}")
print(f"  Skipped non-PID XML paths       : {skipped_xml}")

# STEP 3: Discover patients and series
print(f"\n{SEP}")
print("STEP 3: Discovering series")
print(SEP)

patients = {}   # pid → {path, part, ct_series:[], pt_series:[]}

for label, root_path in [("PART1", PATH_PART1), ("PART2", PATH_PART2)]:
    if not os.path.exists(root_path):
        print(f"  [SKIP] {root_path} not found")
        continue
    for folder in sorted(os.listdir(root_path)):
        folder_path = os.path.join(root_path, folder)
        if not os.path.isdir(folder_path):
            continue
        pid = folder.replace("Lung_Dx-", "").upper()
        patients[pid] = {"path": folder_path, "part": label, "ct_series": [], "pt_series": []}

        for study in sorted(os.listdir(folder_path)):
            study_path = os.path.join(folder_path, study)
            if not os.path.isdir(study_path):
                continue
            for series in sorted(os.listdir(study_path)):
                series_path = os.path.join(study_path, series)
                dcm_files   = glob.glob(os.path.join(series_path, "*.dcm"))
                if not dcm_files:
                    continue
                try:
                    ds  = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                    mod = str(getattr(ds, "Modality", "UNK")).upper()
                    st  = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
                    desc = str(getattr(ds, "SeriesDescription", "")).lower()
                    if any(k in desc for k in ["scout","topogram","localizer","surview"]) or st > 20.0:
                        continue
                    if mod == "CT":
                        patients[pid]["ct_series"].append(series_path)
                    elif mod == "PT":
                        patients[pid]["pt_series"].append(series_path)
                except Exception:
                    pass

ct_only = sum(1 for p in patients.values() if p["ct_series"] and not p["pt_series"])
ct_pt   = sum(1 for p in patients.values() if p["ct_series"] and     p["pt_series"])
print(f"  Total patients: {len(patients)} | CT-only: {ct_only} | CT+PET: {ct_pt}")

# STEP 4: CT Preprocessing → 2-channel patches
print(f"\n{SEP}")
print("STEP 4: CT Preprocessing")
print(SEP)

ct_results = {}
failed_ct  = []

for idx, (pid, info) in enumerate(patients.items()):
    if not info["ct_series"]:
        failed_ct.append((pid, "no_ct_series"))
        continue

    # Pick series with the most .dcm files
    best_series = max(info["ct_series"], key=lambda s: len(glob.glob(os.path.join(s, "*.dcm"))))
    volume, meta = load_dicom_volume(best_series)

    if volume is None or volume.ndim != 3:
        print(f"  [{pid}] CT load FAILED — skipping")
        failed_ct.append((pid, "load_failed"))
        continue

    # Patient-specific stats (computed on full volume)
    vol_mean = float(np.mean(volume))
    vol_std  = float(np.std(volume) + 1e-8)

    # Dual-window → 2-channel
    ct_2ch = apply_ct_windows(volume)       # (2, H, W, D)
    assert ct_2ch.ndim == 4 and ct_2ch.shape[0] == 2

    # Annotation centroid (fallback to center)
    cx = ann_centroids.get(pid, {}).get("cx", volume.shape[1] / 2.0)
    cy = ann_centroids.get(pid, {}).get("cy", volume.shape[0] / 2.0)

    # Extract per-channel patch (each channel is (H,W,D) — 3D ✓)
    patch_lung = extract_patch(ct_2ch[0], cx, cy, PATCH_SHAPE)   # (128,128,64)
    patch_medi = extract_patch(ct_2ch[1], cx, cy, PATCH_SHAPE)   # (128,128,64)
    patch_2ch  = np.stack([patch_lung, patch_medi], axis=0)      # (2,128,128,64)

    out_path = os.path.join(OUT_CT, f"{pid}.npy")
    np.save(out_path, patch_2ch)

    ct_results[pid] = {
        "pid"             : pid,
        "cancer_class"    : get_class(pid),
        "part"            : info["part"],
        "ct_path"         : out_path,
        "ct_patch_shape"  : str(list(patch_2ch.shape)),
        "ct_series_path"  : best_series,
        "n_ct_slices"     : meta["n_slices"],
        "original_shape"  : str(meta["original_shape"]),
        "manufacturer"    : meta["manufacturer"],
        "slice_thickness" : meta["slice_thickness"],
        "vol_mean_hu"     : round(vol_mean, 2),
        "vol_std_hu"      : round(vol_std, 2),
        "ann_cx"          : round(cx, 1),
        "ann_cy"          : round(cy, 1),
        "n_ann_boxes"     : ann_centroids.get(pid, {}).get("n_boxes", 0),
        "has_pet"         : len(info["pt_series"]) > 0,
    }

    if (idx + 1) % 50 == 0:
        print(f"  ... CT {idx+1}/{len(patients)}")

print(f"\n  CT patches saved: {len(ct_results)} | Failures: {len(failed_ct)}")
if failed_ct:
    print(f"  Failed: {[f[0] for f in failed_ct]}")

# STEP 5: PET → SUV patches + zero-fills
print(f"\n{SEP}")
print("STEP 5: PET Preprocessing")
print(SEP)

pet_results  = {}
pet_failures = []

for idx, (pid, info) in enumerate(patients.items()):
    w_kg = float(weight_lookup.get(pid, 0) or 0)

    if info["pt_series"]:
        best_pet = max(info["pt_series"], key=lambda s: len(glob.glob(os.path.join(s, "*.dcm"))))
        volume, meta = load_dicom_volume(best_pet)

        if volume is None or volume.ndim != 3:
            print(f"  [{pid}] PET load FAILED → using zeros")
            pet_failures.append(pid)
            patch = np.zeros((1,) + PATCH_SHAPE, dtype=np.float32)
            pet_type = "zero_fill_fail"
            suv_max_v = suv_mean_v = 0.0
            used_fallback = True
            out_dir = OUT_PET_ZERO
        else:
            dose_bq = meta.get("injected_dose_bq", 0.0)
            suv_3d, used_fallback = convert_pet_suv(volume, w_kg, dose_bq)

            cx = ann_centroids.get(pid, {}).get("cx", volume.shape[1] / 2.0)
            cy = ann_centroids.get(pid, {}).get("cy", volume.shape[0] / 2.0)

            p3d   = extract_patch(suv_3d, cx, cy, PATCH_SHAPE)    # (128,128,64)
            patch = p3d[np.newaxis, ...]                           # (1,128,128,64)

            suv_max_v  = round(float(patch.max()),  4)
            suv_mean_v = round(float(patch.mean()), 6)
            pet_type   = "real"
            out_dir    = OUT_PET

        out_path = os.path.join(out_dir, f"{pid}.npy")
        np.save(out_path, patch)
    else:
        # CT-only patient → zero fill
        patch = np.zeros((1,) + PATCH_SHAPE, dtype=np.float32)
        out_path = os.path.join(OUT_PET_ZERO, f"{pid}.npy")
        np.save(out_path, patch)
        pet_type = "zero_fill"
        suv_max_v = suv_mean_v = 0.0
        used_fallback = True

    pet_results[pid] = {
        "pid"          : pid,
        "pet_path"     : out_path,
        "pet_type"     : pet_type,
        "has_real_pet" : pet_type == "real",
        "pet_shape"    : str(list(patch.shape)),
        "suv_max"      : suv_max_v if "suv_max_v" in dir() else 0.0,
        "suv_mean"     : suv_mean_v if "suv_mean_v" in dir() else 0.0,
        "suv_fallback" : used_fallback,
    }

    if (idx + 1) % 50 == 0:
        print(f"  ... PET {idx+1}/{len(patients)}")

real_c  = sum(1 for v in pet_results.values() if v["pet_type"] == "real")
zero_c  = sum(1 for v in pet_results.values() if "zero" in v["pet_type"])
print(f"\n  Real PET: {real_c} | Zero-fill: {zero_c} | Failures: {len(pet_failures)}")



# STEP 6: E-prefix 20× Augmentation (Large Cell, 5 patients)

print(f"\n{SEP}")
print("STEP 6: E-prefix 20× Augmentation")
print(SEP)

e_patients  = sorted([pid for pid in patients if pid.upper().startswith("E")])
aug_records = []

for pid in e_patients:
    ct_file  = os.path.join(OUT_CT, f"{pid}.npy")
    pet_file = os.path.join(OUT_PET, f"{pid}.npy")
    if not os.path.exists(pet_file):
        pet_file = os.path.join(OUT_PET_ZERO, f"{pid}.npy")

    if not os.path.exists(ct_file):
        print(f"  SKIP {pid} — CT not found")
        continue

    ct_2ch  = np.load(ct_file)                                 # (2, 128, 128, 64)
    pet_1ch = np.load(pet_file) if os.path.exists(pet_file) else None   # (1, 128, 128, 64)

    for aug_i in range(20):
        seed    = (aug_i * 997) + (abs(hash(pid)) % 10000)
        ct_aug, pet_aug = augment_volume(ct_2ch, pet_1ch, seed=seed)

        ct_aug_path = os.path.join(OUT_AUG, f"{pid}_aug{aug_i:02d}_ct.npy")
        np.save(ct_aug_path, ct_aug)

        aug_rec = {
            "patient_id"   : f"{pid}_aug{aug_i:02d}",
            "cancer_class" : "Large Cell Carcinoma",
            "part"         : "AUG",
            "ct_path"      : ct_aug_path,
            "is_augmented" : True,
            "split"        : "train",
            "has_real_pet" : pet_aug is not None and pet_1ch is not None and pet_results.get(pid, {}).get("has_real_pet", False),
        }
        if pet_aug is not None:
            pet_aug_path = os.path.join(OUT_AUG, f"{pid}_aug{aug_i:02d}_pet.npy")
            np.save(pet_aug_path, pet_aug)
            aug_rec["pet_path"] = pet_aug_path
        else:
            aug_rec["pet_path"] = ""

        aug_records.append(aug_rec)

    print(f"  {pid}: 20 CT augmented samples saved")

print(f"  Total augmented E samples: {len(aug_records)}")



# STEP 7: Build master manifest

print(f"\n{SEP}")
print("STEP 7: Building manifest.csv")
print(SEP)

rows = []
for pid, ct_info in ct_results.items():
    pet_info = pet_results.get(pid, {})
    clin     = df_cl[df_cl["NewPatientID"] == pid]

    def cval(col):
        try:
            v = clin[col].values[0]
            return None if pd.isna(v) else v
        except Exception:
            return None

    rows.append({
        "patient_id"      : pid,
        "cancer_class"    : ct_info["cancer_class"],
        "part"            : ct_info["part"],
        "ct_path"         : ct_info["ct_path"],
        "pet_path"        : pet_info.get("pet_path", ""),
        "pet_type"        : pet_info.get("pet_type", "unknown"),
        "has_real_pet"    : bool(pet_info.get("has_real_pet", False)),
        "ct_patch_shape"  : ct_info["ct_patch_shape"],
        "pet_patch_shape" : pet_info.get("pet_shape", ""),
        "manufacturer"    : ct_info.get("manufacturer", ""),
        "n_ct_slices"     : ct_info.get("n_ct_slices", 0),
        "slice_thickness" : ct_info.get("slice_thickness", 0),
        "n_ann_boxes"     : ct_info.get("n_ann_boxes", 0),
        "ann_cx"          : ct_info.get("ann_cx", 0),
        "ann_cy"          : ct_info.get("ann_cy", 0),
        "suv_max"         : pet_info.get("suv_max", 0),
        "suv_fallback"    : bool(pet_info.get("suv_fallback", True)),
        "sex"             : cval("Sex"),
        "age"             : cval("Age"),
        "weight_kg"       : cval("weight (kg)"),
        "t_stage"         : cval("T-Stage"),
        "n_stage"         : cval("N-Stage"),
        "m_stage"         : cval("Ｍ-Stage"),
        "smoking"         : cval("Smoking History"),
        "is_augmented"    : False,
        "split"           : "",
    })

df_manifest = pd.DataFrame(rows)


# STEP 8: Stratified splits + LOOCV for E
print(f"\n{SEP}")
print("STEP 8: Stratified splits + LOOCV")
print(SEP)

non_e_df = df_manifest[df_manifest["cancer_class"] != "Large Cell Carcinoma"].copy()
e_df     = df_manifest[df_manifest["cancer_class"] == "Large Cell Carcinoma"].copy()

for cls, grp in non_e_df.groupby("cancer_class"):
    idx = grp.index.tolist()
    if len(idx) < 3:
        non_e_df.loc[idx, "split"] = "train"
        continue
    tr, tmp  = train_test_split(idx, test_size=0.30, random_state=42)
    val, te  = train_test_split(tmp, test_size=0.50, random_state=42)
    non_e_df.loc[tr,  "split"] = "train"
    non_e_df.loc[val, "split"] = "val"
    non_e_df.loc[te,  "split"] = "test"

e_df = e_df.copy()
e_df["split"] = "loocv"
e_pids = sorted(e_df["patient_id"].tolist())

for fold_i, test_pid in enumerate(e_pids):
    loocv_data = {
        "fold"       : fold_i,
        "test_pid"   : test_pid,
        "train_pids" : [p for p in e_pids if p != test_pid],
    }
    with open(os.path.join(OUT_LOOCV, f"fold{fold_i}.json"), "w") as f:
        json.dump(loocv_data, f, indent=2)
    print(f"  LOOCV fold {fold_i}: test={test_pid} | train={loocv_data['train_pids']}")

df_manifest_final = pd.concat([non_e_df, e_df], ignore_index=True)

# FIX BUG 7: align augmented rows with all manifest columns before concat
if aug_records:
    all_cols = list(df_manifest_final.columns)
    aug_aligned = []
    for ar in aug_records:
        row = {col: ar.get(col, None) for col in all_cols}
        aug_aligned.append(row)
    df_aug = pd.DataFrame(aug_aligned, columns=all_cols)
    df_manifest_final = pd.concat([df_manifest_final, df_aug], ignore_index=True)

manifest_path = os.path.join(OUT_META, "manifest.csv")
df_manifest_final.to_csv(manifest_path, index=False)

# Split summary JSON
split_summary = {}
for cls in df_manifest_final["cancer_class"].unique():
    sub = df_manifest_final[df_manifest_final["cancer_class"] == cls]
    split_summary[str(cls)] = {sp: int((sub["split"] == sp).sum()) for sp in sub["split"].unique()}

with open(os.path.join(OUT_META, "split_summary.json"), "w") as f:
    json.dump(split_summary, f, indent=2)

print(f"\n  Manifest rows: {len(df_manifest_final)}")



# STEP 9: FULL VALIDATION REPORT (share this output)

print(f"\n{SEP}")
print("STEP 9: PART 5 VALIDATION REPORT — Share this output")
print(SEP)

def dir_stats(path):
    files = [f for f in glob.glob(f"{path}/**", recursive=True) if os.path.isfile(f)]
    size  = sum(os.path.getsize(f) for f in files) / (1024**3)
    return len(files), round(size, 4)

print(f"\n{'─'*60}")
print("DIRECTORY STRUCTURE & SIZES")
print(f"{'─'*60}")
for name, path in [("ct_patches", OUT_CT), ("pet_patches", OUT_PET),
                    ("pet_zeros",  OUT_PET_ZERO), ("augmented_e", OUT_AUG),
                    ("loocv_splits", OUT_LOOCV),  ("metadata",   OUT_META)]:
    nf, sz = dir_stats(path)
    print(f"  {name:<22}: {nf:>5} files | {sz:.4f} GB")

total_gb = dir_stats(OUT_ROOT)[1]
print(f"\n  TOTAL Part 5 size          : {total_gb:.4f} GB")

print(f"\n{'─'*60}")
print("MANIFEST STATISTICS")
print(f"{'─'*60}")
df_m = pd.read_csv(manifest_path)
print(f"\n  Total manifest rows        : {len(df_m)}")
print(f"  Original patients (no aug) : {df_m[~df_m['is_augmented']]['patient_id'].nunique()}")
print(f"\n  Class distribution (all):")
print(df_m["cancer_class"].value_counts().to_string())
print(f"\n  Split distribution (non-augmented):")
print(df_m[~df_m["is_augmented"]]["split"].value_counts().to_string())
print(f"\n  PET type distribution:")
print(df_m["pet_type"].value_counts().to_string())
print(f"\n  Real PET patients by class:")
print(df_m[df_m["has_real_pet"] == True]["cancer_class"].value_counts().to_string())
print(f"\n  Manufacturers:")
print(df_m["manufacturer"].value_counts().to_string())
print(f"\n  Patients without annotation (using volume-center fallback):")
no_ann = df_m[(df_m["n_ann_boxes"] == 0) & (~df_m["is_augmented"])]
print(f"  Count: {len(no_ann)} | PIDs: {list(no_ann['patient_id'])[:20]}")
print(f"\n  SUV fallback (no dose tag in DICOM):")
print(df_m[df_m["suv_fallback"] == True]["cancer_class"].value_counts().to_string())

print(f"\n{'─'*60}")
print("SPLIT TABLE BY CLASS")
print(f"{'─'*60}")
non_aug = df_m[~df_m["is_augmented"]]
try:
    pivot = non_aug.groupby(["cancer_class", "split"]).size().unstack(fill_value=0)
    print(pivot.to_string())
except Exception as e:
    print(f"  (pivot error: {e})")
    print(non_aug.groupby(["cancer_class", "split"]).size().to_string())

print(f"\n{'─'*60}")
print("LOOCV FOLDS — Large Cell Carcinoma")
print(f"{'─'*60}")
for fold_i in range(len(e_pids)):
    fpath = os.path.join(OUT_LOOCV, f"fold{fold_i}.json")
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        print(f"  Fold {d['fold']}: TEST={d['test_pid']} | TRAIN={d['train_pids']}")

print(f"\n{'─'*60}")
print("CT PATCH SAMPLE STATS (first 5 patients)")
print(f"{'─'*60}")
for pid in list(ct_results.keys())[:5]:
    fp = os.path.join(OUT_CT, f"{pid}.npy")
    if os.path.exists(fp):
        p = np.load(fp)
        print(f"  {pid}: shape={p.shape} | ch0=[{p[0].min():.4f},{p[0].max():.4f}]"
              f" | ch1=[{p[1].min():.4f},{p[1].max():.4f}] | mean={p.mean():.4f}")

print(f"\n{'─'*60}")
print("PET PATCH SAMPLE STATS (first 5 real PET patients)")
print(f"{'─'*60}")
real_pet_pids = [pid for pid, v in pet_results.items() if v["pet_type"] == "real"][:5]
for pid in real_pet_pids:
    fp = pet_results[pid]["pet_path"]
    if os.path.exists(fp):
        p = np.load(fp)
        print(f"  {pid}: shape={p.shape} | SUV range=[{p.min():.4f},{p.max():.4f}]"
              f" | mean={p.mean():.5f} | fallback={pet_results[pid]['suv_fallback']}")

print(f"\n{'─'*60}")
print("E-CLASS AUGMENTED SAMPLES")
print(f"{'─'*60}")
for pid in e_pids:
    ct_aug_files  = glob.glob(os.path.join(OUT_AUG, f"{pid}_aug*_ct.npy"))
    pet_aug_files = glob.glob(os.path.join(OUT_AUG, f"{pid}_aug*_pet.npy"))
    print(f"  {pid}: {len(ct_aug_files)} CT aug | {len(pet_aug_files)} PET aug")

print(f"\n{'─'*60}")
print("CLINICAL DATA COVERAGE")
print(f"{'─'*60}")
real_pts = df_m[~df_m["is_augmented"]]
print(f"  Age coverage    : {real_pts['age'].notna().sum()}/{len(real_pts)}")
print(f"  Sex coverage    : {real_pts['sex'].notna().sum()}/{len(real_pts)}")
print(f"  T-stage coverage: {real_pts['t_stage'].notna().sum()}/{len(real_pts)}")
print(f"  M-stage coverage: {real_pts['m_stage'].notna().sum()}/{len(real_pts)}")
print(f"  Smoking coverage: {real_pts['smoking'].notna().sum()}/{len(real_pts)}")

print(f"\n{'─'*60}")
print("FAILED / SKIPPED PATIENTS")
print(f"{'─'*60}")
all_pids  = set(patients.keys())
proc_pids = set(ct_results.keys())
skipped   = sorted(all_pids - proc_pids)
print(f"  CT processed    : {len(proc_pids)}/{len(all_pids)}")
print(f"  Skipped/failed  : {skipped if skipped else 'None'}")

print(f"\n{SEP}")
print("PART 5 COMPLETE ✅")
print(f"  Output: {OUT_ROOT}")
print(f"  Total  : {total_gb:.4f} GB")
print(f"  Upload to Kaggle as: 'Cancer Image Dataset Part 5 - Preprocessed'")
print(SEP)

