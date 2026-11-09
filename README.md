# Lung Cancer Imaging Data Processing

Processing pipeline for the **TCIA Lung-PET-CT-Dx** dataset.

Cancer Imaging Archive: https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/

The dataset contains CT and PET DICOM studies labelled with four lung cancer
histology classes, derived from the patient-ID prefix:

| Prefix | Class |
|--------|-------|
| A | Adenocarcinoma |
| B | Small Cell Carcinoma |
| E | Large Cell Carcinoma |
| G | Squamous Cell Carcinoma |

## Repository Structure

```
data_processing/
├── Tciad-processing/                       # DICOM inspection / metadata extraction
├── Automated_Pipeline/                     # End-to-end preprocessing pipeline
├── Cancer_EDA/                             # Exploratory data analysis + report
├── Model_Training/                         # Model training and validation
├── Lung Cancer Classification Using EfficientNet-B0/  # EfficientNet-B0 notebook
├── Predictrion_App/                        # Streamlit prediction app
└── README.md
```

## Folders

### Tciad-processing
[tciad-processing.py](file:///c:/Users/tejaj/Documents/New%20folder/data_processing/Tciad-processing/tciad-processing.py)

Initial dataset inspection. Reads a DICOM series with `pydicom`, prints
metadata (patient ID, modality, study date, manufacturer, image size), renders
sample slices with matplotlib, and builds a per-file metadata table
(min/max/mean/std pixel values, slice thickness, pixel spacing, series/study
UIDs). Writes `dicom_metadata.csv`. Used to understand the raw data before
building the pipeline.

### Automated_Pipeline
[automated-multimodal-ct-pet-preprocessing-pipeline.py](file:///c:/Users/tejaj/Documents/New%20folder/data_processing/Automated_Pipeline/automated-multimodal-ct-pet-preprocessing-pipeline.py)

The core preprocessing pipeline that turns raw DICOM studies into fixed-size
NumPy patches plus a training manifest. Steps:

1. Load clinical metadata from the statistics/clinical XLSX.
2. Parse XML bounding-box annotations into a 3D centroid per patient.
3. Discover patients and their CT / PET series (skipping scout / topogram series).
4. CT preprocessing: dual-window (lung + mediastinal) → 2-channel patch
   `(2, 128, 128, 64)`, values in `[0, 1]`.
5. PET preprocessing: SUV-normalized patch `(1, 128, 128, 64)`; CT-only patients
   get a zero-filled PET patch.
6. 20× augmentation for the under-represented Large Cell (E) class.
7. Build `manifest.csv` (one row per sample).
8. Stratified train/val/test splits for A/B/G plus Leave-One-Patient-Out
   (LOOCV) folds for the E class.
9. Print a full validation report of the produced artifacts.

Outputs: `ct_patches/`, `pet_patches/`, `pet_zeros/`, `augmented_e/`,
`loocv_splits/`, `metadata/manifest.csv`, `metadata/split_summary.json`.

Dataset layout note: the script expects Kaggle-style input paths
(`/kaggle/input/...`) and writes to `/kaggle/working/part5`.

### Cancer_EDA
[cancer-eda.py](file:///c:/Users/tejaj/Documents/New%20folder/data_processing/Cancer_EDA/cancer-eda.py)

Exploratory data analysis over `manifest.csv`. Produces class distribution,
PET availability, train/val/test composition, clinical/demographic plots (age,
sex, smoking, T-stage), and CT/PET characteristics (scanner manufacturer, slice
thickness, SUVmax distribution). Figures are saved to `eda_figures/` and
assembled into a Word report `part5_EDA_report.docx` via `python-docx`.

The manifest path (`MANIFEST_PATH`) must be pointed at the generated
`metadata/manifest.csv` before running.

### Model_Training
[model_training.py](file:///c:/Users/tejaj/Documents/New%20folder/data_processing/Model_Training/model_training.py)

Verification and training-readiness check for the preprocessed Part 5 dataset.
Validates file counts, patch shapes/ranges, NaN/Inf absence, augmented-sample
integrity, LOOCV fold validity, clinical coverage, and runs a DataLoader
simulation (fusing CT + PET into `(3, 128, 128, 64)`). Prints a pass/fail
report and the canonical data-loading snippet.

### Lung Cancer Classification Using EfficientNet-B0
[lung-cancer-classification-using-efficientnet-b0.py](file:///c:/Users/tejaj/Documents/New%20folder/data_processing/Lung%20Cancer%20Classification%20Using%20EfficientNet-B0/lung-cancer-classification-using-efficientnet-b0.py)

Main training notebook. Fine-tunes **EfficientNet-B0** (ImageNet-pretrained,
`timm`) for 4-class histology classification from CT slices, using a two-stage
transfer-learning strategy (frozen-backbone warm-up, then end-to-end
fine-tuning). Includes CT-specific normalization, augmentation, MixUp,
class-balanced loss weighting (effective number, Cui et al. 2019), label
smoothing, and cosine LR scheduling with warm-up. Evaluated with
train/val/test metrics (accuracy, macro F1, ROC-AUC, per-class report,
confusion matrix) and Leave-One-Patient-Out cross-validation on the Large Cell
class. Checkpoints and metrics are saved and pushed to Hugging Face Hub.

### Predictrion_App
[lung-cancer-prediction-app.py](file:///c:/Users/tejaj/Documents/New%20folder/data_processing/Predictrion_App/lung-cancer-prediction-app.py)

Generates and launches a **Streamlit** prediction app for an ensemble of three
models:

- **M1** — EfficientNetV2-L (five E-LOOCV fold checkpoints, 2D slices + flip TTA)
- **M2** — ConvNeXt-Large (2D slices with PET modality embedding)
- **M3** — Swin3D-S (3D CT/PET volume with four-view TTA)

Downloads the seven checkpoints from Hugging Face Hub, writes `app.py`, and
serves it via a Cloudflare quick tunnel. Inputs are CT `(2,128,128,64)` and PET
`(1,128,128,64)`; the final prediction is a weighted average of the three
models. Intended for research use only — not a clinical diagnostic tool.

## Typical Workflow

Run the stages in order; each stage consumes the previous stage's outputs.

| Step | Folder | Script | Produces |
|------|--------|--------|----------|
| 1. Inspect raw data | `Tciad-processing/` | `tciad-processing.py` | `dicom_metadata.csv`, sample slice previews |
| 2. Preprocess | `Automated_Pipeline/` | `automated-multimodal-ct-pet-preprocessing-pipeline.py` | CT/PET patches, `manifest.csv`, train/val/test + LOOCV splits |
| 3. Verify readiness | `Model_Training/` | `model_training.py` | Pass/fail dataset validation report |
| 4. Explore + report | `Cancer_EDA/` | `cancer-eda.py` | `eda_figures/`, `part5_EDA_report.docx` |
| 5. Train + validate | `Lung Cancer Classification Using EfficientNet-B0/` | `lung-cancer-classification-using-efficientnet-b0.py` | Checkpoints and metrics (pushed to Hugging Face Hub) |
| 6. Inference | `Predictrion_App/` | `lung-cancer-prediction-app.py` | Streamlit app served through a Cloudflare tunnel |

Steps 2 and 3 are designed to run inside the Kaggle environment (or locally
after adjusting the path constants). Step 2 must finish before step 3, 4, or 5
can run, since all of them read `manifest.csv` and the generated patch files.

## Data Conventions

| Artifact | Shape / dtype | Notes |
|----------|---------------|-------|
| CT patch | `(2, 128, 128, 64)` float32 | Channel 0 = lung window, channel 1 = mediastinal window; values in `[0, 1]` |
| PET patch | `(1, 128, 128, 64)` float32 | SUV-normalized to `[0, 1]` |
| Zero PET patch | `(1, 128, 128, 64)` float32 | All zeros; used for CT-only patients |
| Fused model input | `(3, 128, 128, 64)` float32 | CT (2 channels) + PET (1 channel) concatenated on the channel axis |

- Spatial patch size: 128 x 128 pixels over 64 slices (depth).
- Label order used by the classifier: `[Adenocarcinoma, Small Cell, Squamous, Large Cell]`.
- Splits: stratified train/val/test for classes A/B/G, plus Leave-One-Patient-Out
  (LOOCV) folds for the under-represented Large Cell (E) class.
- `manifest.csv` is the single source of truth linking patient ID, class label,
  patch file paths, and split assignment.

## Notes

- **Paths:** the pipeline scripts target Kaggle (`/kaggle/input/...` and
  `/kaggle/working/part5`). Update the path constants in each script when
  running outside Kaggle.
- **Secrets:** a Hugging Face token is referenced in the training and prediction
  scripts. Move it to an environment variable or a Kaggle secret, and rotate any
  token that has already been committed to git.
- **Environment:** requires Python 3 with `pydicom`, `numpy`, `pandas`,
  `matplotlib`, `openpyxl`, `python-docx`, `scikit-learn`, `torch`, `timm`, and
  `streamlit`.
- **Data:** download the TCIA Lung-PET-CT-Dx collection together with its
  clinical and annotation files before running the pipeline.
- **Disclaimer:** this project is for research and educational use only.
  Predictions are not a medical diagnosis and must not be used for clinical
  decision-making.
