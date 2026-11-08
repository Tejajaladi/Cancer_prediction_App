
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 120,
})

MANIFEST_PATH = r"C:\path\to\part5\metadata\manifest.csv"
FIG_DIR = "eda_figures"
OUT_PATH = "part5_EDA_report.docx"

os.makedirs(FIG_DIR, exist_ok=True)
SAVED_FIGURES = []

CLASS_COLORS = {
    "Adenocarcinoma": "#2b6cb0",
    "Small Cell Carcinoma": "#c53030",
    "Large Cell Carcinoma": "#dd6b20",
    "Squamous Cell Carcinoma": "#2f855a"
}

classes = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Small Cell Carcinoma",
    "Squamous Cell Carcinoma"
]


def save_fig(fig, name, title):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    SAVED_FIGURES.append((title, path))
    plt.close(fig)


df = pd.read_csv(MANIFEST_PATH)

print("Dataset shape:", df.shape)

n_records = len(df)
unique_df = df[~df["is_augmented"]].copy()
n_unique_patients = unique_df["patient_id"].apply(
    lambda x: x.split("_aug")[0]
).nunique()
n_augmented = int(df["is_augmented"].sum())

print("Total records:", n_records)
print("Unique patients:", n_unique_patients)
print("Augmented records:", n_augmented)

print("\nRecords per class:")
print(df["cancer_class"].value_counts())

print("\nUnique patients per class:")
print(unique_df["cancer_class"].value_counts())


# Class distribution
record_counts = df["cancer_class"].value_counts().reindex(
    classes, fill_value=0
)

patient_counts = unique_df["cancer_class"].value_counts().reindex(
    classes, fill_value=0
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = [CLASS_COLORS[c] for c in classes]

axes[0].bar(classes, record_counts.values, color=colors)
axes[0].set_title("Prepared Records per Class")
axes[0].set_ylabel("Number of records")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(classes, patient_counts.values, color=colors)
axes[1].set_title("Unique Patients per Class")
axes[1].set_ylabel("Number of patients")
axes[1].tick_params(axis="x", rotation=20)

fig.tight_layout()
save_fig(fig, "class_distribution", "Class Distribution")


# PET availability

pet_by_class = (
    unique_df.groupby(["cancer_class", "has_real_pet"])
    .size()
    .unstack(fill_value=0)
    .reindex(classes, fill_value=0)
)

fig, ax = plt.subplots(figsize=(8, 4.5))

bottom = np.zeros(len(classes))

for col, color, label in [
    (True, "#2f855a", "Real PET"),
    (False, "#a0aec0", "Zero-filled PET")
]:
    vals = pet_by_class[col].values if col in pet_by_class.columns else np.zeros(len(classes))
    ax.bar(classes, vals, bottom=bottom, label=label, color=color)
    bottom += vals

ax.set_ylabel("Number of patients")
ax.set_title("PET Availability by Class")
ax.tick_params(axis="x", rotation=20)
ax.legend(frameon=False)

fig.tight_layout()
save_fig(fig, "pet_availability", "PET Availability by Class")

pct_real = 100 * unique_df["has_real_pet"].mean()
print(f"Patients with real PET: {pct_real:.1f}%")


# Train, validation and test split
abg_df = df[df["split"].isin(["train", "val", "test"])]

split_by_class = (
    abg_df.groupby(["cancer_class", "split"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=classes, columns=["train", "val", "test"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(split_by_class.index))
width = 0.25

for i, sp in enumerate(["train", "val", "test"]):
    ax.bar(
        x + (i - 1) * width,
        split_by_class[sp].values,
        width,
        label=sp.capitalize()
    )

ax.set_xticks(x)
ax.set_xticklabels(split_by_class.index, rotation=15)
ax.set_ylabel("Number of patients")
ax.set_title("Train / Validation / Test Split")
ax.legend(frameon=False)

fig.tight_layout()
save_fig(fig, "split_composition", "Train Validation Test Split")

print(split_by_class)


# Clinical and demographic analysis
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].hist(
    unique_df["age"].dropna(),
    bins=15,
    color="#2b6cb0",
    edgecolor="white"
)
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")

sex_counts = unique_df["sex"].value_counts()
axes[0, 1].pie(
    sex_counts.values,
    labels=sex_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
axes[0, 1].set_title("Sex Distribution")

smoking_counts = unique_df["smoking"].value_counts().reindex(
    ["Current", "Former", "Never"], fill_value=0
)
axes[1, 0].bar(smoking_counts.index, smoking_counts.values)
axes[1, 0].set_title("Smoking History")

t_counts = unique_df["t_stage"].value_counts().reindex(
    ["T1", "T2", "T3", "T4"], fill_value=0
)
axes[1, 1].bar(t_counts.index, t_counts.values)
axes[1, 1].set_title("T-Stage Distribution")

fig.tight_layout()
save_fig(fig, "clinical_demographics", "Clinical and Demographic Characteristics")


# CT and PET characteristics
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

mfr_counts = unique_df["manufacturer"].value_counts()
axes[0].bar(mfr_counts.index, mfr_counts.values, color="#2f855a")
axes[0].set_title("CT Scanner Manufacturer")
axes[0].tick_params(axis="x", rotation=30)

axes[1].hist(
    unique_df["slice_thickness"].dropna(),
    bins=10,
    color="#2b6cb0"
)
axes[1].set_title("CT Slice Thickness")
axes[1].set_xlabel("Slice thickness (mm)")

real_pet_suv = unique_df.loc[
    unique_df["has_real_pet"], "suv_max"
].dropna()

axes[2].hist(real_pet_suv, bins=12, color="#dd6b20")
axes[2].set_title(f"SUVmax Distribution (n={len(real_pet_suv)})")
axes[2].set_xlabel("SUVmax")

fig.tight_layout()
save_fig(fig, "ct_pet_characteristics", "CT and PET Characteristics")

print("Mean SUVmax:", real_pet_suv.mean())
print("Mean slice thickness:", unique_df["slice_thickness"].mean())
print("Mean CT slices per patient:", unique_df["n_ct_slices"].mean())


# Patient lookup
CLASS_FILTER = "Adenocarcinoma"

lookup_df = unique_df[
    ["patient_id", "cancer_class", "pet_type", "has_real_pet", "split"]
].copy()

lookup_df["pet_status"] = lookup_df["has_real_pet"].map({
    True: "Real PET",
    False: "Zero-filled PET"
})

lookup_df = lookup_df.drop(
    columns=["has_real_pet"]
).sort_values(
    ["cancer_class", "patient_id"]
).reset_index(drop=True)

if CLASS_FILTER is not None:
    filtered_view = lookup_df[
        lookup_df["cancer_class"] == CLASS_FILTER
    ].reset_index(drop=True)
else:
    filtered_view = lookup_df

print("\nPatient lookup:")
print(filtered_view.head(20))

print("\nPET status breakdown:")
print(filtered_view["pet_status"].value_counts())


# Word report
doc = Document()

doc.add_heading(
    "Lung-PET-CT-Dx part5 - Exploratory Data Analysis",
    level=0
)

doc.add_paragraph(
    f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}"
)

doc.add_heading("Summary Statistics", level=1)

summary_table = doc.add_table(rows=1, cols=2)
summary_table.style = "Light Grid Accent 1"

summary_table.rows[0].cells[0].text = "Metric"
summary_table.rows[0].cells[1].text = "Value"

summary_rows = [
    ("Total prepared records", str(n_records)),
    ("Unique patients", str(n_unique_patients)),
    ("Augmented records", str(n_augmented)),
    ("Patients with real PET", f"{pct_real:.1f}%")
]

for key, value in summary_rows:
    cells = summary_table.add_row().cells
    cells[0].text = key
    cells[1].text = value


doc.add_heading("Patient Lookup", level=1)

doc.add_paragraph(
    f"Class filter: {CLASS_FILTER or 'All classes'}"
)

lookup_table = doc.add_table(rows=1, cols=len(filtered_view.columns))
lookup_table.style = "Light Grid Accent 1"

for i, col in enumerate(filtered_view.columns):
    lookup_table.rows[0].cells[i].text = str(col)

for _, row in filtered_view.head(20).iterrows():
    cells = lookup_table.add_row().cells
    for i, value in enumerate(row):
        cells[i].text = str(value)


doc.add_heading("Charts", level=1)

for chart_title, img_path in SAVED_FIGURES:
    doc.add_heading(chart_title, level=2)
    doc.add_picture(img_path, width=Inches(6))


doc.save(OUT_PATH)

print(f"\nReport saved: {OUT_PATH}")
print(f"Charts saved in: {FIG_DIR}")
