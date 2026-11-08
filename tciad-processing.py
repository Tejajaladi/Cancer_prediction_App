import os
import random
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# List files in current directory
print(os.listdir())

# Dataset folder path
folder_path = '/kaggle/input/datasets/tejajaladi/cancer-image-dataset-part2/Lung_Dx-G0004/07-05-2008-NA-lung-81791/2.000000-A phase 5mm Stnd SS50-71037'

print("Dataset folder contents:")
print(os.listdir(folder_path))

# Get DICOM files
dcm_files = [
    f for f in os.listdir(folder_path)
    if f.endswith(".dcm")
]

print("Total DICOM files:", len(dcm_files))

# Read DICOM files and display images
for file in dcm_files:

    file_path = os.path.join(folder_path, file)

    # Read DICOM
    dicom = pydicom.dcmread(file_path)

    print("\n" + "=" * 50)
    print("File Name:", file)

    # Metadata
    print("Patient ID      :", dicom.get("PatientID", "N/A"))
    print("Modality        :", dicom.get("Modality", "N/A"))
    print("Study Date      :", dicom.get("StudyDate", "N/A"))
    print("Image Size      :", dicom.get("Rows", "N/A"), "x",
          dicom.get("Columns", "N/A"))
    print("Manufacturer    :", dicom.get("Manufacturer", "N/A"))

    # Display image
    try:
        image = dicom.pixel_array

        image = image.astype(np.float32)

        # Normalize image
        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)

        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap="gray")
        plt.title(file)
        plt.axis("off")
        plt.show()

    except Exception as e:
        print("Error displaying image:", e)


# Extract DICOM metadata and image statistics

dataset_info = []

for file in dcm_files:

    file_path = os.path.join(folder_path, file)

    try:
        dicom = pydicom.dcmread(file_path)

        # Image array
        image = dicom.pixel_array

        info = {
            "File": file,
            "Modality": dicom.get("Modality", "N/A"),
            "Rows": dicom.get("Rows", "N/A"),
            "Columns": dicom.get("Columns", "N/A"),
            "Pixel Min": np.min(image),
            "Pixel Max": np.max(image),
            "Pixel Mean": np.mean(image),
            "Pixel Std": np.std(image),
            "Bits Stored": dicom.get("BitsStored", "N/A"),
            "Photometric Interpretation": dicom.get(
                "PhotometricInterpretation", "N/A"
            ),
            "Samples Per Pixel": dicom.get("SamplesPerPixel", "N/A"),
            "Slice Thickness": dicom.get("SliceThickness", "N/A"),
            "Pixel Spacing": dicom.get("PixelSpacing", "N/A"),
            "Image Position": dicom.get("ImagePositionPatient", "N/A"),
            "Image Orientation": dicom.get(
                "ImageOrientationPatient", "N/A"
            ),
            "Series Instance UID": dicom.get("SeriesInstanceUID", "N/A"),
            "Study Instance UID": dicom.get("StudyInstanceUID", "N/A"),
        }

        dataset_info.append(info)

    except Exception as e:
        print(f"Error reading {file}: {e}")


# Convert metadata into DataFrame
df = pd.DataFrame(dataset_info)

print("\nDataset Summary")
print(df.head())

print("\nColumns in metadata:")
print(df.columns)

# Save metadata
df.to_csv("dicom_metadata.csv", index=False)

print("\nMetadata saved to dicom_metadata.csv")


# Dataset shape and information
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())


# Display 6 random DICOM images
if len(dcm_files) > 0:

    random_files = random.sample(
        dcm_files,
        min(6, len(dcm_files))
    )

    plt.figure(figsize=(15, 10))

    for i, file in enumerate(random_files):

        file_path = os.path.join(folder_path, file)

        dicom = pydicom.dcmread(file_path)

        image = dicom.pixel_array

        plt.subplot(2, 3, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(file[:25])
        plt.axis("off")

    plt.tight_layout()
    plt.show()



# Read first DICOM image and check shape


if len(dcm_files) > 0:

    dicom = pydicom.dcmread(
        os.path.join(folder_path, dcm_files[0])
    )

    image = dicom.pixel_array

    print("\nFirst DICOM image shape:")
    print(image.shape)

else:
    print("No DICOM files found in the dataset folder.")
