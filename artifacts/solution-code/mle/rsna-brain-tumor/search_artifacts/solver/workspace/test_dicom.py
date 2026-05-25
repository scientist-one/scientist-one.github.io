import pydicom
import os

def load_dicom_volume(path):
    files = [f for f in os.listdir(path) if f.endswith('.dcm')]
    slices = [pydicom.dcmread(os.path.join(path, f)) for f in files]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices

path = '/data/train/00000/FLAIR'
slices = load_dicom_volume(path)
print(f"Number of slices: {len(slices)}")
print(f"Slice shape: {slices[0].pixel_array.shape}")
