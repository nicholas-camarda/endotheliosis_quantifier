from eq.utils.paths import get_runtime_mitochondria_data_path

# Build dataloader WITHOUT the MaskPreprocessTransform
data_dir = get_runtime_mitochondria_data_path() / "training"

# Let's check the raw mask files first
from eq.data_management.standard_getters import get_y_full
from PIL import Image
import numpy as np

# Get a sample image path
images_dir = data_dir / "images"
sample_img = list(images_dir.glob("*.tif"))[0]
mask_path = get_y_full(sample_img)

print("=== RAW MASK FILE INSPECTION ===")
print(f"Sample mask path: {mask_path}")

# Load raw mask
raw_mask = Image.open(mask_path)
raw_array = np.array(raw_mask)
print(f"Raw mask shape: {raw_array.shape}")
print(f"Raw mask dtype: {raw_array.dtype}")
print(f"Raw mask unique values: {np.unique(raw_array)}")
print(f"Raw mask min/max: {raw_array.min()}/{raw_array.max()}")

if np.unique(raw_array).tolist() == [0, 1]:
    print("✅ Raw mask files are already 0/1 - no preprocessing needed")
elif 255 in np.unique(raw_array):
    print("❌ Raw mask files contain 255 - preprocessing needed")
else:
    print(f"⚠️  Raw mask has unexpected values: {np.unique(raw_array)}")
