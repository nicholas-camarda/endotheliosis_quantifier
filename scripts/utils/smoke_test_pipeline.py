import os
from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model


def find_one_image_and_mask(images_root: Path, masks_root: Path):
    for patient_dir in sorted(images_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_masks = masks_root / patient_dir.name
        if not patient_masks.exists():
            continue
        for img_path in sorted(patient_dir.iterdir()):
            if not img_path.is_file():
                continue
            base = img_path.stem
            mask_path = patient_masks / f"{base}_mask.jpg"
            if mask_path.exists():
                return img_path, mask_path
    raise FileNotFoundError("No paired image/mask found under train directories")


def extract_one_roi(image_bgr: np.ndarray, mask_gray: np.ndarray, size: int = 256) -> np.ndarray:
    mask_gray = cv2.resize(mask_gray, (image_bgr.shape[1], image_bgr.shape[0]))
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_gray)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")
    x, y, w, h = cv2.boundingRect(contours[0])
    x = max(0, x - 5)
    y = max(0, y - 5)
    w = min(image_bgr.shape[1] - x, w + 10)
    h = min(image_bgr.shape[0] - y, h + 10)
    roi = masked[y:y+h, x:x+w]
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_CUBIC)
    return roi


def main() -> None:
    data_root = Path("data/Lauren_PreEclampsia_Data")
    images_root = data_root / "train" / "images"
    masks_root = data_root / "train" / "masks"
    img_path, msk_path = find_one_image_and_mask(images_root, masks_root)
    print(f"Using image: {img_path}")
    print(f"Using mask:  {msk_path}")

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(msk_path), 0)
    roi = extract_one_roi(img, mask, size=256)
    print(f"ROI shape: {roi.shape}")

    model_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = Model(inputs=model_base.input, outputs=model_base.layers[-1].output)

    batch = np.expand_dims(roi, axis=0).astype(np.float32)
    batch = preprocess_input(batch)
    feats = model.predict(batch, verbose=0)
    flat = feats.reshape((feats.shape[0], -1))
    print(f"Features shape: {feats.shape}; Flattened: {flat.shape}")
    print("Smoke test OK")


if __name__ == "__main__":
    main()
