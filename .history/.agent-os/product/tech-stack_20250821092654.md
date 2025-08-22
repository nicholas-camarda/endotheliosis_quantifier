# Tech Stack

## Languages & Runtimes
- Python 3.9

## Core Libraries
- TensorFlow/Keras (Apple Silicon): `tensorflow-macos`, `tensorflow-metal`
- Computer Vision: OpenCV (`cv2`), scikit-image (usage minimal), Pillow
- ML: scikit-learn, LightGBM, XGBoost
- Scientific: NumPy, SciPy, Matplotlib, Pandas
- TF Hub (optional): super-resolution model experiments

## Alternative Stack (to evaluate)
- PyTorch + FastAI for cross-platform training/inference (macOS MPS, Windows CUDA)

## Environment
- Conda environment: `eq` (see `environment.yml`)
- macOS Apple Silicon target for inference; Windows/WSL2 + CUDA (RTX 3080) for heavier training

## Models
- U-Net segmentation (transfer learning from EM-derived features)
- Feature extractor: ResNet50 (imagenet) for ROI features
- Quantifier: Random Forest regression (baseline), options for Bayesian Ridge and Neural Net
