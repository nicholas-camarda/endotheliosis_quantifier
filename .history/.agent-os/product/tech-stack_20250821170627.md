# Tech Stack

## Languages & Runtimes
- Python 3.9

## Core Libraries
- **conda env**: Remember, always start with `conda activate eq`. Install all packages and run all analysis in this `eq` env.
- **Package Management**: Always use `mamba` for faster and more reliable package installations (e.g., `mamba install -c conda-forge package_name`)
- **fastai/PyTorch** (Cross-platform): `fastai`, `torch`, `torchvision`, `torchaudio`
- **Backend Support**: MPS (Apple Silicon), CUDA (NVIDIA RTX 3080)
- Computer Vision: OpenCV (`cv2`), scikit-image (usage minimal), Pillow
- ML: scikit-learn, LightGBM, XGBoost
- Scientific: NumPy, SciPy, Matplotlib, Pandas

## Migration Status
- **COMPLETED 2025-08-21**: Fully migrated from TensorFlow to fastai/PyTorch
- **COMPLETED 2025-08-21**: All TensorFlow packages removed from environment
- **COMPLETED 2025-08-21**: Notebook implementations converted to Python modules
- **COMPLETED 2025-08-21**: Dual-environment architecture implemented with mode selection and backend abstraction
- **COMPLETED 2025-08-21**: Comprehensive fastai segmentation system implemented with U-Net support
- **COMPLETED 2025-08-21**: Mode-specific training strategies (production, development, auto) implemented
- **COMPLETED 2025-08-21**: Advanced data pipeline with cache integration and augmentation
- **RESOLVED**: OpenMP library conflicts eliminated
- Leveraging existing working implementations from notebooks folder
- Dual-environment architecture with explicit mode selection and hardware detection

## Environment
- Conda environment: `eq` (see `environment.yml`)
- macOS Apple Silicon target for inference; Windows/WSL2 + CUDA (RTX 3080) for heavier training

## Models
- **fastai U-Net segmentation** (transfer learning from EM-derived features)
  - Multiple architectures: ResNet18, 34, 50, 101 backbones
  - Mode-specific training strategies with hardware optimization
  - Advanced augmentation pipeline with cache integration
- **fastai ResNet50** (imagenet) for ROI features
- Quantifier: Random Forest regression (baseline), options for Bayesian Ridge and Neural Net

## Package & CLI
- `src/eq` package structure with modular organization
- CLI interface via `eq` command with subcommands for each pipeline step
- Console script entry point in `pyproject.toml`
- Development installation with `pip install -e .`
