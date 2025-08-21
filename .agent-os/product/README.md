# Endotheliosis Quantifier — Product Overview

## Vision
- Computer vision pipeline to automatically segment glomeruli and quantify glomerular endotheliosis on H&E slides.
- Transfer learning: initialize from mitochondria EM-trained features; fine-tune a U-Net for glomerular line structures.

## Target Users
- Kidney pathology and biomedical imaging researchers needing reproducible, automated endotheliosis quantification.

## Current State
- 4-step pipeline in `scripts/main/` (data prep → segmentation → feature extraction → quantification).
- Apple Silicon environment present (TensorFlow + Metal). Cross-platform support planned (macOS M1 for inference; Windows/WSL2 + CUDA for training).

## Start Here
- Roadmap: see `./roadmap.md`
- Tech Stack: see `./tech-stack.md`
- Key Decisions: see `./decisions.md`
- Technical Summary: see `./technical/README.md`
