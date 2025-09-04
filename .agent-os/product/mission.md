# Product Mission

## Pitch

Endotheliosis Quantifier (EQ) is a medical image analysis pipeline that helps researchers and pathologists quantify endotheliosis in preeclampsia histology images by providing automated segmentation, transfer learning, and quantification capabilities.

## Users

### Primary Customers

- **Medical Researchers**: Scientists studying preeclampsia and kidney pathology
- **Pathologists**: Medical professionals analyzing histology samples
- **Computational Biologists**: Researchers developing AI tools for medical imaging

### Use Cases

- **Research Studies**: Quantifying endotheliosis in large cohorts of preeclampsia samples
- **Clinical Validation**: Validating AI models against expert annotations
- **Method Development**: Developing new quantification approaches for glomerular pathology

## The Problem

Preeclampsia is a serious pregnancy complication affecting 5-8% of pregnancies worldwide. Endotheliosis (swelling of blood vessel cells) in the kidneys is a key pathological feature, but manual quantification is:
- **Time-consuming**: Requires expert pathologists to manually analyze hundreds of images
- **Subjective**: Inter-observer variability in scoring (0-3 scale)
- **Limited scalability**: Cannot process large datasets efficiently
- **Inconsistent**: Different scoring methods across research groups

## Differentiators

- **Two-stage transfer learning**: Pre-trains on mitochondria EM data, then fine-tunes on glomeruli H&E images
- **End-to-end pipeline**: From raw images to quantified results in one workflow
- **Hardware-aware execution**: Automatic optimization for GPU, MPS, or CPU
- **Lab notebook transparency**: All parameters, results, and methodology clearly documented
- **Extensible architecture**: Modular design for easy adaptation to new datasets and models

## Key Features

1. **Mitochondria Pre-training**: Trains UNet segmentation on EM data for robust feature learning
2. **Glomeruli Transfer Learning**: Uses mitochondria model as base for H&E image segmentation
3. **Endotheliosis Quantification**: Converts segmented regions to ordinal (0-3) or continuous (0-3) scores
4. **Complete Pipeline**: Automated workflow from data loading to final quantification
5. **Hardware Optimization**: Automatic detection and optimization for available computing resources
6. **Transparent Configuration**: Central configuration file for all runtime parameters and paths
7. **Comprehensive Testing**: End-to-end validation with real medical data
8. **Lab Notebook Documentation**: Clear methodology, results, and reproducibility guidelines
