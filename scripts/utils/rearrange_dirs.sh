#!/bin/bash

mkdir -p train_images/train
mkdir -p train_masks/train
mkdir -p val_images/val
mkdir -p val_masks/val

mv train/image_patches/* train_images/train/
mv train/mask_patches/* train_masks/train/
mv val/image_patches/* val_images/val/
mv val/mask_patches/* val_masks/val/

rm -rf train
rm -rf val