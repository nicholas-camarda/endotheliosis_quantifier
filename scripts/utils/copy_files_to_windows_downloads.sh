#!/bin/bash

to_copy=output/segmentation_models/glomerulus_segmentation/2023-04-10-glom_unet_xfer_seg_model-epochs75_batch8/plots/predictions
desired_target_dir=/mnt/c/Users/ncama/Downloads 

cp -r $to_copy $desired_target_dir