import os

import cv2
import numpy as np

# Import shared functions from helpers to avoid duplication
from .helpers import load_pickled_data, plot_history, expand_scores, extract_features


def preprocess_images(image_folder, mask_folder, output_folder, padding=5):
    rois = []
    for patient_folder in os.listdir(image_folder):
        patient_image_folder = os.path.join(image_folder, patient_folder)
        patient_mask_folder = os.path.join(mask_folder, patient_folder)

        if not os.path.exists(patient_mask_folder):
            continue

        for filename_image, filename_mask in zip(sorted(os.listdir(patient_image_folder)), sorted(os.listdir(patient_mask_folder))):
            # Load image and mask
            image_file = os.path.join(patient_image_folder, filename_image)
            mask_file = os.path.join(patient_mask_folder, filename_mask)

            img = cv2.imread(image_file)
            mask = cv2.imread(mask_file, 0)

            # Resize image and mask if necessary
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filename = os.path.splitext(filename_image)[0]
            # Extract ROIs using contours and save them
            roi_index = 0
            for cnt in contours:
               # Get the minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                x, y, radius = int(x), int(y), int(radius)

                # Add padding to the circle
                radius += padding

                # Extract the circular ROI
                mask_roi = np.zeros_like(img)
                cv2.circle(mask_roi, (x, y), radius, (255, 255, 255), -1)
                roi = cv2.bitwise_and(img, mask_roi)
                roi = roi[y-radius:y+radius, x-radius:x+radius]

                # Check if the ROI is empty, skip if it is
                if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                print(image_file, mask_file)

                # Crop and resize the ROI
                roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_CUBIC)
                rois.append(roi)

                # Save the ROI as an image
                roi_output_folder = os.path.join(output_folder, patient_folder)
                if not os.path.exists(roi_output_folder):
                    os.makedirs(roi_output_folder)
                roi_output_path = os.path.join(roi_output_folder, f"{filename}_ROI_{roi_index}.jpg")
                cv2.imwrite(roi_output_path, roi)
                roi_index += 1

    return np.array(rois)

# Functions moved to eq.features.helpers module
# All side-effecting code removed for clean imports

