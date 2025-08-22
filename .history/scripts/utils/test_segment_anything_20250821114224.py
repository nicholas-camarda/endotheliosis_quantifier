
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


path_to_image = 'data/preeclampsia_data/train/images/T19/T19_Image0.jpg'
image = cv2.imread(path_to_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()

# loadthe SAM model
sam_checkpoint = "output/segmentation_models/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "/GPU:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = torch.device("mps")
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)
