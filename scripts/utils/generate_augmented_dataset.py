
import numpy as np
import tensorflow as tf


def cutout(images, masks, h=10, w=10):
    batch_size, height, width, _ = images.shape
    y1 = np.random.randint(0, height - h, batch_size)
    x1 = np.random.randint(0, width - w, batch_size)
    images_cut = np.copy(images)
    for i in range(batch_size):
        images_cut[i, y1[i]:y1[i]+h, x1[i]:x1[i]+w, :] = 0
    return images_cut, masks

# cutout_augmentation = tf.keras.layers.Lambda(lambda x: tf.numpy_function(cutout, [x[0], x[1]], [tf.float32, tf.float32]))


def mixup(images, masks, alpha=0.2):
    batch_size = images.shape[0]
    indices = np.random.permutation(batch_size)
    images_mix = np.copy(images)
    masks_mix = np.copy(masks)

    lam = np.random.beta(alpha, alpha, batch_size)[:, np.newaxis, np.newaxis, np.newaxis]
    images_mix = images * lam + images[indices] * (1 - lam)
    masks_mix = masks * lam + masks[indices] * (1 - lam)

    return images_mix, masks_mix

# mixup_augmentation = tf.keras.layers.Lambda(lambda x: tf.numpy_function(mixup, [x[0], x[1]], [tf.float32, tf.float32]))


def cutmix(images, masks, alpha=0.2):
    batch_size, height, width, _ = images.shape
    indices = np.random.permutation(batch_size)
    images_mix = np.copy(images)
    masks_mix = np.copy(masks)

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = get_random_bbox(height, width, lam)
    images_mix[:, bbx1:bbx2, bby1:bby2, :] = images[indices][:, bbx1:bbx2, bby1:bby2, :]
    masks_mix = masks * lam + masks[indices] * (1 - lam)

    return images_mix, masks_mix


def get_random_bbox(height, width, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(width * cut_rat)
    cut_h = np.int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


# cutmix_augmentation = tf.keras.layers.Lambda(lambda x: tf.numpy_function(cutmix, [x[0], x[1]], [tf.float32, tf.float32]))


def apply_augmentation(images, masks, cutout_prob=0.3, mixup_prob=0.3, cutmix_prob=0.3):
    choice = np.random.choice(["cutout", "mixup", "cutmix", "none"], p=[cutout_prob, mixup_prob, cutmix_prob, 1-cutout_prob-mixup_prob-cutmix_prob])
    if choice == "cutout":
        return cutout(images, masks)
    elif choice == "mixup":
        return mixup(images, masks)
    elif choice == "cutmix":
        return cutmix(images, masks)
    else:
        return images, masks


def augment(x, y):
    return tf.numpy_function(apply_augmentation, [x, y], [tf.float32, tf.float32])


# dataset = tf.data.Dataset.from_tensor_slices((images, masks))
# dataset = dataset.shuffle(buffer_size=len(images))
# dataset = dataset.batch(batch_size)
# dataset = dataset.map(augment)
