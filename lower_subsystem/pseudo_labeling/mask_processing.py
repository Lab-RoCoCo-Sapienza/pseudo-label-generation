import glob
import os
import time
from pathlib import Path
from skimage.segmentation import slic
import cv2
import numpy as np
import tqdm

from utils.bbox_conversion import yolo_bbox_to_pascal_voc
from utils.save import save_masks


def get_default_kernel():
    kernel = [[0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0]]
    return np.array(kernel).astype(np.uint8)


def mask_touches_bbox(mask, bbox, touches_all_edges=False):
    """
    Check if the mask touches the bounding box
    :param mask: Mask as np.array uint8
    :param bbox: Bounding box as np.array in Pascal VOC format (x_min, y_min, x_max, y_max)
    :param touches_all_edges: (default False)
    :return: If touches_all_edges=True then returns True if the mask touches all the edges of the bbox,
             else returns True if the mask touches at least one of the edges of the bbox
    """
    x = np.where(np.any(mask, axis=0))[0]
    y = np.where(np.any(mask, axis=1))[0]
    temp_bbox = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)
    if touches_all_edges:
        return (temp_bbox[0] <= bbox[0] and
                temp_bbox[1] <= bbox[1] and
                temp_bbox[2] >= bbox[2] and
                temp_bbox[3] >= bbox[3])
    else:
        return (temp_bbox[0] <= bbox[0] or
                temp_bbox[1] <= bbox[1] or
                temp_bbox[2] >= bbox[2] or
                temp_bbox[3] >= bbox[3])


def set_values_outside_bbox_to_zero(mask, bbox):
    """
    Set all the values of the mask outside the bounding box to zero
    :param mask: Mask as np.array uint8
    :param bbox: Bounding box as np.array in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    mask[:int(bbox[1]), :] = 0  # 0 to y_min-1
    mask[int(bbox[3]) + 1:, :] = 0  # y_max+1 to height
    mask[:, :int(bbox[0])] = 0  # 0 to x_min-1
    mask[:, int(bbox[2]) + 1:] = 0  # x_max+1 to width
    return mask


def dilate_pseudomasks(masks, bboxes):
    kernel = get_default_kernel()
    height = masks.shape[0]
    width = masks.shape[1]
    # Dilate the masks until they touch the edges of the bounding boxes
    for i in range(masks.shape[2]):
        if np.all((masks[:, :, i] == 0)):  # if empty mask
            continue
        abs_bbox = yolo_bbox_to_pascal_voc(bboxes[i], img_height=height, img_width=width)
        while not mask_touches_bbox(masks[:, :, i], abs_bbox, touches_all_edges=False):
            masks[:, :, i] = cv2.dilate(masks[:, :, i], kernel, iterations=1)
        masks[:, :, i] = set_values_outside_bbox_to_zero(masks[:, :, i], abs_bbox)
    return masks


def slic_pseudomasks(cfg, masks, bboxes, image_path):
    height = masks.shape[0]
    width = masks.shape[1]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Get slic segmentation
    slic_segmentation = slic(image, start_label=1, convert2lab=True,
                             slic_zero=cfg.PSEUDOMASKS.SLIC.SLIC_ZERO,
                             n_segments=cfg.PSEUDOMASKS.SLIC.N_SEGMENTS,
                             compactness=cfg.PSEUDOMASKS.SLIC.COMPACTNESS,
                             sigma=cfg.PSEUDOMASKS.SLIC.SIGMA)
    threshold = cfg.PSEUDOMASKS.SLIC.THRESHOLD

    for i in range(masks.shape[2]):  # for each mask
        mask = masks[:, :, i]
        mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        if np.all(mask == 0):  # if empty mask continue
            continue
        abs_bbox = yolo_bbox_to_pascal_voc(bboxes[i], img_height=height, img_width=width)

        for cluster_index in np.unique(slic_segmentation)[1:]:
            cluster = slic_segmentation == cluster_index
            intersection_area = np.sum((cluster * mask) > 0, axis=(0, 1))
            cluster_area = np.sum(cluster > 0, axis=(0, 1))

            if intersection_area / cluster_area > threshold:
                mask = ((cluster + mask) > 0).astype(np.uint8)
            if intersection_area / cluster_area < (1-threshold):
                mask = (mask - ((cluster * mask) > 0)).astype(np.uint8)

        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        masks[:, :, i] = set_values_outside_bbox_to_zero(mask, abs_bbox)

    return masks


def grabcut_pseudomasks(masks, bboxes, image_path, gamma_iters=40, median_blur=0):
    height = masks.shape[0]
    width = masks.shape[1]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kernel = get_default_kernel()

    for i in range(masks.shape[2]):  # for each mask
        mask = masks[:, :, i].copy()
        abs_bbox = yolo_bbox_to_pascal_voc(bboxes[i], img_height=height, img_width=width)

        # Dilation and erosion iterations proportional to the minimum size of the bounding box
        iters = int(min(abs_bbox[2] - abs_bbox[0], abs_bbox[3] - abs_bbox[1]) / gamma_iters)

        # Allocate memory for two arrays that GrabCut will use internally
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")

        if not np.all(mask == 0):
            new_mask = mask.copy()
            dilated_mask = cv2.dilate(mask, kernel, iterations=iters)
            eroded_mask = cv2.erode(mask, kernel, iterations=iters)
            new_mask[dilated_mask > 0] = cv2.GC_PR_BGD  # probable background
            new_mask[mask > 0] = cv2.GC_PR_FGD  # probable foreground
            new_mask[eroded_mask > 0] = cv2.GC_FGD  # foreground
        else:
            new_mask = np.full_like(mask, fill_value=cv2.GC_PR_FGD)  # probable foreground
        set_values_outside_bbox_to_zero(new_mask, abs_bbox)  # background

        # GrabCut with mask initialization
        (mask, bgModel, fgModel) = cv2.grabCut(image, new_mask, None, bgModel, fgModel,
                                               iterCount=1, mode=cv2.GC_INIT_WITH_MASK)

        # Set all background and probable background pixels to 0 and
        # set all foreground and probable foreground pixels to 1
        mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)

        if median_blur > 0:
            mask = cv2.medianBlur(mask, ksize=median_blur)

        masks[:, :, i] = mask

    return masks


def process_pseudomasks(cfg, method, masks_folder, data_path, output_path, img_ext):
    input_masks = glob.glob(os.path.expanduser(f'{masks_folder}/*.npz'))
    assert input_masks, "The input path(s) was not found"
    for path in tqdm.tqdm(input_masks):
        masks = np.load(path)['arr_0'].astype(np.uint8)  # H x W x n
        masks_id = os.path.basename(path)
        masks_id = os.path.splitext(masks_id)[0]

        bboxes = np.loadtxt(os.path.join(data_path, f'{masks_id}.txt'), delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
        bboxes = bboxes[:, 1:]  # remove classes

        if method == 'dilation':
            masks = dilate_pseudomasks(masks, bboxes)
        elif method == 'slic':
            image_path = os.path.join(data_path, f'{masks_id}.{img_ext}')
            masks = slic_pseudomasks(cfg, masks, bboxes, image_path)
        elif method == 'grabcut':
            image_path = os.path.join(data_path, f'{masks_id}.{img_ext}')
            masks = grabcut_pseudomasks(masks, bboxes, image_path,
                                        median_blur=cfg.PSEUDOMASKS.GRABCUT.MEDIAN_BLUR)

        # Save masks to file
        Path(output_path).mkdir(parents=True, exist_ok=True)
        save_masks(masks=masks, dest_folder=output_path, filename=f'{masks_id}.npz')