import os.path
from pathlib import Path

import numpy as np
import torch
import tqdm
from detectron2.data.detection_utils import read_image

from pseudo_labeling.predictor import MasksFromBboxesPredictor
from utils.bbox_conversion import yolo_bboxes_to_pascal_voc
from utils.save import save_masks


def generate_masks_from_bboxes(cfg, data_folder, dest_folder, model_weights, img_ext):
    ids = [file.stem for file in Path(data_folder).glob("*.txt")]
    predictor = MasksFromBboxesPredictor(cfg, model_weights=model_weights)

    for img_id in tqdm.tqdm(ids):
        img_path = os.path.join(data_folder, f'{img_id}.{img_ext}')
        img = read_image(img_path, format="BGR")  # H, W, C
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Get bboxes and classes GT
        bboxes_path = os.path.join(data_folder, f'{img_id}.txt')
        bboxes = np.loadtxt(bboxes_path, delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 2:
            classes = bboxes[:, 0].astype(int).tolist()
            bboxes = bboxes[:, 1:]
        else:  # only 1 instance
            classes = [bboxes[0].astype(int)]
            bboxes = [bboxes[1:]]

        # Convert bboxes from YOLO format to Pascal VOC format
        bboxes = yolo_bboxes_to_pascal_voc(bboxes, img_height=img_height, img_width=img_width)

        predictions = predictor(img, bboxes, classes)

        instances = predictions["instances"].to(torch.device("cpu"))
        masks = instances.pred_masks.numpy()
        masks = np.array(masks).transpose((1, 2, 0))  # n x H x W -> H x W x n
        save_masks(dest_folder=dest_folder, filename=f'{img_id}.npz', masks=masks)
