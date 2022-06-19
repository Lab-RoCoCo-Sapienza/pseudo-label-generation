from pathlib import Path
import cv2
import numpy as np
from pycocotools.mask import encode

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils.bbox_conversion import yolo_bboxes_to_pascal_voc


# Dataset
def get_target_dataset_dicts(data_folder, ext, pseudo_masks_folder):
    ids = [file.stem for file in data_folder.glob(f"*.{ext}")]

    dataset_dicts = []
    for img_id in ids:
        record = {}

        filename = str(data_folder / f'{img_id}.{ext}')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_id

        # Dimensions of the output of the model
        record["height"] = height
        record["width"] = width

        mask_path = pseudo_masks_folder / f'{img_id}.npz'
        masks = np.load(mask_path)['arr_0'].astype(np.uint8)

        # Remove empty masks
        indices_to_remove = []
        for i in range(masks.shape[2]):
            if np.all((masks[:, :, i] == 0)):
                indices_to_remove.append(i)
        if len(indices_to_remove) > 0:
            masks = np.delete(masks, indices_to_remove, axis=2)

        box_path = data_folder / f'{img_id}.txt'
        bboxes = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
        if bboxes.ndim == 2:
            bboxes = bboxes[:, 1:]
        else:  # only 1 instance
            bboxes = [bboxes[1:]]

        # Convert bboxes from YOLO format to Pascal VOC format
        bboxes = yolo_bboxes_to_pascal_voc(bboxes, img_height=height, img_width=width)

        # Remove bboxes corresponding to empty masks
        if len(indices_to_remove) > 0:
            bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in indices_to_remove]

        num_objs = masks.shape[2]
        assert (len(bboxes) == num_objs)

        objs = []
        for i in range(num_objs):
            obj = {
                "bbox": bboxes[i],
                "bbox_mode": BoxMode.XYXY_ABS,  # Pascal VOC bbox format
                "category_id": 0,
                "segmentation": encode(np.asarray(masks[:, :, i], order="F"))  # COCO’s compressed RLE format
            }
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)
    return dataset_dicts


def setup_target_dataset(dataset_name, data_folder, ext, pseudo_masks_folder):
    DatasetCatalog.register(dataset_name, get_target_dataset_dicts(data_folder, ext, pseudo_masks_folder))
    MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])
