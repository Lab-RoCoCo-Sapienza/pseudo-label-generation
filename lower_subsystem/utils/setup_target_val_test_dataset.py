from pathlib import Path
import cv2
import numpy as np
from pycocotools.mask import encode

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


def extract_bboxes_from_masks(masks):
    boxes = np.zeros((masks.shape[2], 4), dtype=np.float32)
    x_any = np.any(masks, axis=0)
    y_any = np.any(masks, axis=1)
    for idx in range(masks.shape[2]):
        x = np.where(x_any[:, idx])[0]
        y = np.where(y_any[:, idx])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)

    return boxes


# Dataset
def get_target_val_test_dicts(root, source):
    # Load the dataset subset defined by source
    assert source in ['validation', 'test'], \
        'source should be "validation" or "test"'

    if source == "validation":
        source_path = Path(root, "validation")
    else:  # source == "test":
        source_path = Path(root, "test")

    ids = [file.stem for file in source_path.glob("*.jpg")]

    dataset_dicts = []
    for img_id in ids:
        record = {}

        filename = str(source_path / f'{img_id}.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_id

        # Dimensions of the output of the model
        record["height"] = height
        record["width"] = width

        mask_path = source_path / f'{img_id}.npz'
        masks = np.load(mask_path)['arr_0'].astype(np.uint8)

        bboxes = extract_bboxes_from_masks(masks)  # Pascal VOC format

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


def setup_target_val_test_dataset():
    data_path = "./datasets/target_dataset"
    for name in ["validation", "test"]:
        dataset_name = "target_dataset_" + name
        DatasetCatalog.register(dataset_name, lambda d=name: get_target_val_test_dicts(data_path, d))
        MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])
