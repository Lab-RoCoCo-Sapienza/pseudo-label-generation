import os
import cv2
import numpy as np
from pycocotools.mask import encode

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils.bbox_conversion import yolo_bbox_to_pascal_voc


# Dataset
def get_wgisd_dicts(root, source):
    # Load the dataset subset defined by source
    assert source in ['train', 'valid', 'test'], 'Source should be "train", "valid" or "test"'

    if source == "train":
        source_path = os.path.join(root, 'train_split_masked.txt')
    elif source == "valid":
        source_path = os.path.join(root, 'valid_split_masked.txt')
    else:  # source == "test"
        source_path = os.path.join(root, 'test_masked.txt')
    root = os.path.join(root, "data")

    with open(source_path, 'r') as fp:
        # Read all lines in file
        lines = fp.readlines()
        # Recover the items ids, removing the \n at the end
        ids = [l.rstrip() for l in lines]

    dataset_dicts = []
    for img_id in ids:
        record = {}

        filename = os.path.join(root, f'{img_id}.jpg')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = img_id

        # Dimensions of the output of the model
        record["height"] = height
        record["width"] = width

        box_path = os.path.join(root, f'{img_id}.txt')
        bboxes = np.loadtxt(box_path, delimiter=" ", dtype=np.float32)
        bboxes = bboxes[:, 1:]
        num_objs = bboxes.shape[0]

        mask_path = os.path.join(root, f'{img_id}.npz')
        masks = np.load(mask_path)['arr_0'].astype(np.uint8)
        assert (masks.shape[2] == num_objs)

        objs = []
        for i in range(num_objs):
            # Convert bboxes from YOLO format to Pascal VOC format
            box = yolo_bbox_to_pascal_voc(bboxes[i], img_height=height, img_width=width)

            obj = {
                "bbox": box,
                "bbox_mode": BoxMode.XYXY_ABS,  # Pascal VOC bbox format
                "category_id": 0,
                obj["segmentation"]: encode(np.asarray(masks[:, :, i], order="F"))  # COCO’s compressed RLE format
            }                
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)
    return dataset_dicts


def setup_source_dataset():
    data_path = "./datasets/wgisd"

    for name in ["train"]:  # ["train", "valid", "test"]:
        dataset_name = "wgisd_" + name
        DatasetCatalog.register(dataset_name, lambda d=name: get_wgisd_dicts(data_path, d))
        MetadataCatalog.get(dataset_name).set(thing_classes=["grapes"])
