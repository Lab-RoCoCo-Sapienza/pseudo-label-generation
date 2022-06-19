import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.structures import Instances, Boxes


class MasksFromBboxesPredictor:
    """
    Create a simple end-to-end mask predictor with the given config that runs on
    single device for a single input image with GT (Ground-Truth) bounding boxes and classes.

    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from model_weights if specified, otherwise from 'cfg.PSEUDOMASKS.INITIAL_WEIGHTS'.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
       As input it also requires the GT bboxes and classes that will be used to predict the masks.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST` to both image and labels
    4. Take one input image with the bounding boxes and classes and produce a single output, instead of a batch.
    """
    def __init__(self, cfg, model_weights):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST) > 0:
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        self.checkpointer = DetectionCheckpointer(self.model, cfg.OUTPUT_DIR)
        self.checkpointer.load(model_weights)
        
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def __call__(self, original_image, bboxes, classes):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            bboxes (list): List of bounding boxes (Nx4) in the Pascal VOC format.
            classes (list): List of categories for each bounding box
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            transform = self.aug.get_transform(original_image)

            # Transform the image
            image = transform.apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # C, H, W
            inputs = {"image": image, "height": height, "width": width}

            # Transform the GT bounding boxes
            bboxes = [transform.apply_box(np.array([bbox]))[0].clip(min=0) for bbox in bboxes]
            bboxes = torch.tensor(np.array(bboxes))

            # Create an 'Instances' object with the GT bboxes and classes
            target = Instances(image_size=image.shape[1:])
            target.pred_boxes = Boxes(bboxes)
            target.pred_classes = torch.tensor(classes, dtype=torch.int64)

            predictions = self.model.inference([inputs], detected_instances=[target])[0]
            return predictions
