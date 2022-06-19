def pascal_voc_bbox_to_albumentations(bbox, height, width):
    """
    Convert a bounding box from the pascal_voc format to the albumentations format:
    normalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.
    :param bbox: Denormalized (pascal_voc format) bounding box `(x_min, y_min, x_max, y_max)`.
    :param height: (int) Image height.
    :param width: (int) Image width.
    :return: Normalized (albumentations format) bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox

    # Normalize
    x_min /= width
    x_max /= width
    y_min /= height
    y_max /= height

    # Fix for floating point precision issue
    if -0.001 < x_min < 0.0:
        x_min = 0.0
    if -0.001 < y_min < 0.0:
        y_min = 0.0
    if 1 < x_max < 1.001:
        x_max = 1.0
    if 1 < y_max < 1.001:
        y_max = 1.0

    # Check that the bbox is in the range [0.0, 1.0]
    assert 0 <= x_min <= 1, "Expected bbox to be in the range [0.0, 1.0], got x_min = {x_min}.".format(x_min=x_min)
    assert 0 <= y_min <= 1, "Expected bbox to be in the range [0.0, 1.0], got y_min = {y_min}.".format(y_min=y_min)
    assert 0 <= x_max <= 1, "Expected bbox to be in the range [0.0, 1.0], got x_max = {x_max}.".format(x_max=x_max)
    assert 0 <= y_max <= 1, "Expected bbox to be in the range [0.0, 1.0], got y_max = {y_max}.".format(y_max=y_max)

    return x_min, y_min, x_max, y_max


def albumentations_bbox_to_pascal_voc(bbox, height, width):
    """
    Convert a bounding box from the albumentations format to the pascal_voc format:
    denormalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)`.
    :param bbox: Normalized (albumentations format) bounding box `(x_min, y_min, x_max, y_max)`.
    :param height: (int) Image height.
    :param width: (int) Image width.
    :return: Denormalized (pascal_voc format) bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bbox

    # Denormalize
    x_min *= width
    x_max *= width
    y_min *= height
    y_max *= height

    # Check that the bbox is in the range [0.0, 0.0, height, width]
    assert 0 <= x_min <= x_max, \
        "Expected x_min to be in the range [0.0, x_max], got x_min = {x_min}.".format(x_min=x_min)
    assert 0 <= y_min <= y_max, \
        "Expected y_min to be in the range [0.0, y_max], got y_min = {y_min}.".format(y_min=y_min)
    assert x_min <= x_max <= width, \
        "Expected x_max to be in the range [x_min, width], got x_max = {x_max}.".format(x_max=x_max)
    assert y_min <= y_max <= height, \
        "Expected y_max to be in the range [y_min, height], got y_max = {y_max}.".format(y_max=y_max)

    return x_min, y_min, x_max, y_max


def yolo_bbox_to_albumentations(bbox):
    """
    Convert a bounding box from the yolo format to the albumentations format.
    :param bbox: Normalized (yolo format) bounding box `(x_0, y_0, box_width, box_height)`.
    :return: Normalized (albumentations format) bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_0, y_0, box_width, box_height = bbox
    x_min = x_0 - box_width / 2.0
    x_max = x_0 + box_width / 2.0
    y_min = y_0 - box_height / 2.0
    y_max = y_0 + box_height / 2.0

    # # Fix floating point precision issue
    # if -0.001 < x_min < 0.0:
    #     x_min = 0.0
    # if -0.001 < y_min < 0.0:
    #     y_min = 0.0
    # if 1 < x_max < 1.001:
    #     x_max = 1.0
    # if 1 < y_max < 1.001:
    #     y_max = 1.0

    # Crop bounding boxes
    if x_min < 0.0:
        x_min = 0.0
    if y_min < 0.0:
        y_min = 0.0
    if 1 < x_max:
        x_max = 1.0
    if 1 < y_max:
        y_max = 1.0

    # Check that the bbox is in the range [0.0, 1.0]
    assert 0 <= x_min <= 1, "Expected bbox to be in the range [0.0, 1.0], got x_min = {x_min}.".format(x_min=x_min)
    assert 0 <= y_min <= 1, "Expected bbox to be in the range [0.0, 1.0], got y_min = {y_min}.".format(y_min=y_min)
    assert 0 <= x_max <= 1, "Expected bbox to be in the range [0.0, 1.0], got x_max = {x_max}.".format(x_max=x_max)
    assert 0 <= y_max <= 1, "Expected bbox to be in the range [0.0, 1.0], got y_max = {y_max}.".format(y_max=y_max)

    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def albumentations_bbox_to_yolo(bbox):
    """
    Convert a bounding box from the albumentations format to the yolo format.
    :param bbox: Normalized (albumentations format) bounding box `(x_min, y_min, x_max, y_max)`.
    :return: Normalized (yolo format) bounding box `(x_0, y_0, box_width, box_height)`.
    """
    x_min, y_min, x_max, y_max = bbox
    x_0 = (x_min + x_max) / 2.0
    y_0 = (y_min + y_max) / 2.0
    box_width = x_max - x_min
    box_height = y_max - y_min
    bbox = [x_0, y_0, box_width, box_height]
    return bbox


def pascal_voc_bbox_to_yolo(bbox, img_height, img_width):
    """
    Convert a bounding box from the pascal_voc format to the yolo format.
    :param bbox: Denormalized (pascal_voc format) bounding box `(x_min, y_min, x_max, y_max)`.
    :param img_width: Width of the image
    :param img_height: Height of the image
    :return: Normalized (yolo format) bounding box `(x_0, y_0, box_width, box_height)`.
    """
    x_min, y_min, x_max, y_max = bbox

    # Normalize
    x_min /= img_width
    x_max /= img_width
    y_min /= img_height
    y_max /= img_height

    return albumentations_bbox_to_yolo([x_min, y_min, x_max, y_max])


def yolo_bbox_to_pascal_voc(bbox, img_height, img_width):
    """
    Convert a bounding box from the pascal_voc format to the yolo format.
    :param bbox: Normalized (yolo format) bounding box `(x_0, y_0, box_width, box_height)`.
    :param img_width: Width of the image
    :param img_height: Height of the image
    :return: Denormalized (pascal_voc format) bounding box `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = yolo_bbox_to_albumentations(bbox)

    # Denormalize
    x_min *= img_width
    x_max *= img_width
    y_min *= img_height
    y_max *= img_height

    return [x_min, y_min, x_max, y_max]


def pascal_voc_bboxes_to_albumentations(bboxes, height, width):
    """Convert a list of bounding boxes from the pascal_voc format to the albumentations format"""
    return [pascal_voc_bbox_to_albumentations(bbox, height, width) for bbox in bboxes]


def albumentations_bboxes_to_pascal_voc(bboxes, height, width):
    """Convert a list of bounding boxes from the albumentations format to the pascal_voc format"""
    return [albumentations_bbox_to_pascal_voc(bbox, height, width) for bbox in bboxes]


def yolo_bboxes_to_albumentations(bboxes):
    """Convert a list of bounding boxes from the yolo format to the albumentations format"""
    return [yolo_bbox_to_albumentations(bbox) for bbox in bboxes]


def albumentations_bboxes_to_yolo(bboxes):
    """Convert a list of bounding boxes from the albumentations format to the yolo format"""
    return [albumentations_bbox_to_yolo(bbox) for bbox in bboxes]


def pascal_voc_bboxes_to_yolo(bboxes, img_height, img_width):
    """Convert a list of bounding boxes from the pascal_voc format to the yolo format"""
    return [pascal_voc_bbox_to_yolo(bbox, img_height, img_width) for bbox in bboxes]


def yolo_bboxes_to_pascal_voc(bboxes, img_height, img_width):
    """Convert a list of bounding boxes from the yolo format to the pascal_voc format"""
    return [yolo_bbox_to_pascal_voc(bbox, img_height, img_width) for bbox in bboxes]
